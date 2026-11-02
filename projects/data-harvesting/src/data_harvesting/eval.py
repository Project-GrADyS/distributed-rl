from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import re
from typing import Any

import mlflow
import torch
from tensordict.nn import ProbabilisticTensorDictModule
from rl_core import CategoricalMetricSpec, ScalarMetricSpec
from mlflow import pytorch as mlflow_pytorch
from mlflow import MlflowClient
from torchrl.collectors import MultiSyncCollector
from torchrl.data import Unbounded
from torchrl.envs import Transform, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from data_harvesting.environment import (
    evaluation_environment_overrides,
    make_env,
    make_metrics_spec,
)
from data_harvesting.utils import _FasterTanhDelta, collapse_shared_action_bound


@dataclass(frozen=True)
class LoggedPolicyModel:
    name: str
    model_id: str
    creation_timestamp: int | None = None
    step: int | None = None
    kind: str = "other"
    step_inferred: bool = False


_CHECKPOINT_MODEL_NAME = re.compile(r"policy_checkpoint_step_(\d+)")
_EVALUATION_RUN_INDEX_KEY = "evaluation_run_index"


class _EvaluationEpisodeTransform(Transform):
    """Assign a stable run index and seed to every automatically reset episode."""

    def __init__(
        self,
        *,
        worker_index: int,
        num_workers: int,
        seed: int | None,
    ) -> None:
        super().__init__()
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.seed = seed
        self.episode_index = 0
        self.current_run_index = worker_index

    def _reset_env_preprocess(self, tensordict):
        self.current_run_index = (
            self.worker_index + self.episode_index * self.num_workers
        )
        self.episode_index += 1
        if self.seed is not None:
            # Calling set_seed before the collector reset matches the existing
            # serial evaluation's set_seed() followed by rollout() semantics.
            self.parent.base_env.set_seed(self.seed + self.current_run_index)
        return tensordict

    def _set_run_index(self, tensordict):
        shape = (*tensordict.batch_size, 1)
        return tensordict.set(
            _EVALUATION_RUN_INDEX_KEY,
            torch.full(
                shape,
                self.current_run_index,
                dtype=torch.int64,
                device=tensordict.device,
            ),
        )

    def _call(self, tensordict):
        return self._set_run_index(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._set_run_index(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec[_EVALUATION_RUN_INDEX_KEY] = Unbounded(
            shape=(1,),
            dtype=torch.int64,
            device=observation_spec.device,
        )
        return observation_spec


def _logged_policy_model_metadata(
    *,
    name: str,
    model_id: str,
    creation_timestamp: int | None,
    final_step: int | None,
) -> LoggedPolicyModel:
    checkpoint_match = _CHECKPOINT_MODEL_NAME.fullmatch(name)
    if checkpoint_match is not None:
        return LoggedPolicyModel(
            name=name,
            model_id=model_id,
            creation_timestamp=creation_timestamp,
            step=int(checkpoint_match.group(1)),
            kind="checkpoint",
        )
    if name == "policy_model":
        return LoggedPolicyModel(
            name=name,
            model_id=model_id,
            creation_timestamp=creation_timestamp,
            step=final_step,
            kind="final",
            step_inferred=final_step is not None,
        )
    return LoggedPolicyModel(
        name=name,
        model_id=model_id,
        creation_timestamp=creation_timestamp,
    )


def _latest_metric_step(client: MlflowClient, run) -> int | None:
    metric_names = sorted(getattr(getattr(run, "data", None), "metrics", {}))
    latest_step: int | None = None
    for metric_name in metric_names:
        for point in client.get_metric_history(run.info.run_id, metric_name):
            latest_step = point.step if latest_step is None else max(latest_step, point.step)
    return latest_step


def _scalar_specs(metrics_spec):
    return tuple(spec for spec in metrics_spec if isinstance(spec, ScalarMetricSpec))


def _categorical_specs(metrics_spec):
    return tuple(spec for spec in metrics_spec if isinstance(spec, CategoricalMetricSpec))


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    data = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
    }


def _scenario_key(num_agents: int, num_sensors: int) -> str:
    return f"agents_{num_agents}__sensors_{num_sensors}"


def _empty_categorical_counts(metrics_spec) -> dict[str, dict[str, int]]:
    return {
        metric.resolved_output_prefix: {
            label: 0 for label in metric.value_labels.values()
        }
        for metric in _categorical_specs(metrics_spec)
    }


def _empty_scenario_bucket(metrics_spec, *, num_agents: int, num_sensors: int) -> dict[str, Any]:
    return {
        "scenario": {"agents": num_agents, "sensors": num_sensors},
        "num_runs": 0,
        "scalar_samples": {
            metric.key: []
            for metric in _scalar_specs(metrics_spec)
        },
        "categorical_counts": _empty_categorical_counts(metrics_spec),
    }


def _get_episode_scenario(episode_info) -> tuple[int, int]:
    try:
        num_agents = int(float(episode_info["num_agents"]))
        num_sensors = int(float(episode_info["num_sensors"]))
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(
            f"Evaluation requires '{missing_key}' in terminal agents.info to group scenario metrics."
        ) from exc
    return num_agents, num_sensors


def _finalize_scenario_metrics(scenarios: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, bucket in scenarios.items():
        num_runs = bucket["num_runs"]
        scenario_result = {
            "scenario": bucket["scenario"],
            "num_runs": num_runs,
            "metrics": {
                metric_name: _metric_stats(values)
                for metric_name, values in bucket["scalar_samples"].items()
            },
        }
        for prefix, counts in bucket["categorical_counts"].items():
            scenario_result[f"{prefix}_counts"] = counts
            scenario_result[f"{prefix}_rate"] = {
                label: (count / num_runs if num_runs else 0.0)
                for label, count in counts.items()
            }
        finalized[key] = scenario_result
    return finalized


def _resolve_model_id_from_run(
    run_id: str,
    *,
    model_name: str = "policy_model",
) -> str:
    client = MlflowClient()
    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id

    models = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if not models:
        raise ValueError(f"No logged model was found for run '{run_id}'.")

    preferred = [model for model in models if model.name == model_name]
    candidates = preferred if preferred else models
    candidates.sort(key=lambda item: item.creation_timestamp or 0, reverse=True)
    return candidates[0].model_id


def load_config_from_mlflow_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run = MlflowClient().get_run(run_id)
    config: dict[str, Any] = {}
    for key, value in run.data.params.items():
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value

        if "." not in key:
            config[key] = parsed_value
            continue

        section = config
        parts = key.split(".")
        for part in parts[:-1]:
            section = section.setdefault(part, {})
        section[parts[-1]] = parsed_value

    if "environment" not in config:
        raise ValueError(
            f"Run '{run_id}' does not include a logged environment config; "
            "pass --params to evaluate with a local file."
        )

    return config


def list_policy_models_from_mlflow_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
) -> list[LoggedPolicyModel]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    run = client.get_run(run_id)
    models = list(
        client.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
        )
    )

    if not models:
        raise ValueError(f"No logged models were found for run '{run_id}'.")

    final_step = (
        _latest_metric_step(client, run)
        if any(model.name == "policy_model" for model in models)
        else None
    )
    policy_models = [
        _logged_policy_model_metadata(
            name=model.name,
            model_id=model.model_id,
            creation_timestamp=model.creation_timestamp,
            final_step=final_step,
        )
        for model in models
    ]
    policy_models.sort(
        key=lambda model: (
            model.creation_timestamp or 0,
            model.name,
            model.model_id,
        )
    )
    return policy_models


def _repair_legacy_probabilistic_modules(policy):
    """Repair older pickled modules and remove repeated agent action bounds."""

    modules = policy.modules() if hasattr(policy, "modules") else ()
    for module in modules:
        if not isinstance(module, ProbabilisticTensorDictModule):
            continue
        if not hasattr(module, "_generator"):
            module._generator = None
        if not hasattr(module, "_generator_key"):
            module._generator_key = None
        distribution_class = module.distribution_class
        if isinstance(distribution_class, type) and issubclass(distribution_class, _FasterTanhDelta):
            for key in ("low", "high"):
                if key in module.distribution_kwargs:
                    module.distribution_kwargs[key] = collapse_shared_action_bound(
                        module.distribution_kwargs[key]
                    )
    return policy


def load_policy_from_mlflow_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
    model_name: str = "policy_model",
):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_id = _resolve_model_id_from_run(run_id, model_name=model_name)
    model_uri = f"models:/{model_id}"
    policy = _repair_legacy_probabilistic_modules(
        mlflow_pytorch.load_model(model_uri)
    )
    return policy, model_id


def load_policy_from_model_id(model_id: str, *, tracking_uri: str | None = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return _repair_legacy_probabilistic_modules(
        mlflow_pytorch.load_model(f"models:/{model_id}")
    )


def _episode_row(episode_info, *, run_index: int, metrics_spec) -> dict[str, Any]:
    num_agents, num_sensors = _get_episode_scenario(episode_info)
    row: dict[str, Any] = {
        "run_index": run_index,
        "scenario_key": _scenario_key(num_agents, num_sensors),
        "num_agents": num_agents,
        "num_sensors": num_sensors,
    }
    for metric in metrics_spec:
        if isinstance(metric, ScalarMetricSpec):
            row[metric.key] = float(episode_info[metric.key])
            continue

        value = int(float(episode_info[metric.key]))
        row[metric.key] = value
        label = metric.value_labels.get(value)
        if label is not None:
            row[f"{metric.resolved_output_prefix}_label"] = label
    return row


def _serial_episode_rows(
    policy,
    eval_config: dict[str, Any],
    num_runs: int,
    *,
    seed: int | None,
    metrics_spec,
) -> list[dict[str, Any]]:
    env = make_env(eval_config)
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for run_index in range(num_runs):
                if seed is not None:
                    env.set_seed(seed + run_index)

                rollout = env.rollout(
                    max_steps=eval_config["environment"]["max_episode_length"],
                    policy=policy,
                )
                episode_info = rollout.get(("next", "agents", "info"))[-1, 0]
                rows.append(
                    _episode_row(
                        episode_info,
                        run_index=run_index,
                        metrics_spec=metrics_spec,
                    )
                )
    finally:
        env.close()
    return rows


def _make_parallel_eval_env(
    eval_config: dict[str, Any],
    *,
    worker_index: int,
    num_workers: int,
    seed: int | None,
):
    return TransformedEnv(
        base_env=make_env(eval_config),
        transform=_EvaluationEpisodeTransform(
            worker_index=worker_index,
            num_workers=num_workers,
            seed=seed,
        ),
    )


def _parallel_episode_rows(
    policy,
    eval_config: dict[str, Any],
    num_runs: int,
    *,
    num_workers: int,
    seed: int | None,
    metrics_spec,
) -> list[dict[str, Any]]:
    max_steps = int(eval_config["environment"]["max_episode_length"])
    env_factories = [
        partial(
            _make_parallel_eval_env,
            eval_config,
            worker_index=worker_index,
            num_workers=num_workers,
            seed=seed,
        )
        for worker_index in range(num_workers)
    ]
    collector = MultiSyncCollector(
        env_factories,
        policy,
        frames_per_batch=max_steps * num_workers,
        total_frames=-1,
        max_frames_per_traj=max_steps,
        exploration_type=ExplorationType.MODE,
        trajs_per_batch=num_workers,
        num_sub_threads=1,
    )
    rows_by_index: dict[int, dict[str, Any]] = {}
    try:
        for trajectories in collector:
            masks = trajectories.get(("collector", "mask"))
            for trajectory, mask in zip(
                trajectories.unbind(0), masks.unbind(0), strict=True
            ):
                terminal_index = int(mask.sum().item()) - 1
                run_index = int(
                    trajectory.get(("next", _EVALUATION_RUN_INDEX_KEY))[
                        terminal_index
                    ].item()
                )
                if run_index >= num_runs or run_index in rows_by_index:
                    continue
                episode_info = trajectory.get(("next", "agents", "info"))[
                    terminal_index, 0
                ]
                rows_by_index[run_index] = _episode_row(
                    episode_info,
                    run_index=run_index,
                    metrics_spec=metrics_spec,
                )

            if len(rows_by_index) == num_runs:
                break
    finally:
        collector.shutdown()

    return [rows_by_index[index] for index in range(num_runs)]


def _summarize_episode_rows(
    episode_rows: list[dict[str, Any]],
    metrics_spec,
) -> dict[str, Any]:
    scalar_samples: dict[str, list[float]] = {
        metric.key: [] for metric in _scalar_specs(metrics_spec)
    }
    categorical_counts = _empty_categorical_counts(metrics_spec)
    scenario_buckets: dict[str, dict[str, Any]] = {}

    for row in episode_rows:
        scenario_key = row["scenario_key"]
        scenario_bucket = scenario_buckets.setdefault(
            scenario_key,
            _empty_scenario_bucket(
                metrics_spec,
                num_agents=row["num_agents"],
                num_sensors=row["num_sensors"],
            ),
        )
        scenario_bucket["num_runs"] += 1

        for metric in metrics_spec:
            if isinstance(metric, ScalarMetricSpec):
                value = float(row[metric.key])
                scalar_samples[metric.key].append(value)
                scenario_bucket["scalar_samples"][metric.key].append(value)
                continue

            value = int(row[metric.key])
            label = metric.value_labels.get(value)
            if label is not None:
                prefix = metric.resolved_output_prefix
                categorical_counts[prefix][label] += 1
                scenario_bucket["categorical_counts"][prefix][label] += 1

    results: dict[str, Any] = {
        "num_runs": len(episode_rows),
        "metrics": {
            key: _metric_stats(values) for key, values in scalar_samples.items()
        },
        "scenario_metrics": _finalize_scenario_metrics(scenario_buckets),
        "episodes": episode_rows,
    }
    for prefix, counts in categorical_counts.items():
        results[f"{prefix}_counts"] = counts
        results[f"{prefix}_rate"] = {
            label: count / len(episode_rows) for label, count in counts.items()
        }
    return results


def eval(
    policy,
    config: dict[str, Any],
    num_runs: int,
    *,
    visual: bool = False,
    seed: int | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    if num_runs <= 0:
        raise ValueError("num_runs must be greater than 0")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be an integer")
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")
    if visual and num_workers != 1:
        raise ValueError("visual evaluation requires num_workers=1")

    eval_config = deepcopy(config)
    env_config = eval_config.setdefault("environment", {})
    env_config.update(evaluation_environment_overrides(eval_config))
    env_config["render_mode"] = "visual" if visual else None

    metrics_spec = make_metrics_spec()

    if hasattr(policy, "eval"):
        policy.eval()

    resolved_num_workers = min(num_workers, num_runs)
    if resolved_num_workers == 1:
        episode_rows = _serial_episode_rows(
            policy,
            eval_config,
            num_runs,
            seed=seed,
            metrics_spec=metrics_spec,
        )
    else:
        episode_rows = _parallel_episode_rows(
            policy,
            eval_config,
            num_runs,
            num_workers=resolved_num_workers,
            seed=seed,
            metrics_spec=metrics_spec,
        )
    return _summarize_episode_rows(episode_rows, metrics_spec)
