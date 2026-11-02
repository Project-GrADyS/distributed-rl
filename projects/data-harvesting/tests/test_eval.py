from functools import partial
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from data_harvesting.actor import create_actor
from data_harvesting.environment import make_env
from data_harvesting.eval import eval as run_eval
from data_harvesting.eval import list_policy_models_from_mlflow_run
from data_harvesting.eval import load_config_from_mlflow_run
from data_harvesting.eval import load_policy_from_model_id


def _make_policy() -> TensorDictSequential:
    zero_key = ("agents", "zero_action_component")
    return TensorDictSequential(
        TensorDictModule(
            module=partial(torch.zeros_like, dtype=torch.float32),
            in_keys=[("agents", "mask")],
            out_keys=[zero_key],
        ),
        TensorDictModule(
            module=partial(torch.repeat_interleave, repeats=2, dim=-1),
            in_keys=[zero_key],
            out_keys=[zero_key],
        ),
        TensorDictModule(
            module=nn.Unflatten(-1, (-1, 2)),
            in_keys=[zero_key],
            out_keys=[("agents", "action")],
        ),
    )


def _eval_config() -> dict:
    return {
        "flex_encoder": {"enabled": True},
        "environment": {
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 1,
            "max_num_agents": 1,
            "min_num_sensors": 1,
            "max_num_sensors": 1,
            "scenario_size": 10.0,
            "max_episode_length": 3,
            "max_seconds_stalled": 1,
            "communication_range": 0.0,
            "state_num_closest_sensors": 1,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        }
    }


def test_eval_summarizes_dynamic_scalar_and_categorical_metrics() -> None:
    results = run_eval(_make_policy(), _eval_config(), num_runs=3, seed=100)

    assert results["num_runs"] == 3
    assert "avg_reward" in results["metrics"]
    assert "completion_time" in results["metrics"]
    assert results["metrics"]["avg_reward"]["mean"] == pytest.approx(-1.0)
    assert results["metrics"]["avg_reward"]["std"] == pytest.approx(0.0)
    assert results["metrics"]["episode_duration"]["mean"] == pytest.approx(2.0)
    assert results["metrics"]["completion_time"]["mean"] == pytest.approx(3.0)
    assert results["end_cause_counts"]["STALLED"] == 3
    assert results["end_cause_rate"]["STALLED"] == pytest.approx(1.0)
    assert results["end_cause_counts"]["ALL_COLLECTED"] == 0
    assert "scenario_metrics" in results

    scenario_results = results["scenario_metrics"]["agents_1__sensors_1"]
    assert scenario_results["scenario"] == {"agents": 1, "sensors": 1}
    assert scenario_results["num_runs"] == 3
    assert scenario_results["metrics"]["avg_reward"]["mean"] == pytest.approx(-1.0)
    assert scenario_results["metrics"]["completion_time"]["mean"] == pytest.approx(3.0)
    assert scenario_results["end_cause_counts"]["STALLED"] == 3
    assert scenario_results["end_cause_rate"]["STALLED"] == pytest.approx(1.0)


def test_parallel_eval_matches_serial_seeded_episodes() -> None:
    config = _eval_config()
    config["environment"].update(
        max_num_agents=2,
        max_num_sensors=2,
    )

    serial = run_eval(
        _make_policy(),
        config,
        num_runs=5,
        seed=200,
        num_workers=1,
    )
    parallel = run_eval(
        _make_policy(),
        config,
        num_runs=5,
        seed=200,
        num_workers=2,
    )

    assert parallel == serial


@pytest.mark.parametrize("population", [1, 4, 8])
def test_eval_saved_mlp_with_smaller_population(population: int) -> None:
    config = _eval_config()
    config["flex_encoder"]["enabled"] = False
    config["actor"] = {
        "share_parameters": True,
        "centralized": False,
        "network_depth": 1,
        "network_width": 16,
        "activation_function": "Tanh",
    }
    config["environment"].update(min_num_agents=1, max_num_agents=8)
    env = make_env(config)
    try:
        actor = create_actor(env, torch.device("cpu"), config)
        buffer = BytesIO()
        torch.save(actor, buffer)
        buffer.seek(0)
        actor = torch.load(buffer, weights_only=False)
    finally:
        env.close()

    config["environment"].update(
        min_num_agents=population,
        max_num_agents=population,
    )
    serial = run_eval(actor, config, num_runs=2, seed=200, num_workers=1)
    parallel = run_eval(actor, config, num_runs=2, seed=200, num_workers=2)

    assert parallel == serial
    assert [row["num_agents"] for row in serial["episodes"]] == [population] * 2
    assert actor.module[0].module.mlp.n_agents == 8


@pytest.mark.parametrize("num_workers", [0, -1])
def test_eval_rejects_non_positive_worker_count(num_workers: int) -> None:
    with pytest.raises(ValueError, match="num_workers must be greater than 0"):
        run_eval(
            _make_policy(),
            _eval_config(),
            num_runs=1,
            num_workers=num_workers,
        )


def test_eval_rejects_parallel_visual_evaluation() -> None:
    with pytest.raises(ValueError, match="visual evaluation requires num_workers=1"):
        run_eval(
            _make_policy(),
            _eval_config(),
            num_runs=2,
            visual=True,
            num_workers=2,
        )


def test_eval_applies_environment_overrides_without_mutating_source(monkeypatch) -> None:
    captured_configs: list[dict] = []

    class _FakeRollout:
        def get(self, key):
            assert key == ("next", "agents", "info")
            return TensorDict(
                {
                    "num_agents": torch.tensor([[1.0]]),
                    "num_sensors": torch.tensor([[1.0]]),
                },
                batch_size=[1, 1],
            )

    class _FakeEnv:
        def rollout(self, *, max_steps: int, policy: nn.Module):
            return _FakeRollout()

        def close(self) -> None:
            return None

    def _make_env(config: dict) -> _FakeEnv:
        captured_configs.append(config)
        return _FakeEnv()

    monkeypatch.setattr("data_harvesting.eval.make_env", _make_env)
    monkeypatch.setattr("data_harvesting.eval.make_metrics_spec", lambda: ())

    source_config = _eval_config()
    run_eval(_make_policy(), source_config, num_runs=1)

    assert captured_configs[0]["environment"]["end_when_all_collected"] is True
    assert source_config["environment"]["end_when_all_collected"] is False


def test_load_config_from_mlflow_run_parses_nested_and_dotted_params(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            assert run_id == "run-123"
            return SimpleNamespace(
                data=SimpleNamespace(
                    params={
                        "environment": "{'max_episode_length': 50, 'sequential_obs': True}",
                        "training.algorithm": "maddpg",
                        "evaluation.seed": "None",
                        "label": "experiment-a",
                    }
                )
            )

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    config = load_config_from_mlflow_run("run-123")

    assert config == {
        "environment": {"max_episode_length": 50, "sequential_obs": True},
        "training": {"algorithm": "maddpg"},
        "evaluation": {"seed": None},
        "label": "experiment-a",
    }


def test_load_config_from_mlflow_run_requires_environment_config(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            return SimpleNamespace(data=SimpleNamespace(params={"training.lr": "0.001"}))

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    with pytest.raises(ValueError, match="pass --params"):
        load_config_from_mlflow_run("missing-environment")


def test_list_policy_models_from_mlflow_run_returns_deterministic_order(monkeypatch) -> None:
    logged_models = [
        SimpleNamespace(name="policy_model", model_id="final", creation_timestamp=300),
        SimpleNamespace(name="checkpoint", model_id="checkpoint-b", creation_timestamp=100),
        SimpleNamespace(name="checkpoint", model_id="checkpoint-a", creation_timestamp=100),
        SimpleNamespace(name="auxiliary", model_id="aux", creation_timestamp=None),
    ]

    class _FakeClient:
        def get_run(self, run_id: str):
            assert run_id == "run-123"
            return SimpleNamespace(info=SimpleNamespace(experiment_id="experiment-1"))

        def search_logged_models(self, **kwargs):
            assert kwargs == {
                "experiment_ids": ["experiment-1"],
                "filter_string": "source_run_id = 'run-123'",
            }
            return logged_models

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    models = list_policy_models_from_mlflow_run("run-123")

    assert [model.model_id for model in models] == [
        "aux",
        "checkpoint-a",
        "checkpoint-b",
        "final",
    ]
    assert models[-1].kind == "final"
    assert models[-1].step is None
    assert not models[-1].step_inferred


def test_list_policy_models_from_mlflow_run_rejects_empty_run(monkeypatch) -> None:
    class _FakeClient:
        def get_run(self, run_id: str):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="experiment-1"))

        def search_logged_models(self, **kwargs):
            return []

    monkeypatch.setattr("data_harvesting.eval.MlflowClient", _FakeClient)

    with pytest.raises(ValueError, match="No logged models"):
        list_policy_models_from_mlflow_run("empty-run")


def test_load_policy_from_model_id_uses_mlflow_model_uri(monkeypatch) -> None:
    loaded_uris: list[str] = []
    expected_policy = object()
    monkeypatch.setattr(
        "data_harvesting.eval.mlflow_pytorch.load_model",
        lambda uri: loaded_uris.append(uri) or expected_policy,
    )

    policy = load_policy_from_model_id("model-123")

    assert policy is expected_policy
    assert loaded_uris == ["models:/model-123"]


def test_load_policy_repairs_legacy_probabilistic_module(monkeypatch) -> None:
    legacy_policy = ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=torch.distributions.Normal,
    )
    del legacy_policy._generator
    del legacy_policy._generator_key
    monkeypatch.setattr(
        "data_harvesting.eval.mlflow_pytorch.load_model",
        lambda uri: legacy_policy,
    )

    policy = load_policy_from_model_id("legacy-model")

    assert policy is legacy_policy
    assert policy._generator is None
    assert policy._generator_key is None
