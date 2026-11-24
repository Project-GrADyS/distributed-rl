from copy import deepcopy
from types import SimpleNamespace

import mlflow.pytorch
import pandas as pd
import pytest
import torch

from data_harvesting.analysis import ExperimentRun


def _run(
    *,
    run_id: str = "run-123",
    run_name: str = "example",
    experiment_id: str = "experiment-1",
    metrics: dict[str, float] | None = None,
):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            run_name=run_name,
            experiment_id=experiment_id,
            status="FINISHED",
        ),
        data=SimpleNamespace(
            params={
                "environment": "{'max_episode_length': 50, 'min_num_agents': 1}",
                "evaluation.seed": "None",
            },
            metrics=metrics or {},
        ),
    )


def _point(*, step: int, value: float, timestamp: int):
    return SimpleNamespace(step=step, value=value, timestamp=timestamp)


class _FakeClient:
    def __init__(
        self,
        *,
        runs=None,
        histories=None,
        models=None,
        experiment=None,
    ) -> None:
        self.runs = runs or {}
        self.histories = histories or {}
        self.logged_models = models or []
        self.experiment = experiment

    def get_run(self, run_id: str):
        return self.runs[run_id]

    def get_experiment_by_name(self, experiment_name: str):
        return self.experiment

    def search_runs(self, **kwargs):
        return list(self.runs.values())

    def get_metric_history(self, run_id: str, metric_name: str):
        return self.histories.get(metric_name, [])

    def search_logged_models(self, **kwargs):
        assert kwargs["filter_string"] == "source_run_id = 'run-123'"
        return self.logged_models


def _analysis_run(*, histories=None, models=None, metrics=None) -> ExperimentRun:
    run = _run(metrics=metrics)
    client = _FakeClient(
        runs={run.info.run_id: run},
        histories=histories,
        models=models,
    )
    return ExperimentRun(
        run=run,
        client=client,
        tracking_uri="http://localhost:5000",
    )


def test_from_id_uses_requested_tracking_uri(monkeypatch) -> None:
    run = _run()
    clients = []

    def _client(*, tracking_uri: str):
        client = _FakeClient(runs={run.info.run_id: run})
        clients.append((tracking_uri, client))
        return client

    monkeypatch.setattr("data_harvesting.analysis.MlflowClient", _client)

    analysis_run = ExperimentRun.from_id(
        "run-123",
        tracking_uri="http://localhost:5000",
    )

    assert analysis_run.run_id == "run-123"
    assert clients[0][0] == "http://localhost:5000"


def test_from_name_requires_a_unique_run(monkeypatch) -> None:
    first = _run(run_id="run-a", run_name="duplicate")
    second = _run(run_id="run-b", run_name="duplicate")
    client = _FakeClient(
        runs={"run-a": first, "run-b": second},
        experiment=SimpleNamespace(experiment_id="experiment-1"),
    )
    monkeypatch.setattr(
        "data_harvesting.analysis.MlflowClient",
        lambda *, tracking_uri: client,
    )

    with pytest.raises(ValueError, match="run-a, run-b"):
        ExperimentRun.from_name(
            "experiment",
            "duplicate",
            tracking_uri="http://localhost:5000",
        )


def test_metrics_returns_wide_sparse_history_and_latest_duplicate() -> None:
    histories = {
        "eval/reward": [
            _point(step=20, value=2.0, timestamp=200),
            _point(step=10, value=1.0, timestamp=100),
            _point(step=10, value=1.5, timestamp=150),
        ],
        "loss": [_point(step=15, value=3.0, timestamp=120)],
    }
    run = _analysis_run(
        histories=histories,
        metrics={"eval/reward": 2.0, "loss": 3.0},
    )

    table = run.metrics()

    assert list(table) == ["step", "eval/reward", "loss"]
    assert table["step"].tolist() == [10, 15, 20]
    assert table.loc[table["step"] == 10, "eval/reward"].item() == 1.5
    assert pd.isna(table.loc[table["step"] == 15, "eval/reward"].item())


def test_metrics_validates_requested_names() -> None:
    run = _analysis_run(metrics={"known": 1.0})

    with pytest.raises(ValueError, match="Unknown metrics: missing"):
        run.metrics("missing")


def test_models_parse_steps_infer_final_and_join_exact_metrics() -> None:
    histories = {
        "eval/reward": [
            _point(step=100, value=1.0, timestamp=1000),
            _point(step=150, value=1.5, timestamp=1500),
            _point(step=200, value=2.0, timestamp=2000),
        ],
        "loss": [_point(step=200, value=3.0, timestamp=2000)],
    }
    models = [
        SimpleNamespace(
            name="policy_model",
            model_id="final",
            creation_timestamp=3000,
        ),
        SimpleNamespace(
            name="policy_checkpoint_step_100",
            model_id="checkpoint-100",
            creation_timestamp=1000,
        ),
        SimpleNamespace(
            name="policy_checkpoint_step_200",
            model_id="checkpoint-200",
            creation_timestamp=2000,
        ),
        SimpleNamespace(name="critic", model_id="critic", creation_timestamp=None),
    ]
    run = _analysis_run(
        histories=histories,
        models=models,
        metrics={"eval/reward": 2.0, "loss": 3.0},
    )

    model_table = run.models()
    joined = run.models_with_metrics()

    assert model_table["model_id"].tolist() == [
        "checkpoint-100",
        "checkpoint-200",
        "final",
        "critic",
    ]
    final_row = model_table.loc[model_table["model_id"] == "final"].iloc[0]
    assert final_row["step"] == 200
    assert bool(final_row["step_inferred"])
    assert run.checkpoint(200).model_id == "checkpoint-200"
    assert run.final_model().model_id == "final"
    assert joined.loc[joined["model_id"] == "checkpoint-100", "eval/reward"].item() == 1.0
    assert joined.loc[joined["model_id"] == "checkpoint-200", "eval/reward"].item() == 2.0
    assert pd.isna(joined.loc[joined["model_id"] == "critic", "eval/reward"].item())


def test_checkpoint_reports_available_steps() -> None:
    models = [
        SimpleNamespace(
            name="policy_checkpoint_step_100",
            model_id="checkpoint-100",
            creation_timestamp=1000,
        )
    ]
    run = _analysis_run(models=models)

    with pytest.raises(ValueError, match="Available steps: 100"):
        run.checkpoint(200)


def test_model_load_and_evaluation_return_episode_dataframe(monkeypatch) -> None:
    models = [
        SimpleNamespace(
            name="policy_checkpoint_step_100",
            model_id="checkpoint-100",
            creation_timestamp=1000,
        )
    ]
    run = _analysis_run(models=models)
    original_config = run.config
    overrides = {
        "environment": {"min_num_agents": 4, "max_num_agents": 4},
        "label": "scenario-a",
    }
    original_overrides = deepcopy(overrides)
    loaded = []
    evaluated = []

    monkeypatch.setattr(
        "data_harvesting.analysis.load_policy_from_model_id",
        lambda model_id, *, tracking_uri: loaded.append((model_id, tracking_uri))
        or "policy",
    )

    def _evaluate(policy, config, num_runs, *, visual, seed, num_workers):
        evaluated.append((policy, config, num_runs, visual, seed, num_workers))
        return {
            "episodes": [
                {
                    "run_index": 0,
                    "scenario_key": "agents_4__sensors_1",
                    "num_agents": 4,
                    "num_sensors": 1,
                    "avg_reward": 2.5,
                }
            ]
        }

    monkeypatch.setattr("data_harvesting.analysis.run_evaluation", _evaluate)

    checkpoint = run.checkpoint(100)
    assert checkpoint.load() == "policy"
    table = checkpoint.evaluate(
        3,
        config_overrides=overrides,
        seed=42,
        num_workers=4,
    )

    assert loaded == [("checkpoint-100", "http://localhost:5000")]
    assert evaluated[0][1]["environment"] == {
        "max_episode_length": 50,
        "min_num_agents": 4,
        "max_num_agents": 4,
    }
    assert evaluated[0][1]["evaluation"]["seed"] is None
    assert evaluated[0][1]["label"] == "scenario-a"
    assert evaluated[0][2:] == (3, False, 42, 4)
    assert table.loc[0, "source_run_id"] == "run-123"
    assert table.loc[0, "model_step"] == 100
    assert table.loc[0, "avg_reward"] == 2.5
    assert run.config == original_config
    assert overrides == original_overrides


def test_export_and_open_moved_run_offline(tmp_path, monkeypatch):
    model_names = ["policy_checkpoint_step_100", "policy_model"]
    models = [
        SimpleNamespace(name=name, model_id=name, creation_timestamp=1000)
        for name in model_names
    ]
    run = _analysis_run(
        metrics={"eval/reward": 2.0, "empty": 0},
        histories={"eval/reward": [
            _point(step=100, timestamp=1, value=1.0),
            _point(step=100, timestamp=2, value=2.0),
        ]},
        models=models,
    )
    inputs = torch.ones(1, 2)
    outputs = {}
    for index, name in enumerate(model_names):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(index + 1)
            model.bias.zero_()
        mlflow.pytorch.save_model(model, str(tmp_path / name), pip_requirements=[])
        outputs[name] = model(inputs).detach()

    def artifact(model_id):
        return SimpleNamespace(artifact_location=(tmp_path / model_id).as_uri())

    monkeypatch.setattr(run._client, "get_logged_model", artifact, raising=False)
    bundle = run.export(tmp_path / "exported").rename(tmp_path / "relocated")
    monkeypatch.setattr("data_harvesting.analysis.MlflowClient", lambda **kwargs: pytest.fail("tracking access"))
    local = ExperimentRun.from_directory(bundle)
    assert local.config == run.config
    assert local.status == run.status
    pd.testing.assert_frame_equal(local.models_with_metrics(), run.models_with_metrics())
    pd.testing.assert_frame_equal(local.metrics(), run.metrics())
    for model in [local.final_model(), local.checkpoint(100)]:
        torch.testing.assert_close(model.load()(inputs), outputs[model.name])
    monkeypatch.setattr("data_harvesting.analysis.run_evaluation",
                        lambda *args, **kwargs: {"episodes": [{"run_index": 0}]})
    row = local.final_model().evaluate(1, seed=42).iloc[0]
    assert row["source_run_id"] == run.run_id
    assert row["model_id"] == "policy_model"
    with pytest.raises(FileExistsError):
        run.export(bundle)
