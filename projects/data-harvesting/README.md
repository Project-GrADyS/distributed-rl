# Data Harvesting

The data-harvesting project trains and evaluates fixed-slot multi-agent policies
with GrADyS, PyTorch, TorchRL, `flex-marl`, and `rl-core`.

## Layout

```text
projects/data-harvesting/
├── src/data_harvesting/  # Environment, algorithms, and project integration
├── scripts/              # Training, evaluation, tuning, profiling, visualization
├── tests/                # Project-owned tests
├── mlflow_server/        # Private MLflow deployment and backup support
└── params.yaml           # Default runtime configuration
```

Sequential observations expose explicit Boolean `sensors_mask` and
`drones_mask` tensors. `True` means that the corresponding element is valid;
consumers must not infer validity from the `-1` padding value.

## Commands

Run commands from the workspace root:

```bash
uv run --package data-harvesting python projects/data-harvesting/scripts/main.py -E default
uv run --package data-harvesting python projects/data-harvesting/scripts/evaluate.py --run-id <RUN_ID> --num-runs 10
uv run --package data-harvesting python projects/data-harvesting/scripts/tune.py -E tuning
uv run --package data-harvesting python projects/data-harvesting/scripts/profile_training.py
uv run --package data-harvesting python projects/data-harvesting/scripts/visualize_env.py
uv run --package data-harvesting pytest projects/data-harvesting/tests
```

Evaluation uses the configuration logged on the MLflow run by default. Pass
`--params <PATH>` to override it with a local YAML file. To evaluate every
model logged to a run and write a combined per-episode table, run:

```bash
uv run --package data-harvesting python projects/data-harvesting/scripts/evaluate.py \
  --run-id <RUN_ID> --num-runs <N> --all-models --output-table results.csv
```

The combined table includes `model_name` and `model_id` columns. The printed
comparison ranks models by total `all_collected`; evaluation forces
`end_when_all_collected` on so collection success terminates the episode.

Scripts that use a default configuration resolve `params.yaml` relative to this
project. The default MLflow tracking location is
`projects/data-harvesting/mlruns`; set
`MLFLOW_TRACKING_URI` or pass `--tracking-uri` to use the private server.

Older MLflow policies saved as pickled full modules before the `flex-marl`
migration are not compatible with this project layout.

## Initial-team-size bucketed replay

To run the additional MADDPG "Alarcon Granadeno et al. (2025)" baseline, copy `params.yaml` and set:

```yaml
environment:
  agent_count_sampling: uniform
replay_buffer:
  sampling_strategy: initial_team_size
```

These are overrides to the full configuration, not a standalone config. Launch
with `main.py --params <PATH>` using the command above. Omitting
`sampling_strategy` (or setting it to `uniform`) retains transition-uniform
replay. Initial-team collection sampling (`uniform` or `inverse`) is independent
of replay sampling; the bucketed baseline pairs uniform collection with bucketed
replay.

Each update selects one nonempty bucket uniformly, then draws the whole
minibatch from it uniformly with replacement. A bucket becomes eligible after
its first transition; empty buckets are skipped, and sampling an entirely empty
buffer raises an error. This matches the existing replay's lack of a separate
warmup threshold and ability to sample batches larger than the stored data.
The `initial_team_size` metadata is recorded at reset and stays unchanged after
deaths, across collection batches, and through replay.

`buffer_size` remains the total transition capacity. It is divided equally
across all configured team sizes, with any remainder assigned to the smallest
sizes. Each bucket evicts its own oldest transitions; unused bucket capacity is
reserved, not borrowed. The budget must allow at least one transition per team
size. Bucketed replay samples on demand and ignores `prefetch`, avoiding a queue
per bucket. Storage and sample devices follow the existing settings.

This adapts the replay sampling described by [Alarcon Granadeno et al. (2025)](https://doi.org/10.1016/j.mlwa.2025.100737).
The existing joint-transition minibatches, active-agent masking, actor/value
losses, and optimizer updates are unchanged; it does not implement their
single-host-agent update or reproduce ADDRQN. MAPPO does not use replay and
rejects the bucketed option.

## Notebook analysis

`data_harvesting.analysis` provides a pandas-oriented interface for inspecting
training runs and evaluating their logged policies:

```python
from data_harvesting.analysis import ExperimentRun

run = ExperimentRun.from_name(
    "death",
    "6_mlp",
    tracking_uri="http://localhost:5000",
)

run.metric_names
metrics = run.metrics(["eval/all_collected", "eval/avg_reward"])
models = run.models_with_metrics()

checkpoint = run.checkpoint(1_000_448)
policy = checkpoint.load()
episodes = checkpoint.evaluate(
    100,
    config_overrides={
        "environment": {
            "min_num_agents": 4,
            "max_num_agents": 4,
        }
    },
    seed=1000,
)
```

Run names must be unique within their experiment. Use
`ExperimentRun.from_id(...)` when names are duplicated. Model/metric tables use
exact MLflow steps; the final `policy_model` is assigned the run's maximum
logged metric step and marked with `step_inferred=True`.

A clean starter notebook is available at
`analysis/mlflow_analysis.ipynb`.
