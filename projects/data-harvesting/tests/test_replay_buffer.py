from collections import Counter

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer

from data_harvesting.replay import TeamSizeBucketedReplayBuffer, create_replay_buffer


def _replay_config(*, buffer_size: int = 32, batch_size: int = 8, prefetch: int = 0, buffer_device: str = "cpu") -> dict:
    return {
        "training": {
            "batch_size": batch_size,
        },
        "replay_buffer": {
            "buffer_size": buffer_size,
            "prefetch": prefetch,
            "buffer_device": buffer_device,
        },
    }


def _make_batch(frames: int, *, n_agents: int = 2, action_dim: int = 2) -> TensorDict:
    return TensorDict(
        {
            "agents": {
                "action": torch.rand(frames, n_agents, action_dim),
                "reward": torch.rand(frames, n_agents, 1),
                "done": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                "terminated": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                "mask": torch.ones(frames, n_agents, dtype=torch.bool),
                "observation": {
                    "sensors": torch.rand(frames, n_agents, 1, 2),
                    "drones": torch.rand(frames, n_agents, 1, 2),
                },
            },
            "next": {
                "agents": {
                    "reward": torch.rand(frames, n_agents, 1),
                    "done": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                    "terminated": torch.zeros(frames, n_agents, 1, dtype=torch.bool),
                    "observation": {
                        "sensors": torch.rand(frames, n_agents, 1, 2),
                        "drones": torch.rand(frames, n_agents, 1, 2),
                    },
                }
            },
        },
        batch_size=[frames],
    )


def test_replay_buffer_uses_configured_batch_size() -> None:
    config = _replay_config(buffer_size=64, batch_size=7, prefetch=0)
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))

    assert replay_buffer.batch_size == 7


def test_replay_buffer_sample_has_expected_shapes_and_keys() -> None:
    config = _replay_config(buffer_size=64, batch_size=6, prefetch=0)
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))
    replay_buffer.extend(_make_batch(24))

    sample = replay_buffer.sample()

    assert sample.batch_size == torch.Size([6])
    assert tuple(sample.get(("agents", "action")).shape) == (6, 2, 2)
    assert tuple(sample.get(("agents", "reward")).shape) == (6, 2, 1)
    assert tuple(sample.get(("next", "agents", "observation", "sensors")).shape) == (6, 2, 1, 2)
    assert sample.get(("agents", "done")).dtype == torch.bool
    assert sample.get(("agents", "terminated")).dtype == torch.bool


def test_replay_buffer_respects_capacity() -> None:
    config = _replay_config(buffer_size=10, batch_size=4, prefetch=0)
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))
    replay_buffer.extend(_make_batch(25))

    assert len(replay_buffer) == 10


def test_replay_buffer_samples_on_requested_device() -> None:
    config = _replay_config(buffer_size=32, batch_size=5, prefetch=0, buffer_device="cpu")
    replay_buffer = create_replay_buffer(config, torch.device("cpu"))
    replay_buffer.extend(_make_batch(16))

    sample = replay_buffer.sample()

    assert sample.get(("agents", "action")).device.type == "cpu"
    assert sample.get(("next", "agents", "reward")).device.type == "cpu"


def _bucket_config(*, buffer_size: int = 32, batch_size: int = 8) -> dict:
    config = _replay_config(buffer_size=buffer_size, batch_size=batch_size)
    config["replay_buffer"]["sampling_strategy"] = "initial_team_size"
    config["environment"] = {"min_num_agents": 1, "max_num_agents": 3}
    return config


def _bucket_batch(sizes: list[int]) -> TensorDict:
    batch = _make_batch(len(sizes), n_agents=3)
    batch.set("initial_team_size", torch.tensor(sizes, dtype=torch.int64).unsqueeze(-1))
    batch.set("transition_id", torch.arange(len(sizes)))
    return batch


def test_default_and_explicit_uniform_replay_are_compatible() -> None:
    config = _replay_config(prefetch=0)
    default = create_replay_buffer(config, "cpu")
    config["replay_buffer"]["sampling_strategy"] = "uniform"
    explicit = create_replay_buffer(config, "cpu")
    assert type(default) is type(explicit) is ReplayBuffer
    data = _make_batch(16)  # Legacy batches need no episode provenance.
    default.extend(data)
    explicit.extend(data)
    with torch.random.fork_rng():
        torch.manual_seed(123)
        expected = default.sample()
        torch.manual_seed(123)
        actual = explicit.sample()
    torch.testing.assert_close(actual, expected)


def test_bucket_assignment_uses_initial_size_despite_deaths() -> None:
    replay = create_replay_buffer(_bucket_config(batch_size=20), "cpu")
    data = _bucket_batch([3, 1, 3, 2, 3])
    data["agents", "mask"][:] = False
    data["agents", "mask"][:, 0] = True  # Only one survivor, even for size 3.
    replay.extend(data)
    assert [len(buffer) for buffer in replay._buffers.values()] == [1, 1, 3]
    assert replay._buffers[3].sample()["initial_team_size"].unique().tolist() == [3]


def test_bucket_selection_is_uniform_despite_unequal_occupancy() -> None:
    config = _bucket_config(buffer_size=600, batch_size=4)
    replay = create_replay_buffer(config, "cpu")
    replay.extend(_bucket_batch([1] + [2] * 20 + [3] * 200))
    counts = Counter()
    with torch.random.fork_rng():
        torch.manual_seed(437)
        for _ in range(900):
            sample = replay.sample()
            sizes = sample["initial_team_size"].unique().tolist()
            assert len(sizes) == 1
            counts[sizes[0]] += 1
    assert set(counts) == {1, 2, 3}
    assert all(240 < count < 360 for count in counts.values())


def test_bucket_minibatch_samples_uniformly_with_replacement() -> None:
    replay = create_replay_buffer(_bucket_config(batch_size=2000), "cpu")
    replay.extend(_bucket_batch([2] * 5))
    with torch.random.fork_rng():
        torch.manual_seed(27)
        sample = replay.sample()
    assert sample.batch_size == torch.Size([2000])
    assert sample["agents", "action"].shape == (2000, 3, 2)
    assert sample["next", "agents", "reward"].device.type == "cpu"
    counts = sample["transition_id"].bincount(minlength=5)
    assert torch.all((counts > 320) & (counts < 480))


def test_empty_buckets_are_skipped_and_one_transition_is_eligible() -> None:
    replay = create_replay_buffer(_bucket_config(batch_size=12), "cpu")
    with pytest.raises(RuntimeError, match="empty"):
        replay.sample()
    replay.extend(_bucket_batch([3]))
    sample = replay.sample()
    assert sample.batch_size == torch.Size([12])
    assert sample["initial_team_size"].unique().tolist() == [3]
    assert sample["transition_id"].unique().tolist() == [0]


def test_bucket_capacities_share_budget_and_evict_oldest_per_bucket() -> None:
    replay = create_replay_buffer(_bucket_config(buffer_size=11), "cpu")
    assert isinstance(replay, TeamSizeBucketedReplayBuffer)
    assert [buffer.storage.max_size for buffer in replay._buffers.values()] == [4, 4, 3]
    replay.extend(_bucket_batch([1] * 4 + [2] * 4 + [3] * 3))
    replacement = _bucket_batch([1, 1])
    replacement["transition_id"] += 100
    replay.extend(replacement)
    assert len(replay) == 11
    assert set(replay._buffers[1][:]["transition_id"].tolist()) == {2, 3, 100, 101}
    assert set(replay._buffers[2][:]["transition_id"].tolist()) == {4, 5, 6, 7}
    replay.extend(_bucket_batch([3] * 20))
    assert len(replay) == 11
    assert len(replay._buffers[3]) == 3


def test_bucket_budget_reserves_capacity_for_unseen_sizes() -> None:
    replay = create_replay_buffer(_bucket_config(buffer_size=11), "cpu")
    replay.extend(_bucket_batch([2] * 30))
    assert len(replay) == 4
    assert len(replay._buffers[1]) == len(replay._buffers[3]) == 0


def test_bucketed_replay_disables_per_bucket_prefetch() -> None:
    config = _bucket_config()
    config["replay_buffer"]["prefetch"] = 32
    replay = create_replay_buffer(config, "cpu")
    assert all(not buffer._prefetch for buffer in replay._buffers.values())


def test_bucketed_replay_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="at least one transition"):
        create_replay_buffer(_bucket_config(buffer_size=2), "cpu")


@pytest.mark.parametrize("bad_sizes", [[0], [4], [1, 4]])
def test_bucketed_replay_rejects_invalid_team_sizes_before_writing(bad_sizes) -> None:
    replay = create_replay_buffer(_bucket_config(), "cpu")
    with pytest.raises(ValueError, match="outside"):
        replay.extend(_bucket_batch(bad_sizes))
    assert len(replay) == 0


def test_bucketed_replay_requires_provenance() -> None:
    replay = create_replay_buffer(_bucket_config(), "cpu")
    with pytest.raises(ValueError, match="requires initial_team_size"):
        replay.extend(_make_batch(4))


def test_replay_rejects_unknown_strategy() -> None:
    config = _replay_config()
    config["replay_buffer"]["sampling_strategy"] = "typo"
    with pytest.raises(ValueError, match="sampling_strategy"):
        create_replay_buffer(config, "cpu")
