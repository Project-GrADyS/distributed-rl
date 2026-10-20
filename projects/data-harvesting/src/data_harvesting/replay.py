from torchrl.data import ReplayBuffer, RandomSampler, LazyTensorStorage
from typing import Any, Dict
import torch


class TeamSizeBucketedReplayBuffer:
    """Uniformly choose a nonempty initial-team-size bucket per minibatch.

    The total transition budget is divided across all configured team sizes.
    Each bucket uses TorchRL's usual FIFO eviction and sampling with replacement,
    so even one transition makes a bucket eligible. No extra warmup is required.
    """

    def __init__(self, config: Dict[str, Any], device: Any):
        environment = config["environment"]
        minimum = int(environment["min_num_agents"])
        maximum = int(environment["max_num_agents"])
        if minimum < 1 or maximum < minimum:
            raise ValueError("Replay requires 1 <= min_num_agents <= max_num_agents.")
        bucket_count = maximum - minimum + 1
        capacity, remainder = divmod(config["replay_buffer"]["buffer_size"], bucket_count)
        if capacity < 1:
            raise ValueError("buffer_size must allow at least one transition per team-size bucket.")
        self.batch_size = config["training"]["batch_size"]
        self._buffers = {}
        for offset, team_size in enumerate(range(minimum, maximum + 1)):
            buffer = ReplayBuffer(
                storage=LazyTensorStorage(
                    capacity + (offset < remainder),
                    device=config["replay_buffer"]["buffer_device"],
                ),
                sampler=RandomSampler(),
                batch_size=self.batch_size,
                # Prefetching every bucket would multiply the queued minibatch
                # budget and retain stale samples from rarely chosen buckets.
                prefetch=0,
            )
            buffer.append_transform(lambda x: x.to(device))
            self._buffers[team_size] = buffer

    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self._buffers.values())

    def extend(self, batch) -> None:
        team_sizes = batch.get("initial_team_size", None)
        if team_sizes is None:
            raise ValueError("Bucketed replay requires initial_team_size on every transition.")
        if team_sizes.shape != batch.batch_size + torch.Size([1]) or team_sizes.dtype != torch.int64:
            raise ValueError("initial_team_size must be an int64 scalar per transition (trailing size 1).")
        team_sizes = team_sizes.squeeze(-1)
        sizes = team_sizes.unique().tolist()
        if any(size not in self._buffers for size in sizes):
            raise ValueError("initial_team_size is outside the configured team-size range.")
        for size in sizes:
            self._buffers[size].extend(batch[team_sizes == size])

    def sample(self, batch_size: int | None = None):
        eligible = [buffer for buffer in self._buffers.values() if len(buffer)]
        if not eligible:
            raise RuntimeError("Cannot sample from an empty storage.")
        bucket = int(torch.randint(len(eligible), ()).item())
        return eligible[bucket].sample(batch_size)


def create_replay_buffer(
    config: Dict[str, Any], device: Any
) -> ReplayBuffer | TeamSizeBucketedReplayBuffer:
    """
    Creates a replay buffer using parameters from config.
    Args:
        config: Configuration dictionary (expects 'replay_buffer' and 'training' sections).
        device: The device to move samples to.
    Returns:
        Configured uniform replay buffer or initial-team-size bucketed replay.
    """
    strategy = config["replay_buffer"].get("sampling_strategy", "uniform")
    if strategy == "initial_team_size":
        return TeamSizeBucketedReplayBuffer(config, device)
    if strategy != "uniform":
        raise ValueError("replay_buffer.sampling_strategy must be 'uniform' or 'initial_team_size'.")
    buffer_size = config["replay_buffer"]["buffer_size"]
    batch_size = config["training"]["batch_size"]
    prefetch = config["replay_buffer"]["prefetch"]
    buffer_device = config["replay_buffer"]["buffer_device"]
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(buffer_size, device=buffer_device),
        sampler=RandomSampler(),
        batch_size=batch_size,
        prefetch=prefetch,
    )
    replay_buffer.append_transform(lambda x: x.to(device))
    return replay_buffer
