import pytest
import torch
import torch.nn.utils as nn_utils

from data_harvesting.algorithm import MADDPGAlgorithm, MAPPOAlgorithm
from data_harvesting.collector import create_collector
from data_harvesting.environment.data_collection import make_data_collection_env


def _maddpg_test_config() -> dict:
    return {
        "environment": {
            "algorithm_iteration_interval": 1.0,
            "min_num_agents": 2,
            "max_num_agents": 2,
            "min_num_sensors": 2,
            "max_num_sensors": 2,
            "scenario_size": 20.0,
            "max_episode_length": 50,
            "max_seconds_stalled": 20,
            "communication_range": 0.0,
            "state_num_closest_sensors": 2,
            "state_num_closest_drones": 1,
            "id_on_state": True,
            "min_sensor_priority": 1.0,
            "max_sensor_priority": 1.0,
            "full_random_drone_position": False,
            "reward": "punish",
            "speed_action": True,
            "end_when_all_collected": False,
        },
        "actor": {
            "share_parameters": True,
            "network_depth": 1,
            "network_width": 32,
            "activation_function": "Tanh",
            "centralized": False,
        },
        "critic": {
            "network_depth": 1,
            "network_width": 32,
            "activation_function": "Tanh",
            "share_parameters": True,
            "centralized": True,
        },
        "training": {
            "total_timesteps": 64,
            "batch_size": 8,
            "algorithm": "maddpg",
            "exploration_sigma_init": 0.2,
            "exploration_sigma_end": 0.1,
            "exploration_annealing_steps": 100,
        },
        "collector": {
            "num_collectors": 1,
            "frames_per_batch": 16,
            "async_collector": False,
            "device": "cpu",
        },
        "replay_buffer": {
            "buffer_size": 256,
            "prefetch": 0,
            "buffer_device": "cpu",
        },
        "optimization": {
            "gamma": 0.99,
            "lr": 1e-3,
            "tau": 0.5,
            "num_optimizer_steps": 1,
            "grad_clip": 0,
            "use_amp": False,
        },
        "flex_encoder": {
            "enabled": False,
        },
    }


def _maddpg_test_config_with_overrides(**overrides) -> dict:
    config = _maddpg_test_config()
    for dotted_key, value in overrides.items():
        section, key = dotted_key.split(".", 1)
        config[section][key] = value
    return config


def _collect_training_batch(algorithm: MADDPGAlgorithm, config: dict) -> torch.Tensor:
    with create_collector(
        algorithm.exploratory_policy,
        torch.device("cpu"),
        lambda: make_data_collection_env(config),
        config,
    ) as collector:
        batch = next(iter(collector))
    return batch.reshape(-1)


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in module.parameters()]


def _clone_td_params(td_params) -> list[torch.Tensor]:
    return [value.detach().clone() for value in td_params.flatten_keys().values()]


def _any_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return any(not torch.allclose(prev, curr) for prev, curr in zip(before, after, strict=True))


def test_maddpg_learn_returns_finite_losses_and_updates_models() -> None:
    config = _maddpg_test_config()
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)

        actor_before = _clone_params(algorithm.policy)
        critic_before = _clone_params(algorithm.critic)

        losses = algorithm.learn(batch)

        assert torch.isfinite(losses["loss_actor"])
        assert torch.isfinite(losses["loss_value"])
        assert len(algorithm.replay_buffer) > 0

        actor_after = _clone_params(algorithm.policy)
        critic_after = _clone_params(algorithm.critic)

        assert _any_changed(actor_before, actor_after)
        assert _any_changed(critic_before, critic_after)
    finally:
        env.close()


def test_variable_agent_mlp_maddpg_learns_with_finite_losses() -> None:
    config = _maddpg_test_config_with_overrides(
        **{
            "environment.min_num_agents": 1,
            "environment.max_num_agents": 3,
        }
    )
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)

        losses = algorithm.learn(batch)

        assert torch.isfinite(losses["loss_actor"])
        assert torch.isfinite(losses["loss_value"])
        assert batch.get(("agents", "mask")).shape[-1] == 3
    finally:
        env.close()


@pytest.mark.parametrize("collection_sampling", ["uniform", "inverse"])
def test_bucketed_maddpg_collects_provenance_and_uses_existing_loss(collection_sampling) -> None:
    config = _maddpg_test_config_with_overrides(
        **{
            "environment.min_num_agents": 1,
            "environment.max_num_agents": 3,
            "environment.agent_count_sampling": collection_sampling,
            "environment.max_episode_length": 4,
            "environment.death_scheduler": {"type": "scheduled", "timesteps": [1]},
            "replay_buffer.sampling_strategy": "initial_team_size",
        }
    )
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)
        sizes = batch["initial_team_size"]
        assert sizes.dtype == torch.int64
        torch.testing.assert_close(sizes, batch["next", "initial_team_size"])
        traj_ids = batch["collector", "traj_ids"]
        for trajectory in traj_ids.unique():
            episode = batch[traj_ids == trajectory]
            assert episode["initial_team_size"].unique().numel() == 1
            assert torch.all(episode["initial_team_size"].squeeze(-1) >= episode["agents", "mask"].sum(-1))
        actor_before = _clone_params(algorithm.policy)
        losses = algorithm.learn(batch)
        assert torch.isfinite(losses["loss_actor"])
        assert torch.isfinite(losses["loss_value"])
        assert _any_changed(actor_before, _clone_params(algorithm.policy))
        assert len(algorithm.replay_buffer) == batch.numel()
    finally:
        env.close()


def test_mappo_rejects_bucketed_replay() -> None:
    config = _maddpg_test_config_with_overrides(
        **{"replay_buffer.sampling_strategy": "initial_team_size"}
    )
    with pytest.raises(ValueError, match="MAPPO does not use replay"):
        MAPPOAlgorithm(None, torch.device("cpu"), config)


def test_maddpg_target_network_updates_after_learn() -> None:
    config = _maddpg_test_config()
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)

        target_actor_before = _clone_td_params(algorithm.loss_module.target_actor_network_params)
        target_value_before = _clone_td_params(algorithm.loss_module.target_value_network_params)

        _ = algorithm.learn(batch)

        target_actor_after = _clone_td_params(algorithm.loss_module.target_actor_network_params)
        target_value_after = _clone_td_params(algorithm.loss_module.target_value_network_params)

        assert _any_changed(target_actor_before, target_actor_after)
        assert _any_changed(target_value_before, target_value_after)
    finally:
        env.close()


def test_maddpg_learn_anneals_exploration_noise_by_collected_frames() -> None:
    config = _maddpg_test_config_with_overrides(
        **{
            "training.exploration_sigma_init": 0.4,
            "training.exploration_sigma_end": 0.1,
            "training.exploration_annealing_steps": 8,
            "collector.frames_per_batch": 16,
        }
    )
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)

        sigma_before = float(algorithm.exploration_noise.sigma.item())
        _ = algorithm.learn(batch)
        sigma_after = float(algorithm.exploration_noise.sigma.item())

        assert abs(sigma_before - 0.4) < 1e-6
        assert abs(sigma_after - 0.1) < 1e-6
    finally:
        env.close()


def test_maddpg_learn_applies_grad_clip_when_enabled(monkeypatch) -> None:
    config = _maddpg_test_config_with_overrides(
        **{
            "optimization.grad_clip": 0.5,
            "optimization.num_optimizer_steps": 2,
        }
    )
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        batch = _collect_training_batch(algorithm, config)

        calls: list[float] = []
        original_clip = nn_utils.clip_grad_norm_

        def _recording_clip(params, max_norm, *args, **kwargs):
            calls.append(float(max_norm))
            return original_clip(params, max_norm, *args, **kwargs)

        monkeypatch.setattr(nn_utils, "clip_grad_norm_", _recording_clip)

        _ = algorithm.learn(batch)

        assert len(calls) == 2 * config["optimization"]["num_optimizer_steps"]
        assert all(abs(value - 0.5) < 1e-9 for value in calls)
    finally:
        env.close()
