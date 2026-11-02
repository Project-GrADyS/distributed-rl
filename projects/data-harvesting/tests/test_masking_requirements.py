from __future__ import annotations

import pytest
import torch
from torch import nn
from torchrl.modules import MultiAgentMLP

from data_harvesting.actor import create_actor, create_ppo_actor
from data_harvesting.critic import create_critic, create_ppo_value_net
from data_harvesting.environment import requires_masking as requires_environment_masking
from data_harvesting.environment.data_collection import make_data_collection_env
from data_harvesting.environment.data_collection.config import (
    requires_masking as requires_data_collection_masking,
)
from data_harvesting.environment.data_collection.data_collection import DataCollectionEnvironmentConfig
from data_harvesting.environment.data_collection.death import StochasticDeathScheduler
from data_harvesting.algorithm import MADDPGAlgorithm
from data_harvesting.encoding import MaskedMultiAgentMLP
from data_harvesting.optimization import create_ppo_loss


def _base_env_config(
    *,
    min_num_agents: int,
    max_num_agents: int,
    reward: str = "punish",
    death_scheduler: dict | None = None,
    prevent_last_agent_death: bool = True,
) -> dict:
    return {
        "algorithm_iteration_interval": 1.0,
        "min_num_agents": min_num_agents,
        "max_num_agents": max_num_agents,
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
        "reward": reward,
        "speed_action": True,
        "end_when_all_collected": False,
        "death_scheduler": (
            death_scheduler
            if death_scheduler is not None
            else {"type": "stochastic", "probability": 0.0}
        ),
        "prevent_last_agent_death": prevent_last_agent_death,
    }


def _base_config(
    *,
    min_num_agents: int,
    max_num_agents: int,
    flex_enabled: bool = False,
    algorithm: str = "maddpg",
    death_scheduler: dict | None = None,
    prevent_last_agent_death: bool = True,
) -> dict:
    return {
        "environment": _base_env_config(
            min_num_agents=min_num_agents,
            max_num_agents=max_num_agents,
            death_scheduler=death_scheduler,
            prevent_last_agent_death=prevent_last_agent_death,
        ),
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
            "algorithm": algorithm,
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
            "enabled": flex_enabled,
            "sequential_heads": {
                "embed_dim": 16,
                "num_heads": 2,
                "ff_dim": 32,
                "depth": 1,
                "dropout": 0.0,
                "critic_agent_embedding": True,
            },
            "flat_heads": {
                "embed_dim": 8,
                "depth": 1,
                "num_cells": 16,
                "activation_function": "Tanh",
            },
            "mix_layer_depth": 1,
            "mix_layer_num_cells": 32,
            "mix_activation_function": "Tanh",
        },
        "ppo": {
            "clip_epsilon": 0.2,
            "gae_lambda": 0.95,
            "entropy_coef": 0.0,
            "value_coef": 0.5,
            "num_epochs": 1,
            "minibatch_size": 8,
        },
    }


@pytest.mark.parametrize(
    "min_num_agents,max_num_agents,expected",
    [
        (2, 2, False),
        (1, 3, True),
    ],
)
def test_requires_masking_helpers_match_environment_config(
    min_num_agents: int,
    max_num_agents: int,
    expected: bool,
) -> None:
    data_collection_config = DataCollectionEnvironmentConfig(
        render_mode=None,
        algorithm_iteration_interval=1.0,
        min_num_agents=min_num_agents,
        max_num_agents=max_num_agents,
        min_num_sensors=2,
        max_num_sensors=2,
        scenario_size=20.0,
        max_episode_length=50,
        max_seconds_stalled=20,
        communication_range=0.0,
        state_num_closest_sensors=2,
        state_num_closest_drones=1,
        id_on_state=True,
        min_sensor_priority=1.0,
        max_sensor_priority=1.0,
        full_random_drone_position=False,
        reward="punish",
        speed_action=True,
        end_when_all_collected=False,
    )
    full_config = {
        "environment": {
            **_base_env_config(
                min_num_agents=min_num_agents,
                max_num_agents=max_num_agents,
            )
        }
    }

    assert requires_data_collection_masking(
        data_collection_config,
        StochasticDeathScheduler(0.0),
    ) is expected
    assert requires_environment_masking(full_config) is expected


def test_requires_masking_when_mid_episode_death_is_enabled() -> None:
    data_collection_config = DataCollectionEnvironmentConfig(
        render_mode=None,
        algorithm_iteration_interval=1.0,
        min_num_agents=2,
        max_num_agents=2,
        min_num_sensors=2,
        max_num_sensors=2,
        scenario_size=20.0,
        max_episode_length=50,
        max_seconds_stalled=20,
        communication_range=0.0,
        state_num_closest_sensors=2,
        state_num_closest_drones=1,
        id_on_state=True,
        min_sensor_priority=1.0,
        max_sensor_priority=1.0,
        full_random_drone_position=False,
        reward="punish",
        speed_action=True,
        end_when_all_collected=False,
        prevent_last_agent_death=True,
    )
    full_config = {
        "environment": {
            **_base_env_config(
                min_num_agents=2,
                max_num_agents=2,
                death_scheduler={"type": "stochastic", "probability": 0.1},
                prevent_last_agent_death=True,
            )
        }
    }

    assert requires_data_collection_masking(
        data_collection_config,
        StochasticDeathScheduler(0.1),
    ) is True
    assert requires_environment_masking(full_config) is True


def test_requires_masking_is_true_for_single_agent_when_death_is_enabled() -> None:
    data_collection_config = DataCollectionEnvironmentConfig(
        render_mode=None,
        algorithm_iteration_interval=1.0,
        min_num_agents=1,
        max_num_agents=1,
        min_num_sensors=2,
        max_num_sensors=2,
        scenario_size=20.0,
        max_episode_length=50,
        max_seconds_stalled=20,
        communication_range=0.0,
        state_num_closest_sensors=2,
        state_num_closest_drones=1,
        id_on_state=True,
        min_sensor_priority=1.0,
        max_sensor_priority=1.0,
        full_random_drone_position=False,
        reward="punish",
        speed_action=True,
        end_when_all_collected=False,
        prevent_last_agent_death=True,
    )
    full_config = {
        "environment": {
            **_base_env_config(
                min_num_agents=1,
                max_num_agents=1,
                death_scheduler={"type": "stochastic", "probability": 0.1},
                prevent_last_agent_death=True,
            )
        }
    }

    assert requires_data_collection_masking(
        data_collection_config,
        StochasticDeathScheduler(0.1),
    ) is True
    assert requires_environment_masking(full_config) is True


@pytest.mark.parametrize(
    ("timesteps", "expected"),
    [([], False), ([5], True)],
)
def test_requires_masking_reflects_scheduled_deaths(
    timesteps: list[int],
    expected: bool,
) -> None:
    config = _base_config(
        min_num_agents=2,
        max_num_agents=2,
        death_scheduler={"type": "scheduled", "timesteps": timesteps},
    )

    assert requires_environment_masking(config) is expected


def test_mlp_actor_and_critic_support_masking_required_environments() -> None:
    config = _base_config(min_num_agents=1, max_num_agents=3, flex_enabled=False)
    env = make_data_collection_env(config)
    try:
        actor = create_actor(env, torch.device("cpu"), config)
        critic = create_critic(env, torch.device("cpu"), config)
        td = actor(env.reset(seed=0).unsqueeze(0))
        td = critic(td)

        assert td.get(("agents", "action")).shape == (1, 3, 2)
        assert td.get(("agents", "state_action_value")).shape == (1, 3, 1)
        assert torch.isfinite(td.get(("agents", "state_action_value"))).all()
    finally:
        env.close()


def test_masked_mlp_replaces_only_inactive_agent_inputs() -> None:
    class _Capture(nn.Module):
        n_agents = 3

        def __init__(self) -> None:
            super().__init__()
            self.inputs = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.inputs = inputs
            return inputs

    capture = _Capture()
    module = MaskedMultiAgentMLP(capture)
    inputs = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    output = module(inputs, mask)
    expected = inputs.masked_fill(~mask.unsqueeze(-1), -1.0)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(capture.inputs, expected)


@pytest.mark.parametrize("centralized", [False, True])
@pytest.mark.parametrize("share_params", [False, True])
@pytest.mark.parametrize("population", [1, 4, 8])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_masked_mlp_smaller_population_matches_inactive_slots(
    centralized: bool,
    share_params: bool,
    population: int,
    batch_shape: tuple[int, ...],
) -> None:
    module = MaskedMultiAgentMLP(
        MultiAgentMLP(
            n_agent_inputs=5,
            n_agent_outputs=2,
            n_agents=8,
            centralized=centralized,
            share_params=share_params,
            depth=1,
            num_cells=16,
        )
    )
    inputs = torch.randn(*batch_shape, population, 5)
    mask = torch.ones(*batch_shape, population, dtype=torch.bool)
    mask[..., 1::2] = False
    full_inputs = torch.randn(*batch_shape, 8, 5)
    full_inputs[..., :population, :] = inputs
    full_mask = torch.zeros(*batch_shape, 8, dtype=torch.bool)
    full_mask[..., :population] = mask

    expected = module(full_inputs, full_mask)[..., :population, :]
    actual = module(inputs, mask)

    assert actual.shape == (*batch_shape, population, 2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("population", [9, 16, 20])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_shared_decentralized_mlp_larger_population_matches_independent_inference(
    population: int,
    batch_shape: tuple[int, ...],
) -> None:
    network_kwargs = {
        "n_agent_inputs": 5,
        "n_agent_outputs": 2,
        "centralized": False,
        "share_params": True,
        "depth": 1,
        "num_cells": 16,
    }
    module = MaskedMultiAgentMLP(MultiAgentMLP(n_agents=8, **network_kwargs))
    single_agent_mlp = MultiAgentMLP(n_agents=1, **network_kwargs)
    original_state = {
        key: value.clone() for key, value in module.mlp.state_dict().items()
    }
    single_agent_mlp.load_state_dict(original_state)
    inputs = torch.randn(*batch_shape, population, 5, requires_grad=True)
    mask = torch.ones(*batch_shape, population, dtype=torch.bool)
    mask[..., ::3] = False
    mask[..., 8:16] = False
    masked_inputs = inputs.masked_fill(~mask.unsqueeze(-1), -1.0)

    # Treat each agent as a separate batch element of the same one-agent network.
    expected = single_agent_mlp(masked_inputs.unsqueeze(-2)).squeeze(-2)
    actual = module(inputs, mask)

    assert actual.shape == (*batch_shape, population, 2)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(inputs.grad[~mask], torch.zeros_like(inputs.grad[~mask]))
    assert torch.count_nonzero(inputs.grad[mask]) > 0
    assert module.mlp.n_agents == 8
    for key, value in module.mlp.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("centralized,share_params", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("population", [9, 16, 20])
def test_masked_mlp_rejects_population_above_trained_capacity(
    centralized: bool,
    share_params: bool,
    population: int,
) -> None:
    module = MaskedMultiAgentMLP(
        MultiAgentMLP(
            n_agent_inputs=5,
            n_agent_outputs=2,
            n_agents=8,
            centralized=centralized,
            share_params=share_params,
            depth=1,
            num_cells=16,
        )
    )
    with pytest.raises(ValueError, match="shape\\[-2\\]=8"):
        module(torch.zeros(population, 5), torch.ones(population, dtype=torch.bool))


def test_centralized_mlp_critic_ignores_inactive_observations_and_actions() -> None:
    config = _base_config(
        min_num_agents=1,
        max_num_agents=3,
        flex_enabled=False,
    )
    env = make_data_collection_env(config)
    try:
        critic = create_critic(env, torch.device("cpu"), config)
        base = env.reset(seed=0).unsqueeze(0)
        mask = torch.tensor([[True, False, True]])
        base.set(("agents", "mask"), mask)
        base.set(("agents", "action"), torch.zeros(1, 3, 2))

        clean = base.clone()
        clean_values = critic(clean).get(("agents", "state_action_value")).clone()

        corrupt = base.clone()
        corrupt.get(("agents", "observation", "flat"))[:, 1].fill_(10_000.0)
        corrupt.get(("agents", "action"))[:, 1].fill_(-10_000.0)
        corrupt_values = critic(corrupt).get(("agents", "state_action_value"))

        torch.testing.assert_close(clean_values, corrupt_values)

        active = corrupt.clone()
        active_mask = active.get(("agents", "mask")).clone()
        active_mask[:, 1] = True
        active.set(("agents", "mask"), active_mask)
        active_values = critic(active).get(("agents", "state_action_value"))

        assert not torch.allclose(clean_values, active_values)
    finally:
        env.close()


def test_mappo_rejects_flex_encoder() -> None:
    config = _base_config(
        min_num_agents=2,
        max_num_agents=2,
        flex_enabled=True,
        algorithm="mappo",
    )
    env = make_data_collection_env(config)
    try:
        with pytest.raises(ValueError, match=r"flex_encoder\.enabled.*false"):
            create_ppo_actor(env, torch.device("cpu"), config)

        with pytest.raises(ValueError, match=r"flex_encoder\.enabled.*false"):
            create_ppo_value_net(env, torch.device("cpu"), config)

        with pytest.raises(ValueError, match=r"flex_encoder\.enabled.*false"):
            create_ppo_loss(nn.Identity(), nn.Identity(), config)
    finally:
        env.close()


def test_mappo_components_reject_masking_required_environments() -> None:
    config = _base_config(
        min_num_agents=1,
        max_num_agents=3,
        flex_enabled=False,
        algorithm="mappo",
    )
    env = make_data_collection_env(config)
    try:
        with pytest.raises(NotImplementedError, match="masking"):
            create_ppo_actor(env, torch.device("cpu"), config)

        with pytest.raises(NotImplementedError, match="masking"):
            create_ppo_value_net(env, torch.device("cpu"), config)

        with pytest.raises(NotImplementedError, match="masking"):
            create_ppo_loss(nn.Identity(), nn.Identity(), config)
    finally:
        env.close()


def test_maddpg_loss_binds_mask_when_environment_requires_it() -> None:
    config = _base_config(min_num_agents=1, max_num_agents=3, flex_enabled=True)
    env = make_data_collection_env(config)
    try:
        algorithm = MADDPGAlgorithm(env, torch.device("cpu"), config)
        assert algorithm.loss_module.tensor_keys.mask == ("agents", "mask")
    finally:
        env.close()
