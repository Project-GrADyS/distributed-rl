import dataclasses
import math
import random
from typing import Optional

import numpy as np
import torch
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from data_harvesting.environment import EndCause
from data_harvesting.environment.gradys_env import BaseGrADySEnvironment
from .death import DeathScheduler
from .metrics import make_data_collection_metrics_spec
from .protocols import DroneProtocol, SensorProtocol

@dataclasses.dataclass(slots=True)
class EpisodeAgentState:
    """Bookkeeping for a single stable agent slot in the current episode."""

    slot_index: int
    name: str
    exists: bool = False
    """Whether this agent slot is occupied by an actual agent in the current episode. If False, this slot is a
    placeholder that gets truncated immediately at reset and should not be filled with meaningful data."""

    active: bool = False
    """Whether this agent slot is currently active and should be stepped. If False, this agent should not be stepped or
     filled with meaningful data, but it may still exist in the simulation if it was active in the past"""

    node_id: int | None = None

@dataclasses.dataclass
class DataCollectionEnvironmentConfig:
    """Configuration for GrADyS environment (only 'relative' observation mode retained)."""

    render_mode: Optional[str] = None  # "visual" | "console"
    algorithm_iteration_interval: float = 0.5
    # Number of drone agents is sampled from [min_num_agents, max_num_agents].
    # To fix the number, set min_num_agents == max_num_agents.
    min_num_agents: int = 1
    max_num_agents: int = 1
    # How to sample the number of agents: uniformly, or with probability proportional to 1 / n.
    agent_count_sampling: str = "uniform"
    # Number of sensors is always sampled each reset from [min_num_sensors, max_num_sensors].
    # To fix the number, set min_num_sensors == max_num_sensors.
    min_num_sensors: int = 2
    max_num_sensors: int = 2
    scenario_size: float = 100
    max_episode_length: int = 500
    max_seconds_stalled: int = 30
    communication_range: float = 20
    state_num_closest_sensors: int = 2
    state_num_closest_drones: int = 2
    id_on_state: bool = True
    min_sensor_priority: float = 0.1
    max_sensor_priority: float = 1
    full_random_drone_position: bool = False
    reward: str = 'punish'  # Fixed reward mode: punish
    speed_action: bool = True
    end_when_all_collected: bool = True
    prevent_last_agent_death: bool = True


class DataCollectionEnvironment(BaseGrADySEnvironment, EnvBase):
    """
    A specialized environment for data collection in simulations, extending the GrADySEnvironment.
    This environment simulates sensor data collection with autonomous agents.
    Per-episode agent state is centralized in `episode_agents`.
    """

    _simulation_configuration: SimulationConfiguration

    batch_locked: bool = True

    def __init__(
        self,
        config: DataCollectionEnvironmentConfig,
        *,
        death_scheduler: DeathScheduler,
        device=None,
    ):
        BaseGrADySEnvironment.__init__(self, config.algorithm_iteration_interval, visual_mode=(config.render_mode == "visual"))
        EnvBase.__init__(self, device=device)

        self.render_mode = config.render_mode
        self.algorithm_iteration_interval = config.algorithm_iteration_interval

        if config.min_num_sensors > config.max_num_sensors:
            raise ValueError("min_num_sensors cannot be greater than max_num_sensors.")

        self.min_num_sensors = config.min_num_sensors
        self.max_num_sensors = config.max_num_sensors
        self.min_num_agents = config.min_num_agents
        self.max_num_agents = config.max_num_agents
        if self.min_num_agents < 1:
            raise ValueError("min_num_agents must be at least 1.")
        if self.min_num_agents > self.max_num_agents:
            raise ValueError("min_num_agents cannot be greater than max_num_agents.")
        if config.agent_count_sampling not in {"uniform", "inverse"}:
            raise ValueError("agent_count_sampling must be either 'uniform' or 'inverse'.")
        self.agent_count_sampling = config.agent_count_sampling

        self.scenario_size = config.scenario_size
        self.max_episode_length = config.max_episode_length
        self.max_seconds_stalled = config.max_seconds_stalled
        self.communication_range = config.communication_range
        self.state_num_closest_sensors = config.state_num_closest_sensors
        self.state_num_closest_drones = config.state_num_closest_drones
        self.id_on_state = config.id_on_state
        self.min_sensor_priority = config.min_sensor_priority
        self.max_sensor_priority = config.max_sensor_priority
        self.full_random_drone_position = config.full_random_drone_position
        if config.reward != "punish":
            raise ValueError("Only reward='punish' is supported.")
        self.speed_action = config.speed_action
        self.end_when_all_collected = config.end_when_all_collected
        self.death_scheduler = death_scheduler
        self.prevent_last_agent_death = config.prevent_last_agent_death

        self.possible_agents = [f"drone{i}" for i in range(self.max_num_agents)]
        self.group_map = {"agents": self.possible_agents}

        self.episode_agents: list[EpisodeAgentState] = []
        self.active_num_sensors: int = -1
        self.episode_duration = 0
        self.stall_duration = 0
        self.reward_sum = 0.0
        self.max_reward = -math.inf
        self.collection_times: list[float] = []

        self._metrics_spec = make_data_collection_metrics_spec()
        self._info_keys = tuple(metric.key for metric in self._metrics_spec)
        self._extra_info_keys = ("num_sensors", "num_agents")
        self._all_info_keys = self._info_keys + self._extra_info_keys

        self._simulation_configuration = SimulationConfiguration(
            debug=False,
            execution_logging=False,
            duration=self.max_episode_length,
        )

        self._build_specs()
        self._cached_reset_zero = self.full_observation_spec.zero()
        self._cached_reset_zero.update(self.full_done_spec.zero())

        self._cached_step_zero = self.full_observation_spec.zero()
        self._cached_step_zero.update(self.full_reward_spec.zero())
        self._cached_step_zero.update(self.full_done_spec.zero())

    def _existing_episode_agents(self) -> list[EpisodeAgentState]:
        """Return episode slots that exist in the current episode, in slot order."""
        return [agent for agent in self.episode_agents if agent.exists]

    def _active_episode_agents(self) -> list[EpisodeAgentState]:
        """Return the episode slots that are currently active and should act."""
        return [agent for agent in self.episode_agents if agent.exists and agent.active]

    def _inactive_existing_episode_agents(self) -> list[EpisodeAgentState]:
        """Return real episode slots that are currently inactive."""
        return [agent for agent in self.episode_agents if agent.exists and not agent.active]

    def _agent_slot_tensor(self, agents: list[EpisodeAgentState]) -> torch.Tensor:
        """Return the given episode agents' slot indices on the environment device."""
        return torch.tensor([agent.slot_index for agent in agents], device=self.device, dtype=torch.long)

    def _build_simulation(self, builder: SimulationBuilder):
        """
        Set up the GrADyS-SIM NextGen simulation environment with the provided configuration.

        Args:
            builder (SimulationBuilder): Builder object for setting up the simulation.
        """
        # Adding necessary handlers to the simulation builder
        builder.add_handler(CommunicationHandler(CommunicationMedium(
            transmission_range=self.communication_range
        )))
        builder.add_handler(MobilityHandler(MobilityConfiguration(
            update_rate=self.algorithm_iteration_interval / 2
        )))
        builder.add_handler(TimerHandler())
        if self.render_mode == "visual":
            builder.add_handler(VisualizationHandler(VisualizationConfiguration(
                open_browser=False,
                x_range=(-self.scenario_size, self.scenario_size),
                y_range=(-self.scenario_size, self.scenario_size),
                z_range=(0, self.scenario_size),
            )))


        self.sensor_node_ids = []
        SensorProtocol.min_priority = self.min_sensor_priority
        SensorProtocol.max_priority = self.max_sensor_priority

        for i in range(self.active_num_sensors):
            self.sensor_node_ids.append(builder.add_node(SensorProtocol, (
                random.uniform(-self.scenario_size, self.scenario_size),
                random.uniform(-self.scenario_size, self.scenario_size),
                0
            )))

        DroneProtocol.speed_action = self.speed_action
        DroneProtocol.algorithm_interval = self.algorithm_iteration_interval

        # The episode state keeps stable slot identity; only existing agents get simulator nodes.
        for agent in self._existing_episode_agents():
            if self.full_random_drone_position:
                agent.node_id = builder.add_node(DroneProtocol, (
                    random.uniform(-self.scenario_size, self.scenario_size),
                    random.uniform(-self.scenario_size, self.scenario_size),
                    0
                ))
            else:
                agent.node_id = builder.add_node(DroneProtocol, (
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    0
                ))

        self.simulator = builder.build()

    def _build_specs(self) -> None:
        device = self.device

        action_dim = 2 if self.speed_action else 1
        sensors_shape = (self.max_num_agents, self.state_num_closest_sensors, 2)
        drones_shape = (self.max_num_agents, self.state_num_closest_drones, 2)
        sensors_mask_shape = (self.max_num_agents, self.state_num_closest_sensors)
        drones_mask_shape = (self.max_num_agents, self.state_num_closest_drones)
        agent_id_shape = (self.max_num_agents, 1)
        mask_shape = (self.max_num_agents,)
        action_shape = (self.max_num_agents, action_dim)
        reward_shape = (self.max_num_agents, 1)
        done_shape = (self.max_num_agents, 1)

        obs_inner = {
            "sensors": Bounded(
                -torch.ones(sensors_shape, device=device),
                torch.ones(sensors_shape, device=device),
                sensors_shape,
                dtype=torch.float32,
                device=device,
            ),
            "drones": Bounded(
                -torch.ones(drones_shape, device=device),
                torch.ones(drones_shape, device=device),
                drones_shape,
                dtype=torch.float32,
                device=device,
            ),
            "sensors_mask": Categorical(
                n=2,
                shape=sensors_mask_shape,
                dtype=torch.bool,
                device=device,
            ),
            "drones_mask": Categorical(
                n=2,
                shape=drones_mask_shape,
                dtype=torch.bool,
                device=device,
            ),
        }
        if self.id_on_state:
            obs_inner["agent_id"] = Bounded(
                torch.zeros(agent_id_shape, device=device),
                torch.ones(agent_id_shape, device=device),
                agent_id_shape,
                dtype=torch.float32,
                device=device,
            )

        observation_spec = Composite(
            {
                # Episode provenance for replay; never part of policy inputs.
                "initial_team_size": Unbounded(shape=(1,), dtype=torch.int64, device=device),
                "agents": Composite(
                    {
                        "observation": Composite(obs_inner, device=device),
                        "mask": Categorical(
                            n=2,
                            shape=mask_shape,
                            dtype=torch.bool,
                            device=device,
                        ),
                        "info": Composite(
                            {
                                key: Unbounded(
                                    shape=(self.max_num_agents,),
                                    device=device,
                                    dtype=torch.float32,
                                )
                                for key in self._all_info_keys
                            },
                            device=device,
                        ),
                    },
                    device=device,
                )
            },
            device=device,
        )

        action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": Bounded(
                            torch.zeros(action_shape, device=device),
                            torch.ones(action_shape, device=device),
                            action_shape,
                            dtype=torch.float32,
                            device=device,
                        )
                    },
                    device=device,
                )
            },
            device=device,
        )

        reward_spec = Composite(
            {
                "agents": Composite(
                    {
                        "reward": Unbounded(
                            shape=reward_shape, device=device, dtype=torch.float32
                        )
                    },
                    device=device,
                )
            },
            device=device,
        )

        done_spec = Composite(
            {
                "done": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "terminated": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "truncated": Categorical(
                    n=2, shape=(1,), dtype=torch.bool, device=device
                ),
                "agents": Composite(
                    {
                        "done": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                        "terminated": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                        "truncated": Categorical(
                            n=2, shape=done_shape, dtype=torch.bool, device=device
                        ),
                    },
                    device=device,
                ),
            },
            device=device,
        )

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        actions = tensordict.get(("agents", "action"))
        stepped_agents = self._active_episode_agents()
        self._apply_actions(actions)

        # We record the collected sensors before stepping the simulation so we can compare with
        # that figure after the step to see if any new sensors were collected
        collected_before = self._get_sensor_collected()

        status = self.step_simulation()
        end_cause = EndCause.NONE

        collected_after = self._get_sensor_collected()

        # Update the stall counter based on whether new sensors were collected
        # A stall is when the agents fail to collect any new sensors. If the stall
        # persists for too long, the episode ends.
        self._update_stall(collected_before, collected_after)

        # Check if all sensors have been collected
        all_sensors_collected = sum(collected_after) == self.active_num_sensors

        reward = self._compute_reward(collected_before, collected_after)
        self._update_collection_times(collected_after)
        dying_agents = self._get_dying_agents(stepped_agents)
        self._deactivate_agents(dying_agents)
        self._reward_sum_update(reward, stepped_agents)

        active_agents_after = self._active_episode_agents()
        if all_sensors_collected and self.end_when_all_collected:
            end_cause = EndCause.ALL_COLLECTED
        elif not active_agents_after:
            end_cause = EndCause.ALL_AGENTS_INACTIVE
        elif self.stall_duration > self.max_seconds_stalled:
            end_cause = EndCause.STALLED
        elif status.has_ended:
            end_cause = EndCause.TIMEOUT

        # Filling the output tensordict for the step
        tensordict_out = self._cached_step_zero.clone()
        self._fill_observation(tensordict_out, self._observe_simulation())
        self._fill_rewards(tensordict_out, reward, stepped_agents)
        self._fill_done(tensordict_out, end_cause, dying_agents)
        self._fill_info(tensordict_out, sum(collected_after), end_cause, end_cause != EndCause.NONE)
        return tensordict_out

    def _reset_statistics(self):
        """
        Resets the statistics for a new episode.
        """
        self.episode_duration = 0
        self.stall_duration = 0
        self.reward_sum = 0
        self.max_reward = -math.inf
        self.collection_times = [self.max_episode_length for _ in range(self.active_num_sensors)]

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # Picking number of sensors and drones for this episode
        self.active_num_sensors = random.randint(self.min_num_sensors, self.max_num_sensors)
        agent_counts = range(self.min_num_agents, self.max_num_agents + 1)
        if self.agent_count_sampling == "inverse":
            num_active_agents = random.choices(
                agent_counts,
                weights=[1.0 / count for count in agent_counts],
                k=1,
            )[0]
        else:
            num_active_agents = random.randint(self.min_num_agents, self.max_num_agents)
        self.episode_agents = [
            EpisodeAgentState(
                slot_index=i,
                name=f"drone{i}",
                exists=i < num_active_agents,
                active=i < num_active_agents,
            )
            for i in range(self.max_num_agents)
        ]
        self.death_scheduler.reset()

        self._reset_statistics()

        self.reset_simulation(self._simulation_configuration)

        active_agents = self._active_episode_agents()
        max_ready_steps = len(active_agents) * 10  # Arbitrary large number of steps to wait for drones to be ready
        ready_steps = 0
        while not self._all_active_drones_ready():
            status = self.step_simulation()
            ready_steps += 1
            if status.has_ended:
                raise RuntimeError("Simulation ended before all drones received initial telemetry")
            if ready_steps >= max_ready_steps:
                raise RuntimeError("Timed out waiting for initial telemetry for all drones")

        all_obs = self._observe_simulation()

        # The initial observation has to contain observations for all possible agents
        # We repeat the last active agent's observation for the inactive agents
        # This is not a problem because these agents will be truncated immediately
        active_agents = self._active_episode_agents()
        if not active_agents:
            raise RuntimeError("No active agents after reset")
        # TorchRL still expects a dense agent axis, so padded slots reuse the last active observation and are
        # immediately truncated.
        for agent in self.episode_agents:
            if not agent.exists:
                all_obs[agent.name] = all_obs[active_agents[-1].name]

        tensordict_out = self._cached_reset_zero.clone()
        self._fill_observation(tensordict_out, all_obs)
        self._fill_done(tensordict_out, EndCause.NONE, [])
        self._fill_info(tensordict_out, 0, EndCause.NONE, False)
        return tensordict_out

    def _all_active_drones_ready(self) -> bool:
        """Return True once every active agent has received initial telemetry."""
        return all(
            getattr(self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol, "ready", False)
            for agent in self._active_episode_agents()
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.reset(seed=seed)

    def _apply_actions(self, actions: torch.Tensor) -> None:
        active_agents = self._active_episode_agents()
        actions_cpu = actions.detach().cpu()

        for agent in active_agents:
            agent_node = self.simulator.get_node(agent.node_id)
            action = actions_cpu[agent.slot_index].tolist()
            agent_node.protocol_encapsulator.protocol.act(action, self.scenario_size)

    def _get_dying_agents(
        self,
        stepped_agents: list[EpisodeAgentState],
    ) -> list[EpisodeAgentState]:
        active_slots = [agent.slot_index for agent in stepped_agents]
        scheduled_slots = self.death_scheduler.get_deaths(
            self.episode_duration,
            active_slots,
        )
        if scheduled_slots is None:
            return []
        if not isinstance(scheduled_slots, list):
            raise TypeError("DeathScheduler.get_deaths() must return a list or None.")
        if any(
            isinstance(slot, bool) or not isinstance(slot, int)
            for slot in scheduled_slots
        ):
            raise TypeError("Death scheduler slot IDs must be integers.")
        if len(scheduled_slots) != len(set(scheduled_slots)):
            raise ValueError("Death scheduler returned duplicate agent slot IDs.")

        active_by_slot = {agent.slot_index: agent for agent in stepped_agents}
        invalid_slots = [slot for slot in scheduled_slots if slot not in active_by_slot]
        if invalid_slots:
            raise ValueError(
                f"Death scheduler returned non-active agent slot IDs: {invalid_slots}"
            )

        dying_agents = [active_by_slot[slot] for slot in scheduled_slots]
        if self.prevent_last_agent_death and len(dying_agents) == len(stepped_agents):
            # Keep one active slot alive to avoid ending episodes solely from scheduled deaths.
            return dying_agents[:-1]
        return dying_agents

    def _deactivate_agents(self, agents: list[EpisodeAgentState]) -> None:
        for agent in agents:
            if not agent.active or agent.node_id is None:
                continue
            protocol = self.simulator.get_node(agent.node_id).protocol_encapsulator.protocol
            protocol.die()
            agent.active = False

    def _get_sensor_collected(self) -> list[bool]:
        return [
            self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        ]

    def _update_stall(self, collected_before: list[bool], collected_after: list[bool]) -> None:
        if sum(collected_after) > sum(collected_before):
            self.stall_duration = 0
        else:
            self.stall_duration += self.algorithm_iteration_interval

        self.episode_duration += 1

    def _compute_reward(self, collected_before: list[bool], collected_after: list[bool]) -> float:
        before = sum(collected_before)
        after = sum(collected_after)
        if after > before:
            return float((after - before) * 10)
        remaining = self.active_num_sensors - after
        return float(-(remaining) / max(1, self.active_num_sensors))

    def _update_collection_times(self, collected_after: list[bool]) -> None:
        current_timestamp = self.episode_duration * self.algorithm_iteration_interval
        for index, sensor_id in enumerate(self.sensor_node_ids):
            if collected_after[index] and self.collection_times[index] == self.max_episode_length:
                self.collection_times[index] = current_timestamp

    def _reward_sum_update(self, reward: float, stepped_agents: list[EpisodeAgentState]) -> None:
        if stepped_agents:
            self.reward_sum += reward
            self.max_reward = max(self.max_reward, reward)

    def _observe_simulation(self) -> dict:
        """
        Extracts information from the simulation to form the observation for each agent.
        Each agent's observation includes the positions of the closest unvisited sensors
        and the closest other drones, normalized within the scenario size.
        Returns:
            dict: A dictionary containing observations for each agent.
        """
        sensor_nodes = np.array([
            self.simulator.get_node(sensor_id).position[:2]
            for sensor_id in self.sensor_node_ids
        ])
        unvisited_sensor_mask = np.array([
            not self.simulator.get_node(sensor_id)
            .protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        ])
        unvisited_sensor_nodes = sensor_nodes[unvisited_sensor_mask]

        existing_agents = self._active_episode_agents()
        if not existing_agents:
            return {}
        agent_nodes = np.array([
            self.simulator.get_node(agent.node_id).position[:2]
            for agent in existing_agents
        ])

        max_distance = self.scenario_size * 2

        state = {}
        for agent_index, agent in enumerate(existing_agents):
            agent_position = agent_nodes[agent_index]

            sensor_distances = np.linalg.norm(
                unvisited_sensor_nodes - agent_position, axis=1
            ) if len(unvisited_sensor_nodes) else np.array([])
            sorted_sensor_indices = np.argsort(sensor_distances)

            closest_unvisited_sensors = np.full(
                (self.state_num_closest_sensors, 2), -1.0
            )
            sensors_mask = np.zeros(self.state_num_closest_sensors, dtype=np.bool_)
            sensor_count = min(
                len(sorted_sensor_indices), self.state_num_closest_sensors
            )
            if sensor_count:
                closest_unvisited_sensors[:sensor_count] = unvisited_sensor_nodes[
                    sorted_sensor_indices[:sensor_count]
                ]
                sensors_mask[:sensor_count] = True

            agent_distances = np.linalg.norm(agent_nodes - agent_position, axis=1)
            sorted_agent_indices = np.argsort(agent_distances)

            closest_agents = np.full((self.state_num_closest_drones, 2), -1.0)
            drones_mask = np.zeros(self.state_num_closest_drones, dtype=np.bool_)
            drone_count = min(
                max(len(sorted_agent_indices) - 1, 0),
                self.state_num_closest_drones,
            )
            if drone_count:
                closest_agents[:drone_count] = agent_nodes[
                    sorted_agent_indices[1:drone_count + 1]
                ]
                drones_mask[:drone_count] = True

            if drone_count:
                closest_agents[:drone_count] = (
                    closest_agents[:drone_count] - agent_position + max_distance
                ) / (max_distance * 2)

            if sensor_count:
                closest_unvisited_sensors[:sensor_count] = (
                    closest_unvisited_sensors[:sensor_count] - agent_position + max_distance
                ) / (max_distance * 2)

            state[agent.name] = {
                "drones": closest_agents,
                "drones_mask": drones_mask,
                "sensors": closest_unvisited_sensors,
                "sensors_mask": sensors_mask,
            }
            if self.id_on_state:
                state[agent.name]["agent_id"] = np.array([
                    agent_index / (self.max_num_agents - 1) if self.max_num_agents > 1 else 0
                ])
        return state

    def _fill_observation(
        self,
        td: TensorDictBase,
        observation_dict: dict,
    ) -> None:
        """
        Fills the input tensordict with observations from the simulation.
        Args:
            td (TensorDictBase): The tensordict to fill with observations.
            observation_dict (Optional[dict]): Precomputed observations.
        """
        td.get("initial_team_size").fill_(len(self._existing_episode_agents()))
        agents_td = td.get("agents")
        obs_td = agents_td.get("observation")

        sensors = obs_td.get("sensors")
        drones = obs_td.get("drones")
        sensors_mask = obs_td.get("sensors_mask")
        drones_mask = obs_td.get("drones_mask")
        sensors.zero_()
        drones.zero_()
        sensors_mask.zero_()
        drones_mask.zero_()

        if self.id_on_state:
            agent_id = obs_td.get("agent_id")
            agent_id.zero_()

        mask = agents_td.get("mask")
        mask.zero_()

        active_agents = self._active_episode_agents()
        if not active_agents:
            return

        active_slots = self._agent_slot_tensor(active_agents)
        sensors.index_copy_(
            0,
            active_slots,
            torch.as_tensor(
                np.stack([observation_dict[agent.name]["sensors"] for agent in active_agents]),
                device=self.device,
                dtype=sensors.dtype,
            ),
        )
        drones.index_copy_(
            0,
            active_slots,
            torch.as_tensor(
                np.stack([observation_dict[agent.name]["drones"] for agent in active_agents]),
                device=self.device,
                dtype=drones.dtype,
            ),
        )
        sensors_mask.index_copy_(
            0,
            active_slots,
            torch.as_tensor(
                np.stack([
                    observation_dict[agent.name]["sensors_mask"]
                    for agent in active_agents
                ]),
                device=self.device,
                dtype=sensors_mask.dtype,
            ),
        )
        drones_mask.index_copy_(
            0,
            active_slots,
            torch.as_tensor(
                np.stack([
                    observation_dict[agent.name]["drones_mask"]
                    for agent in active_agents
                ]),
                device=self.device,
                dtype=drones_mask.dtype,
            ),
        )
        if self.id_on_state:
            agent_id.index_copy_(
                0,
                active_slots,
                torch.as_tensor(
                    np.stack([observation_dict[agent.name]["agent_id"] for agent in active_agents]),
                    device=self.device,
                    dtype=agent_id.dtype,
                ),
            )
        mask.index_fill_(0, active_slots, True)

    def _fill_rewards(
        self,
        td: TensorDictBase,
        reward_value: float,
        rewarded_agents: list[EpisodeAgentState],
    ) -> None:
        reward = td.get(("agents", "reward"))
        reward.zero_()
        if rewarded_agents:
            rewarded_slots = self._agent_slot_tensor(rewarded_agents)
            reward.index_fill_(0, rewarded_slots, reward_value)

    def _fill_done(
        self,
        td: TensorDictBase,
        end_cause: EndCause,
        dying_agents: list[EpisodeAgentState],
    ) -> None:
        done = td.get(("agents", "done"))
        terminated = td.get(("agents", "terminated"))
        truncated = td.get(("agents", "truncated"))
        done.zero_()
        terminated.zero_()
        truncated.zero_()

        active_agents = self._active_episode_agents()
        dead_agents = [
            agent for agent in self._inactive_existing_episode_agents()
            if agent not in dying_agents
        ]
        inactive_agents = [agent for agent in self.episode_agents if not agent.exists]
        active_slots = self._agent_slot_tensor(active_agents)
        dying_slots = self._agent_slot_tensor(dying_agents)
        dead_slots = self._agent_slot_tensor(dead_agents)
        inactive_slots = self._agent_slot_tensor(inactive_agents)

        if end_cause != EndCause.NONE and active_slots.numel() > 0:
            terminated.index_fill_(0, active_slots, True)
            done.index_fill_(0, active_slots, True)
        if dying_slots.numel() > 0:
            terminated.index_fill_(0, dying_slots, True)
            done.index_fill_(0, dying_slots, True)

        # Previously dead real slots and padded slots are bookkeeping only.
        truncated.index_fill_(0, dead_slots, True)
        done.index_fill_(0, dead_slots, True)
        truncated.index_fill_(0, inactive_slots, True)
        done.index_fill_(0, inactive_slots, True)

        episode_done = end_cause != EndCause.NONE
        td.set("done", torch.tensor([episode_done], device=self.device))
        td.set("terminated", torch.tensor([episode_done], device=self.device))
        td.set("truncated", torch.tensor([False], device=self.device))

    def _fill_info(self, td: TensorDictBase, num_collected: int, end_cause: EndCause, ended: bool) -> None:
        all_collected = num_collected == self.active_num_sensors
        info_td = td.get(("agents", "info"))
        for key in self._all_info_keys:
            info_td.get(key).zero_()

        existing_agents = self._existing_episode_agents()
        if not ended or not existing_agents:
            return
        existing_slots = self._agent_slot_tensor(existing_agents)

        avg_reward = self.reward_sum / max(1, self.episode_duration)
        avg_collection_time = (
            sum(self.collection_times) / self.active_num_sensors
            if self.active_num_sensors > 0
            else 0.0
        )
        completion_time = (
            self.max_episode_length
            if not all_collected
            else self.simulator._current_timestamp
        )

        num_dead = len([agent for agent in existing_agents if not agent.active])

        metrics = {
            "avg_reward": avg_reward,
            "max_reward": self.max_reward,
            "sum_reward": self.reward_sum,
            "avg_collection_time": avg_collection_time,
            "episode_duration": float(self.episode_duration),
            "completion_time": float(completion_time),
            "all_collected": float(int(all_collected)),
            "num_collected": float(num_collected),
            "cause": float(end_cause.value),
            "num_sensors": float(self.active_num_sensors),
            "num_agents": float(len(existing_agents)),
            "num_dead": float(num_dead),
        }

        for key, value in metrics.items():
            info_td.get(key).index_fill_(0, existing_slots, value)

    def close(self, *, raise_if_closed: bool = True):
        super().close(raise_if_closed=raise_if_closed)
        self.finalize_simulation()
