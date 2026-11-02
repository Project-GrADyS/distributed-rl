import torch
from typing import Dict, Any
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import (
    ProbabilisticActor,
    MultiAgentMLP,
    AdditiveGaussianModule,
    NormalParamExtractor,
)
from torchrl.modules.distributions import TanhNormal
from data_harvesting.encoding import MaskedMultiAgentMLP, make_flex_encoder_module
from data_harvesting.environment import requires_masking
from data_harvesting.utils import collapse_shared_action_bound, get_faster_tanh_delta, get_activation_class

def create_mlp_module(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    activation_class = get_activation_class(config["actor"]["activation_function"])
    policy_net = MaskedMultiAgentMLP(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[("agents", "observation", "flat")].shape[-1],
            n_agent_outputs=env.full_action_spec[("agents", "action")].shape[-1],
            n_agents=config["environment"]["max_num_agents"],
            centralised=config["actor"]["centralized"],
            share_params=config["actor"]["share_parameters"],
            device=device,
            depth=config["actor"]["network_depth"],
            num_cells=config["actor"]["network_width"],
            activation_class=activation_class,
        )
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation", "flat"), ("agents", "mask")],
        out_keys=[("agents", "param")],
    )
    return policy_module

def create_flex_policy_module(env: EnvBase, config: Dict[str, Any], device: torch.device) -> TensorDictModule:
    return make_flex_encoder_module(
        env=env,
        config=config,
        network_config=config["actor"],
        output_dim=env.full_action_spec[("agents", "action")].shape[-1],
        output_key=("agents", "param"),
        include_action=False,
        encode_agent_identity=False,
        device=device,
    )

def create_actor(
    env: EnvBase,
    device: torch.device,
    config: Dict[str, Any],
) -> ProbabilisticActor:
    """Create the deterministic (Tanh-squashed delta) multi-agent actor.

    Args:
        env: TorchRL environment providing observation and action specs.
        device: Target device for modules.
        config: Hierarchical configuration dictionary.

    Returns:
        ProbabilisticActor: Actor producing actions under a TanhDelta distribution.
    """
    policy_module = (
        create_flex_policy_module(env, config, device)
        if config["flex_encoder"]["enabled"]
        else create_mlp_module(env, config, device)
    )

    # Shared limits broadcast over agent slots; the full spec retains the slot count.
    high = collapse_shared_action_bound(
        env.full_action_spec_unbatched["agents", "action"].space.high.to(device)
    )
    low = collapse_shared_action_bound(
        env.full_action_spec_unbatched["agents", "action"].space.low.to(device)
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec[("agents", "action")],
        in_keys=[("agents", "param")],
        out_keys=[("agents", "action")],
        distribution_class=get_faster_tanh_delta(high, low),
        distribution_kwargs={
            "low": low,
            "high": high,
        },
        return_log_prob=False,
    )
    return policy


def create_exploratory_actor(
    actor: ProbabilisticActor,
    device: torch.device,
    config: Dict[str, Any],
) -> tuple[TensorDictModule, AdditiveGaussianModule]:
    """
    Creates an exploratory actor by adding Gaussian noise to the actions of the given actor.
    :param actor: The base actor.
    :param device: The device to run the exploratory actor on.
    :param config: Configuration dictionary.
    :return: The exploratory actor and the exploration noise module.
    """
    exploration_noise = AdditiveGaussianModule(
        spec=actor.spec,
        annealing_num_steps=config["training"]["exploration_annealing_steps"],
        action_key=("agents", "action"),
        sigma_init=config["training"]["exploration_sigma_init"],
        sigma_end=config["training"]["exploration_sigma_end"],
        device=device
    )

    return TensorDictSequential(
        actor,
        exploration_noise
    ), exploration_noise


def create_ppo_actor(
    env: EnvBase,
    device: torch.device,
    config: Dict[str, Any],
) -> ProbabilisticActor:
    """Create a multi-agent PPO actor with a TanhNormal squashed Gaussian policy.

    Returns a ProbabilisticActor that emits actions within the environment bounds and stores
    the log probability under the key ("agents", "sample_log_prob") for PPO updates.
    """
    if config["flex_encoder"]["enabled"]:
        raise ValueError("MAPPO requires flex_encoder.enabled to be false.")
    if requires_masking(config):
        raise NotImplementedError("PPO Actor does not support environments that require masking.")

    activation_class = get_activation_class(config["actor"]["activation_function"])
    action_dim = env.full_action_spec[("agents", "action")].shape[-1]

    # Policy backbone produces concatenated normal params (mean, std) per action dim
    policy_backbone = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[("agents", "observation", "flat")].shape[-1],
            n_agent_outputs=action_dim * 2,
            n_agents=config["environment"]["max_num_agents"],
            centralised=config["actor"]["centralized"],
            share_params=config["actor"]["share_parameters"],
            device=device,
            depth=config["actor"]["network_depth"],
            num_cells=config["actor"]["network_width"],
            activation_class=activation_class,
        ),
        NormalParamExtractor()
    )

    policy_module = TensorDictModule(
        module=policy_backbone,
        in_keys=[("agents", "observation", "flat")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Squashed Gaussian policy with log-prob output
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        spec=env.full_action_spec[("agents", "action")],
        distribution_class=TanhNormal,
        # TanhNormal will be created from loc/scale; env spec handles bounds
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return policy
