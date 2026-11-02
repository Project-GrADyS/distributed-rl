from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from flex_marl import (
    CentralizedOutput,
    FlatFieldConfig,
    MultiAgentEncoderConfig,
    MultiAgentEncoderModule,
    MultiAgentMode,
    SequentialFieldConfig,
    SequentialFieldOptions,
)
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import EnvBase
from torchrl.modules import MultiAgentMLP

from data_harvesting.utils import get_activation_class


class _TensorDictEncoderRouter(nn.Module):
    """Route named TensorDict inputs into flex-marl without changing their values."""

    def __init__(self, encoder: MultiAgentEncoderModule) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class MaskedMultiAgentMLP(nn.Module):
    """Pad inactive and missing agent slots to the MLP's trained capacity."""

    padding_value = -1.0

    def __init__(self, mlp: MultiAgentMLP) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        inputs: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        padded_inputs = inputs.masked_fill(
            ~agent_mask.unsqueeze(-1),
            self.padding_value,
        )
        num_agents = inputs.shape[-2]
        missing_agents = self.mlp.n_agents - num_agents
        if missing_agents > 0:
            padded_inputs = torch.nn.functional.pad(
                padded_inputs,
                (0, 0, 0, missing_agents),
                value=self.padding_value,
            )
        # Keep the network's fixed slot count, but match the caller's agent axis.
        # Populations above capacity still fail the underlying MLP's shape check.
        return self.mlp(padded_inputs)[..., :num_agents, :]


def _encoder_mode(network_config: Mapping[str, Any]) -> MultiAgentMode:
    if bool(network_config["centralized"]):
        return MultiAgentMode.CENTRALIZED
    if bool(network_config["share_parameters"]):
        return MultiAgentMode.SHARED
    return MultiAgentMode.INDEPENDENT


def make_flex_encoder_module(
    *,
    env: EnvBase,
    config: Mapping[str, Any],
    network_config: Mapping[str, Any],
    output_dim: int,
    output_key: tuple[str, ...],
    include_action: bool,
    encode_agent_identity: bool,
    device: torch.device,
) -> TensorDictModule:
    """Build the project-to-flex-marl integration for one actor or critic."""

    environment_config = config["environment"]
    flex_config = config["flex_encoder"]
    sequential_config = flex_config["sequential_heads"]
    flat_config = flex_config["flat_heads"]
    should_compile = bool(flex_config.get("compile", False))

    mode = _encoder_mode(network_config)

    sequential_options = SequentialFieldOptions(
        num_heads=int(sequential_config["num_heads"]),
        ff_dim=int(sequential_config["ff_dim"]),
        depth=int(sequential_config["depth"]),
        dropout=float(sequential_config["dropout"]),
        encode_agent_identity=encode_agent_identity,
    )

    fields: list[FlatFieldConfig | SequentialFieldConfig] = []
    in_keys: dict[str, tuple[str, ...]] = {
        "agent_mask": ("agents", "mask"),
    }

    for key in ("drones", "sensors"):
        path = ("agents", "observation", key)
        mask_key = f"{key}_mask"
        fields.append(
            SequentialFieldConfig(
                key=key,
                mask_key=mask_key,
                input_size=env.observation_spec[path].shape[-1],
                output_size=int(sequential_config["embed_dim"]),
                sequential_options=sequential_options,
            )
        )
        in_keys[key] = path
        in_keys[mask_key] = ("agents", "observation", mask_key)

    if bool(environment_config["id_on_state"]):
        path = ("agents", "observation", "agent_id")
        fields.append(
            FlatFieldConfig(
                key="agent_id",
                input_size=env.observation_spec[path].shape[-1],
                output_size=int(flat_config["embed_dim"]),
                depth=int(flat_config["depth"]),
                hidden_layer_size=int(flat_config["num_cells"]),
                activation_class=get_activation_class(flat_config["activation_function"]),
                sequential_options=sequential_options,
            )
        )
        in_keys["agent_id"] = path

    if include_action:
        path = ("agents", "action")
        fields.append(
            FlatFieldConfig(
                key="action",
                input_size=env.full_action_spec[path].shape[-1],
                output_size=int(flat_config["embed_dim"]),
                depth=int(flat_config["depth"]),
                hidden_layer_size=int(flat_config["num_cells"]),
                activation_class=get_activation_class(flat_config["activation_function"]),
                sequential_options=sequential_options,
            )
        )
        in_keys["action"] = path

    encoder = MultiAgentEncoderModule(
        MultiAgentEncoderConfig(
            fields=tuple(fields),
            num_agents=int(environment_config["max_num_agents"]),
            mode=mode,
            agent_mask_key="agent_mask",
            output_dim=output_dim,
            mix_layer_depth=int(flex_config["mix_layer_depth"]),
            mix_layer_num_cells=int(flex_config["mix_layer_num_cells"]),
            mix_activation_class=get_activation_class(flex_config["mix_activation_function"]),
            centralized_output=CentralizedOutput.BROADCAST,
        ),
        device=device,
        run_pre_forward_checks=not should_compile,
    )

    if should_compile:
        encoder = torch.compile(encoder)

    return TensorDictModule(
        _TensorDictEncoderRouter(encoder),
        in_keys=in_keys,
        out_keys=[output_key],
        out_to_in_map=True,
    )
