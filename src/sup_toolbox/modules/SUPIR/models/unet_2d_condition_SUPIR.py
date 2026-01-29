# This file contains code derived from multiple sources:
# - Original UNet/ControlNet architecture from Diffusers (HuggingFace Team, Apache 2.0 License).
# - Concepts and code, such as ZeroSFT, from the SUPIR project (SupPixel Pty Ltd).
#
# Copyright (c) 2024, SupPixel Pty Ltd. for SUPIR-specific components.
# Copyright 2024, The HuggingFace Team for original Diffusers components.
# Modifications and adaptations to this file are Copyright (c) 2025 by DEVAIEXP.
#
# Due to the inclusion of proprietary components and concepts from the SUPIR project,
# this entire module is governed by the SUPIR Software License Agreement.
# Its use is strictly limited to NON-COMMERCIAL purposes.
#
# For the full license terms, please see the LICENSE_SUPIR.md file in this directory.

import copy
import inspect
import json
import os
from dataclasses import dataclass
from functools import partial  # Import partial
from typing import Any, Dict, Optional, Tuple, Union

import accelerate
import torch
import torch.nn as nn

# Diffusers Imports
from diffusers import __version__
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)

# Import loading utilities
from diffusers.models.model_loading_utils import (
    _determine_device_map,
    _fetch_index_file,
    _get_model_file,
    load_state_dict,
)
from diffusers.models.modeling_utils import (
    ContextManagers,
    ModelMixin,
    no_init_weights,
)
from diffusers.models.unets.unet_2d_blocks import Attention as DiffusersAttention
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnUpBlock2D,
    UpBlock2D,
    get_down_block,
    get_mid_block,
)
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    USE_PEFT_BACKEND,
    WEIGHTS_NAME,
    BaseOutput,
    _add_variant,
    _get_checkpoint_shard_files,
    deprecate,
    is_accelerate_available,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import apply_freeu
from einops import rearrange
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file

from ...diffusers_local.single_file_model import FromOriginalModelMixin


if is_torch_version(">=", "1.9.0"):
    _LOW_CPU_MEM_USAGE_DEFAULT = True
else:
    _LOW_CPU_MEM_USAGE_DEFAULT = False

logger = logging.get_logger(__name__)


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer with 32 groups.
    This specific implementation does not force float32 conversion internally,
    relying on the underlying PyTorch implementation to handle dtypes.
    """

    def forward(self, x):
        return super().forward(x)


def normalization(channels):
    """Make a standard normalization layer (GroupNorm with 32 groups)."""
    return GroupNorm32(32, channels)


class ZeroSFT(nn.Module):
    """
    ZeroSFT module. Applies Spatial Feature Transform (SFT) modulation
    based on a control tensor 'c' to an input tensor 'h_decoder', potentially using
    a skip connection 'h_skip'. It replaces the standard concatenation of
    'h_decoder' and 'h_skip' in the UNet up blocks.

    Args:
        label_nc (`int`): Number of channels in the control tensor `c`.
        norm_nc (`int`): Number of channels in the main input tensor `h_decoder`.
        concat_channels (`int`, *optional*, defaults to 0):
            Number of channels in the skip connection tensor `h_skip`. If 0, no skip connection is used.
        norm (`bool`, *optional*, defaults to `True`): Whether to apply GroupNorm.
        mask (`bool`, *optional*, defaults to `False`): Legacy masking option (not used).
    """

    def __init__(self, label_nc, norm_nc, concat_channels=0, norm=True, mask=False):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.norm = norm
        combined_channels = norm_nc + concat_channels
        if self.norm:
            self.param_free_norm = normalization(combined_channels)
        else:
            self.param_free_norm = nn.Identity()

        nhidden = 128  # internal hidden dimension
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.SiLU())
        self.zero_mul = zero_module(nn.Conv2d(nhidden, combined_channels, kernel_size=ks, padding=pw))
        self.zero_add = zero_module(nn.Conv2d(nhidden, combined_channels, kernel_size=ks, padding=pw))

        # 1x1 convolution applied to control 'c', output has 'norm_nc' channels (to match 'h')
        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.pre_concat = bool(concat_channels != 0)
        self.mask = mask

    def forward(self, c, h_skip, h_decoder=None, control_scale=1.0):
        """
        Applies the ZeroSFT operation.

        Args:
            c (`torch.Tensor`): The control tensor.
            h_skip (`torch.Tensor`, *optional*): The skip connection tensor.
            h_decoder (`torch.Tensor`): The main input tensor (e.g., from the previous UNet layer).
            control_scale (`float`, *optional*, defaults to 1.0): Scaling factor for the SFT modulation.

        Returns:
            `torch.Tensor`: The modulated tensor.
        """
        assert not self.mask, "Masking is not supported/used."

        if h_decoder is not None and self.pre_concat:
            h_raw = torch.cat([h_decoder, h_skip], dim=1)
        else:
            h_raw = h_skip

        h_intermediate = h_skip
        conv_out = self.zero_conv(c)
        if not self.mask:
            h_intermediate = h_skip + conv_out
        else:
            h_intermediate = h_skip + conv_out * torch.zeros_like(h_skip)

        if h_decoder is not None and self.pre_concat:
            h_concat = torch.cat([h_decoder, h_intermediate], dim=1)
        else:
            h_concat = h_intermediate

        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        if self.mask:
            gamma = gamma * torch.zeros_like(gamma)
            beta = beta * torch.zeros_like(beta)
        h_modulated = self.param_free_norm(h_concat) * (gamma + 1) + beta
        h_final = h_modulated
        if h_decoder is not None and not self.pre_concat:
            h_final = torch.cat([h_decoder, h_final], dim=1)

        return h_final * control_scale + h_raw * (1.0 - control_scale)


class ZeroCrossAttn(nn.Module):
    """
    ZeroCrossAttn module adapted to use diffusers Attention class and processor system.
    Applies cross-attention between an input tensor `x` and a control tensor `context`,
    adding the result back to `x`.

    Includes workarounds for potential shape mismatches.

    Args:
        context_dim (`int`): Number of channels in the control tensor `context`.
        query_dim (`int`): Number of channels in the input tensor `x`.
        heads (`int`, *optional*): Number of attention heads. Defaults to `query_dim // 64`.
        dim_head (`int`, *optional*): Dimension of each attention head. Defaults to 64.
        dropout (`float`, *optional*, defaults to 0.0): Dropout probability for the attention output.
        bias (`bool`, *optional*, defaults to `False`): Whether to use bias in linear layers of attention.
        # Removed zero_out, mask
    """

    def __init__(self, context_dim, query_dim, heads=-1, dim_head=-1, dropout=0.0, bias=False):
        super().__init__()

        # Determine heads and dim_head (similar logic, adapt if needed)
        if dim_head == -1:
            dim_head = 64
        if heads == -1:
            if query_dim % dim_head != 0:
                heads = 8  # Fallback
                dim_head = query_dim // heads
                if query_dim % heads != 0:
                    raise ValueError(f"The query dimension ({query_dim}) must be evenly divisible by the number of attention heads ({heads}).")
            else:
                heads = query_dim // dim_head

        # Use diffusers Attention
        self.attn = Attention(
            query_dim=query_dim,
            cross_attention_dim=context_dim,  # Use cross_attention_dim for context
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
        )

        self.norm1 = normalization(query_dim)
        self.norm2 = normalization(context_dim)

        # Initialize projection layers for workaround as None initially
        self._query_proj_layer = None
        self._attn_out_proj_layer = None

    def forward(self, context, x, control_scale=1.0, **cross_attention_kwargs):
        logger.debug("    --- Entering ZeroCrossAttn ---")
        logger.debug(f"      Input x shape: {x.shape}")
        logger.debug(f"      Input context shape: {context.shape}")

        x_in = x
        b, c_q, h, w = x.shape
        x_processed = x

        # Normalize inputs
        x_norm = self.norm1(x_processed)
        context_norm = self.norm2(context)

        logger.debug(f"      After Norm: x_norm={x_norm.shape}, context_norm={context_norm.shape}")

        # Reshape for Attention layer
        x_attn_in = rearrange(x_norm, "b c h w -> b (h w) c").contiguous()
        context_attn_in = rearrange(context_norm, "b c h w -> b (h w) c").contiguous()

        logger.debug(f"      Attention Input: x_attn_in={x_attn_in.shape}, context_attn_in={context_attn_in.shape}")

        # Call diffusers Attention forward
        logger.debug("      Calling self.attn...")
        attn_output = self.attn(hidden_states=x_attn_in, encoder_hidden_states=context_attn_in, attention_mask=None, **cross_attention_kwargs)
        logger.debug(f"      Attention Output shape (before reshape): {attn_output.shape}")

        # Reshape output back to image format
        attn_output = rearrange(attn_output, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        logger.debug(f"      Attention Output shape (after reshape): {attn_output.shape}")

        x_out = x_in + attn_output * control_scale
        logger.debug(f"      Final Output x_out shape: {x_out.shape}")
        logger.debug("    --- Exiting ZeroCrossAttn ---")

        return x_out

    def set_processor(self, processor: AttentionProcessor):
        self.attn.set_processor(processor)

    def get_processor(self) -> AttentionProcessor:
        return self.attn.get_processor()


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Output class for UNet2DConditionModel.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The output prediction tensor (e.g., predicted noise).
    """

    sample: torch.Tensor = None


class SUPIRUpBlock2D(UpBlock2D):
    """
    An UpBlock2D modified for SUPIR to handle ZeroSFT injection.
    It replaces the standard skip connection concatenation with a ZeroSFT module call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution_idx = kwargs.pop("resolution_idx", None)
        self.supir_active = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # Ignorado
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # Ignorado
        encoder_attention_mask: Optional[torch.Tensor] = None,  # Ignorado
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        is_freeu_enabled = getattr(self, "s1", None) and getattr(self, "s2", None) and getattr(self, "b1", None) and getattr(self, "b2", None)

        logger.debug(f"\n--- Entering SUPIRUpBlock2D (Index: {getattr(self, 'resolution_idx', 'N/A')}) ---")
        logger.debug(f"  Input hidden_states shape: {hidden_states.shape}")
        logger.debug(f"  Input res_hidden_states shapes: {[res.shape for res in res_hidden_states_tuple]}")
        if temb is not None:
            logger.debug(f"  Input temb shape: {temb.shape}")

        control_hs = None
        project_modules = None
        block_stages_map = []
        is_supir_mode = False
        block_cross_attn_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        if block_cross_attn_kwargs:
            control_hs = block_cross_attn_kwargs.pop("control_hs", None)
            project_modules = block_cross_attn_kwargs.pop("project_modules", None)
            block_stages_map = block_cross_attn_kwargs.pop("injection_stages_map", [])
            block_cross_attn_kwargs.pop("control_scale_global_dynamic", None)
            block_cross_attn_kwargs.pop("use_linear_control_scale_pipeline", None)
            block_cross_attn_kwargs.pop("injection_scales", None)

            if control_hs is not None and project_modules is not None and block_stages_map:
                logger.debug(f"  SUPIR Mode ACTIVE: Controls Provided={len(control_hs)}, ProjMods Available={len(project_modules)}")
                is_supir_mode = True
            else:
                logger.debug("  SUPIR Mode INACTIVE (control_hs, project_modules, or block_config missing)")
                is_supir_mode = False
        else:
            logger.debug("  SUPIR Mode INACTIVE (cross_attention_kwargs missing)")
            is_supir_mode = False

        current_hidden_states = hidden_states

        # ResNet Stages
        for i, resnet in enumerate(self.resnets):
            stage_map = block_stages_map[i] if i < len(block_stages_map) else {}
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            logger.debug(f"\n  --- ResNet Stage {i} ---")
            logger.debug(f"    Input hidden_states shape: {current_hidden_states.shape}")
            logger.debug(f"    Using Skip Connection (res_hidden_states) shape: {res_hidden_states.shape}")

            # Apply SFT or Concat before ResNet
            applied_sft_logic = False
            if is_supir_mode and stage_map:
                sft_proj_idx = stage_map.get("sft_proj")
                sft_ctrl_idx = stage_map.get("sft_ctrl")
                apply_sft_flag = stage_map.get("sft_active", True)
                sft_scale = stage_map.get("sft_scale", 1.0)

                logger.debug(
                    f"    Checking UpBlock[{self.resolution_idx}]_stage_{i}: ProjIdx={sft_proj_idx}, CtrlIdx={sft_ctrl_idx}, Scale={sft_scale}, Active={apply_sft_flag}"
                )
                if (
                    apply_sft_flag
                    and sft_proj_idx is not None
                    and sft_ctrl_idx is not None
                    and control_hs is not None
                    and project_modules is not None
                    and sft_ctrl_idx < len(control_hs)
                    and sft_proj_idx < len(project_modules)
                ):
                    module_sft = project_modules[sft_proj_idx]
                    control = control_hs[sft_ctrl_idx]
                    if isinstance(module_sft, ZeroSFT) and control is not None:
                        logger.debug(f"    Applying SFT (Stage {i} - Proj[{sft_proj_idx}], Ctrl[{sft_ctrl_idx}]) Scale={sft_scale}")
                        current_hidden_states = module_sft(c=control, h_skip=res_hidden_states, h_decoder=current_hidden_states, control_scale=sft_scale)
                        applied_sft_logic = True
                    else:
                        logger.warning(
                            f"SUPIRUpBlock[{self.resolution_idx}] - Initial SFT failed checks: ModType={type(module_sft)}, CtrlIsNone={control is None}"
                        )
                else:
                    if not apply_sft_flag:
                        logger.debug(f"SUPIRUpBlock[{self.resolution_idx}] - SFT not applied.")
                    else:
                        logger.warning(
                            f"SUPIRUpBlock[{self.resolution_idx}] - Initial SFT failed checks: ModType={type(module_sft)}, CtrlIsNone={control is None}"
                        )

            if not applied_sft_logic:
                logger.debug(f"    Applying Standard Concatenation (Stage {i})")
                current_hidden_states = torch.cat([current_hidden_states, res_hidden_states.to(current_hidden_states.dtype)], dim=1)
            logger.debug(f"    Shape after Concat: {current_hidden_states.shape}")

            # Apply ResNet
            h_before_resnet = current_hidden_states.shape
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                current_hidden_states = self._gradient_checkpointing_func(resnet, current_hidden_states, temb)
            else:
                current_hidden_states = resnet(current_hidden_states, temb)

            logger.debug(f"    Applied ResNet {i}: Input={h_before_resnet}, Output={current_hidden_states.shape}")
            logger.debug(f"  --- End ResNet Stage {i} ---")

        hidden_states = current_hidden_states

        # Apply upsampler(s)
        if hasattr(self, "upsamplers") and self.upsamplers is not None and len(self.upsamplers) > 0:
            logger.debug("\n  --- Applying Upsamplers ---")
            for i_up, upsampler in enumerate(self.upsamplers):
                h_before_upsample = hidden_states.shape
                hidden_states = upsampler(hidden_states, upsample_size)
                logger.debug(f"    Upsampler {i_up} ({type(upsampler).__name__}): Input={h_before_upsample}, Output={hidden_states.shape}")
        else:
            logger.debug("  --- No Upsamplers in this block ---")

        logger.debug("--- Exiting SUPIRUpBlock2D ---")
        return hidden_states


class SUPIRCrossAttnUpBlock2D(CrossAttnUpBlock2D):
    """
    A CrossAttnUpBlock2D modified for SUPIR to handle ZeroSFT injection
    (replacing skip concatenation) and potential extra ZeroCrossAttn injection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution_idx = kwargs.pop("resolution_idx", None)
        self.supir_active = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        is_freeu_enabled = getattr(self, "s1", None) and getattr(self, "s2", None) and getattr(self, "b1", None) and getattr(self, "b2", None)

        logger.debug(f"\n--- Entering SUPIRCrossAttnUpBlock2D (Index: {getattr(self, 'resolution_idx', 'N/A')}) ---")
        logger.debug(f"  Input hidden_states shape: {hidden_states.shape}")
        logger.debug(f"  Input res_hidden_states shapes: {[res.shape for res in res_hidden_states_tuple]}")
        if temb is not None:
            logger.debug(f"  Input temb shape: {temb.shape}")
        if encoder_hidden_states is not None:
            logger.debug(f"  Input encoder_hidden_states shape: {encoder_hidden_states.shape}")

        control_hs = None
        project_modules = None
        block_stages_map = []
        is_supir_mode = False
        standard_attn_kwargs = {}
        block_cross_attn_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        if block_cross_attn_kwargs:
            control_hs = block_cross_attn_kwargs.pop("control_hs", None)
            project_modules = block_cross_attn_kwargs.pop("project_modules", None)
            block_stages_map = block_cross_attn_kwargs.pop("injection_stages_map", {})
            block_cross_attn_kwargs.pop("control_scale_global_dynamic", None)
            block_cross_attn_kwargs.pop("use_linear_control_scale_pipeline", None)
            block_cross_attn_kwargs.pop("injection_scales", None)
            standard_attn_kwargs = block_cross_attn_kwargs

            if control_hs is not None and project_modules is not None and block_stages_map:
                logger.debug(f"  SUPIR Mode ACTIVE: Controls Provided={len(control_hs)}, ProjMods Available={len(project_modules)}")
                is_supir_mode = True
            else:
                logger.debug("  SUPIR Mode INACTIVE (control_hs, project_modules, or block_config missing)")
                is_supir_mode = False
        else:
            logger.debug("  SUPIR Mode INACTIVE (cross_attention_kwargs missing)")
            is_supir_mode = False

        current_hidden_states = hidden_states

        # ResNet/Attention Stages
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            stage_map = block_stages_map[i] if i < len(block_stages_map) else {}
            logger.debug(f"\n  --- ResNet/Attn Stage {i} ---")
            logger.debug(f"    Input hidden_states shape: {current_hidden_states.shape}")
            logger.debug(f"    Using Skip Connection (res_hidden_states) shape: {res_hidden_states.shape}")

            # Apply SFT or Concat before ResNet
            applied_sft_logic = False
            if is_supir_mode and stage_map:
                sft_proj_idx = stage_map.get("sft_proj")
                sft_ctrl_idx = stage_map.get("sft_ctrl")
                apply_sft_flag = stage_map.get("sft_active", True)
                sft_scale = stage_map.get("sft_scale", 1.0)

            logger.debug(
                f"    Checking UpBlock[{self.resolution_idx}]_stage_{i}: ProjIdx={sft_proj_idx}, CtrlIdx={sft_ctrl_idx}, Scale={sft_scale}, Active={apply_sft_flag}"
            )
            if (
                apply_sft_flag
                and sft_proj_idx is not None
                and sft_ctrl_idx is not None
                and control_hs is not None
                and project_modules is not None
                and sft_ctrl_idx < len(control_hs)
                and sft_proj_idx < len(project_modules)
            ):
                module_sft = project_modules[sft_proj_idx]
                control = control_hs[sft_ctrl_idx]
                if isinstance(module_sft, ZeroSFT) and control is not None:
                    logger.debug(f"    Applying SFT (Stage {i} - Proj[{sft_proj_idx}], Ctrl[{sft_ctrl_idx}]) Scale={sft_scale}")
                    current_hidden_states = module_sft(c=control, h_skip=res_hidden_states, h_decoder=current_hidden_states, control_scale=sft_scale)
                    applied_sft_logic = True
                else:
                    logger.warning(
                        f"SUPIRCrossAttnUpBlock[{self.resolution_idx}] - Initial SFT failed checks: ModType={type(module_sft)}, CtrlIsNone={control is None}"
                    )
            else:
                if not apply_sft_flag:
                    logger.debug(f"SUPIRCrossAttnUpBlock[{self.resolution_idx}] - SFT not applied.")
                else:
                    logger.warning(
                        f"SUPIRCrossAttnUpBlock[{self.resolution_idx}] - Initial SFT failed checks: ModType={type(module_sft)}, CtrlIsNone={control is None}"
                    )

            if not applied_sft_logic:
                logger.debug("    Applying Standard Concatenation (Initial)")
                current_hidden_states = torch.cat([current_hidden_states, res_hidden_states.to(current_hidden_states.dtype)], dim=1)
            logger.debug(f"    Output shape after Initial SFT/Concat: {current_hidden_states.shape}")

            # Apply ResNet
            h_before_resnet = current_hidden_states.shape
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                current_hidden_states = self._gradient_checkpointing_func(resnet, current_hidden_states, temb)
            else:
                current_hidden_states = resnet(current_hidden_states, temb)

            logger.debug(f"    Applied ResNet {i}: Input={h_before_resnet}, Output={current_hidden_states.shape}")

            # Apply default Attention
            h_before_attn = current_hidden_states.shape
            logger.debug(f"    Applying Standard Attention {i} (Input Shape: {h_before_attn})")
            attn_args = {
                "hidden_states": current_hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "cross_attention_kwargs": standard_attn_kwargs if standard_attn_kwargs else None,
                "attention_mask": attention_mask,
                "encoder_attention_mask": encoder_attention_mask,
                "return_dict": False,
            }
            valid_attn_args = {k: v for k, v in attn_args.items() if k in inspect.signature(attn.forward).parameters}
            current_hidden_states = attn(**valid_attn_args)[0]
            logger.debug(f"    Output hidden_states shape after Standard Attention: {current_hidden_states.shape}")

            # Apply Extra Cross Attention
            if is_supir_mode and stage_map:
                cross_proj_idx = stage_map.get("cross_proj")
                cross_ctrl_idx = stage_map.get("cross_ctrl")
                apply_cross_flag = stage_map.get("cross_active", False)
                cross_scale = stage_map.get("cross_scale", 1.0)

                logger.debug(
                    f"    Checking Cross UpBlock[{self.resolution_idx}]_stage_{i}: ProjIdx={cross_proj_idx}, CtrlIdx={cross_ctrl_idx}, Scale={sft_scale}, Active={apply_cross_flag}"
                )
                if (
                    apply_cross_flag
                    and cross_proj_idx is not None
                    and cross_ctrl_idx is not None
                    and control_hs is not None
                    and project_modules is not None
                    and cross_ctrl_idx < len(control_hs)
                    and cross_proj_idx < len(project_modules)
                ):
                    module_cross = project_modules[cross_proj_idx]
                    control = control_hs[cross_ctrl_idx]
                    if isinstance(module_cross, ZeroCrossAttn) and control is not None:
                        logger.debug(
                            f"    Applying EXTRA CrossAttn Injection (Stage {i} - ProjMod[{cross_proj_idx}], Ctrl[{cross_ctrl_idx}]) Scale={cross_scale}"
                        )
                        logger.debug(f"      Input x (hidden_states): {current_hidden_states.shape}")
                        logger.debug(f"      Input context (control): {control.shape}")
                        target_dtype = current_hidden_states.dtype

                        current_hidden_states = module_cross(
                            context=control.to(target_dtype), x=current_hidden_states.to(target_dtype), control_scale=cross_scale
                        )
                        logger.debug(f"    Output hidden_states shape after CrossAttn: {current_hidden_states.shape}")
                    else:
                        logger.warning(
                            f"SupirCrossAttnUpBlock[{self.resolution_idx}] - Stage {i}: CrossAttn Injection failed checks: ModType={type(module_cross)}, CtrlIsNone={control is None}"
                        )
                else:
                    if not apply_cross_flag:
                        logger.debug(f"SUPIRCrossAttnUpBlock[{self.resolution_idx}] - Cross not applied.")
                    else:
                        logger.warning(
                            f"SupirCrossAttnUpBlock[{self.resolution_idx}] - Stage {i}: CrossAttn Injection failed checks: ModType={type(module_cross)}, CtrlIsNone={control is None}"
                        )

                logger.debug(f"  --- End ResNet/Attn Stage {i} ---")

        hidden_states = current_hidden_states

        # Apply upsampler(s)
        if hasattr(self, "upsamplers") and self.upsamplers is not None and len(self.upsamplers) > 0:
            logger.debug("\n  --- Applying Upsamplers ---")
            for i_up, upsampler in enumerate(self.upsamplers):
                h_before_upsample = hidden_states.shape
                hidden_states = upsampler(hidden_states, upsample_size)
                logger.debug(f"    Upsampler {i_up} ({type(upsampler).__name__}): Input={h_before_upsample}, Output={hidden_states.shape}")
        else:
            logger.debug("  --- No Upsamplers in this block ---")

        logger.debug("--- Exiting SUPIRCrossAttnUpBlock2D ---")
        return hidden_states


class UNet2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin):
    """
    A UNet2DConditionModel modified to integrate SUPIR's ZeroSFT and ZeroCrossAttn
    modules for high-resolution image restoration tasks.

    This class loads a base UNet model (like SDXL or a fine-tuned variant) and
    additional weights for the SUPIR-specific projection modules. It modifies the
    forward pass to inject control signals from a separate ControlNet model
    (SUPIRControlNetModel) via the ZeroSFT and ZeroCrossAttn modules within the up-blocks.

    Key features:
    - `from_pretrained_supir`: Class method for unified loading of base UNet and SUPIR weights.
    - `_build_project_modules`: Instantiates the 12 SUPIR project modules based on checkpoint shapes.
    - `SUPIR_INJECTION_MAP`: Defines where and how control signals are injected.
    - Modified `forward` pass orchestrating the process.
    - Custom `SupirUpBlock2D` and `SupirCrossAttnUpBlock2D` handle internal injection logic.
    """

    _supports_gradient_checkpointing = True

    _no_split_modules = [
        "DownBlock2D",
        "ResnetBlock2D",
        "BasicTransformerBlock",
        "Transformer2DModel",
        "UNetMidBlock2DCrossAttn",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
        "SupirCrossAttnUpBlock2D",  # Custom up-block
        "SupirUpBlock2D",  # Custom up-block
        "Attention",
        "ZeroSFT",  # Custom SGM module
        "ZeroCrossAttn",  # Custom SGM module
    ]
    _skip_layerwise_casting_patterns = ["norm"]

    # Defines the injection points and corresponding control/project module indices.
    SUPIR_INJECTION_MAP = {
        "post_mid": {
            # SGM use proj[11] / ctrl[9]
            "sft_proj": 11,
            "sft_ctrl": 9,
            "sft_active": True,
            "sft_scale": 1.0,
            "cross_proj": None,
            "cross_ctrl": None,
            "cross_active": False,
            "cross_scale": 1.0,
        },
        "up_block_0": [  # Corresponds to SGM Out Blocks 0, 1, 2
            # Stage 0 (SGM Out Block 0)
            {"sft_proj": 10, "sft_ctrl": 8, "sft_active": True, "sft_scale": 1.0, "cross_proj": 7, "cross_ctrl": 6, "cross_active": False, "cross_scale": 1.0},
            # Stage 1 (SGM Out Block 1)
            {"sft_proj": 9, "sft_ctrl": 7, "sft_active": True, "sft_scale": 1.0, "cross_proj": 7, "cross_ctrl": 6, "cross_active": False, "cross_scale": 1.0},
            # Stage 2 (SGM Out Block 2)
            {
                "sft_proj": 8,
                "sft_ctrl": 6,
                "sft_active": True,
                "sft_scale": 1.0,
                "cross_proj": 7,
                "cross_ctrl": 6,
                "cross_active": True,
                "cross_scale": 1.0,
            },  # Default CrossAttn
        ],
        "up_block_1": [  # Corresponds to SGM Out Blocks 3, 4, 5
            # Stage 0 (SGM Out Block 3)
            {"sft_proj": 6, "sft_ctrl": 5, "sft_active": True, "sft_scale": 1.0, "cross_proj": 3, "cross_ctrl": 3, "cross_active": False, "cross_scale": 1.0},
            # Stage 1 (SGM Out Block 4)
            {"sft_proj": 5, "sft_ctrl": 4, "sft_active": True, "sft_scale": 1.0, "cross_proj": 3, "cross_ctrl": 3, "cross_active": False, "cross_scale": 1.0},
            # Stage 2 (SGM Out Block 5)
            {
                "sft_proj": 4,
                "sft_ctrl": 3,
                "sft_active": True,
                "sft_scale": 1.0,
                "cross_proj": 3,
                "cross_ctrl": 3,
                "cross_active": True,
                "cross_scale": 1.0,
            },  # Default CrossAttn
        ],
        "up_block_2": [  # Corresponds to SGM Out Blocks 6, 7, 8
            # Stage 0 (SGM Out Block 6)
            {
                "sft_proj": 2,
                "sft_ctrl": 2,
                "sft_active": True,
                "sft_scale": 1.0,
                "cross_proj": None,
                "cross_ctrl": None,
                "cross_active": False,
                "cross_scale": 1.0,
            },
            # Stage 1 (SGM Out Block 7)
            {
                "sft_proj": 1,
                "sft_ctrl": 1,
                "sft_active": True,
                "sft_scale": 1.0,
                "cross_proj": None,
                "cross_ctrl": None,
                "cross_active": False,
                "cross_scale": 1.0,
            },
            # Stage 2 (SGM Out Block 8)
            {
                "sft_proj": 0,
                "sft_ctrl": 0,
                "sft_active": True,
                "sft_scale": 1.0,
                "cross_proj": None,
                "cross_ctrl": None,
                "cross_active": False,
                "cross_scale": 1.0,
            },
        ],
    }

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (  # Default to SDXL base structure
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = (  # Default to SDXL base structure
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280),  # Default to SDXL base structure
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 2048,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = (1, 2, 10),  # Default to SDXL base structure
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = (5, 10, 20),  # Default to SDXL base structure
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = True,  # Default to SDXL base structure
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = "text_time",
        addition_time_embed_dim: Optional[int] = 256,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = 2816,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        supir_mode: Optional[str] = "XL-base",
        project_channel_scale: float = 2.0,  # Note: Scale is not used in current build logic
        project_type: Optional[str] = "ZeroSFT",
    ):
        """
        Initializes the UNet2DConditionModel model.

        Args:
            See `diffusers.models.unets.unet_2d_condition.UNet2DConditionModel` for standard UNet parameters.
            supir_mode (`Optional[str]`, *optional*, defaults to `"XL-base"`): SUPIR mode, determines project module architecture.
            project_channel_scale (`float`, *optional*, defaults to 2.0): Scaling factor (currently unused in module build).
            project_type (`Optional[str]`, *optional*, defaults to `"ZeroSFT"`): Base type for project modules (currently unused in build).
        """
        super().__init__()

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )

        # Input layer
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

        # Time embedding
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        # Encoder hidden states projection
        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )

        # Class embedding
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )

        # Addition embedding
        self._set_add_embedding(
            addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        # Time embed activation
        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # Process block-specific parameters
        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # ================== Down Blocks ==================
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # ================== Mid Block ==================
        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
        )

        # ================== Zero-SFT Project Modules ==================
        self.project_modules = None

        # ================== Up Blocks =================================
        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block)) if reverse_transformer_layers_per_block is None else reverse_transformer_layers_per_block
        )
        reversed_only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            add_upsample = not is_final_block
            if add_upsample:
                self.num_upsamplers += 1

            # Determine block class (SUPIR or Standard)
            up_block_class = None
            is_cross_attn = "CrossAttn" in up_block_type  # Check if it's a cross-attention block type
            if self.config.supir_mode is not None:
                up_block_class = SUPIRCrossAttnUpBlock2D if is_cross_attn else SUPIRUpBlock2D
            else:  # Standard mode
                if up_block_type == "UpBlock2D":
                    up_block_class = UpBlock2D
                elif up_block_type == "CrossAttnUpBlock2D":
                    up_block_class = CrossAttnUpBlock2D
                else:
                    raise ValueError(f"Unsupported up_block_type {up_block_type}")

            # Build common kwargs
            block_kwargs = {
                "num_layers": reversed_layers_per_block[i] + 1,
                "in_channels": input_channel,
                "out_channels": output_channel,
                "prev_output_channel": prev_output_channel,
                "temb_channels": blocks_time_embed_dim,
                "add_upsample": add_upsample,
                "resnet_eps": norm_eps,
                "resnet_act_fn": act_fn,
                "resolution_idx": i,
                "resnet_groups": norm_num_groups,
                "resnet_time_scale_shift": resnet_time_scale_shift,
                "dropout": dropout,
            }

            # Add cross-attention specific kwargs if needed
            if is_cross_attn:
                if cross_attention_dim is None:
                    raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")

                block_kwargs.update(
                    {
                        "transformer_layers_per_block": reversed_transformer_layers_per_block[i],
                        "cross_attention_dim": reversed_cross_attention_dim[i],
                        "num_attention_heads": reversed_num_attention_heads[i],
                        "dual_cross_attention": dual_cross_attention,
                        "use_linear_projection": use_linear_projection,
                        "only_cross_attention": reversed_only_cross_attention[i],
                        "upcast_attention": upcast_attention,
                        "attention_type": attention_type,
                    }
                )

            # Instantiate block
            up_block = up_block_class(**block_kwargs)
            self.up_blocks.append(up_block)

        # ================== Output Layer ==================
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
            self.conv_act = get_activation(act_fn)
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
        )

        # GLIGEN position net
        self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim)

    def _build_project_modules(self):
        """
        Builds the SGM-style project modules (ZeroSFT and ZeroCrossAttn)
        based on the specific shapes inferred from the functional SUPIR-Q checkpoint.
        This method ignores the project_channel_scale from the config and uses
        hardcoded dimensions derived from analyzing the checkpoint structure.
        """
        build_mode = getattr(self.config, "supir_mode", None)

        if build_mode != "XL-base":
            logger.info(f"Project module build skipped: Unsupported mode '{build_mode}'. Only 'XL-base' is supported.")
            self.project_modules = None
            return

        if hasattr(self, "project_modules") and self.project_modules is not None and len(self.project_modules) == 12:
            logger.info("Project modules seem already built. Skipping.")
            return

        logger.info("Building SGM project_modules for XL-base (Matching Checkpoint Shapes)")

        modules = []
        # Final Order: [S0, S1, S2, C3, S4, S5, S6, C7, S8, S9, S10, S11]
        # Dimensions derived from analyzing the working checkpoint shapes

        try:
            # Module 0 (SFT SGM 0) - label=320, norm=320, concat=320
            modules.append(ZeroSFT(label_nc=320, norm_nc=320, concat_channels=320))
            # Module 1 (SFT SGM 1) - label=320, norm=320, concat=320
            modules.append(ZeroSFT(label_nc=320, norm_nc=320, concat_channels=320))
            # Module 2 (SFT SGM 2) - label=320, norm=320, concat=640
            modules.append(ZeroSFT(label_nc=320, norm_nc=320, concat_channels=640))
            # Module 3 (CrossAttn C_idx3) - context=320, query=640
            modules.append(ZeroCrossAttn(context_dim=320, query_dim=640))
            # Module 4 (SFT SGM 3) - label=320, norm=320, concat=640
            modules.append(ZeroSFT(label_nc=320, norm_nc=320, concat_channels=640))
            # Module 5 (SFT SGM 4) - label=640, norm=640, concat=640
            modules.append(ZeroSFT(label_nc=640, norm_nc=640, concat_channels=640))
            # Module 6 (SFT SGM 5) - label=640, norm=640, concat=1280
            modules.append(ZeroSFT(label_nc=640, norm_nc=640, concat_channels=1280))
            # Module 7 (CrossAttn C_idx6) - context=640, query=1280
            modules.append(ZeroCrossAttn(context_dim=640, query_dim=1280))
            # Module 8 (SFT SGM 6) - label=640, norm=640, concat=1280
            modules.append(ZeroSFT(label_nc=640, norm_nc=640, concat_channels=1280))
            # Module 9 (SFT SGM 7) - label=1280, norm=1280, concat=1280
            modules.append(ZeroSFT(label_nc=1280, norm_nc=1280, concat_channels=1280))
            # Module 10 (SFT SGM 8) - label=1280, norm=1280, concat=1280
            modules.append(ZeroSFT(label_nc=1280, norm_nc=1280, concat_channels=1280))
            # Module 11 (SFT SGM 9) - label=1280, norm=1280, concat=0
            modules.append(ZeroSFT(label_nc=1280, norm_nc=1280, concat_channels=0))

        except Exception as e:
            logger.error(f"Error during project module instantiation: {e}")
            raise e

        self.project_modules = nn.ModuleList(modules)
        logger.info(f"Project modules built. Total: {len(self.project_modules)}")
        self._built_mode = build_mode
        self._built_project_type_base = "SFT"

    def _load_project_modules_state_dict(self, state_dict, strict=True):
        """Loads the state dict for the project_modules, handling prefixes."""

        project_modules_state_dict = {}
        direct_prefix = "project_modules."
        sgm_prefix = "model.diffusion_model.project_modules."

        found_direct = any(k.startswith(direct_prefix) for k in state_dict.keys())
        target_prefix = direct_prefix if found_direct else sgm_prefix
        prefix_len = len(target_prefix)
        keys_found_count = 0

        for key, value in state_dict.items():
            if key.startswith(target_prefix):
                new_key = key[prefix_len:]
                project_modules_state_dict[new_key] = value
                keys_found_count += 1

        if keys_found_count == 0:
            fallback_prefix = sgm_prefix if found_direct else direct_prefix
            fallback_len = len(fallback_prefix)
            for key, value in state_dict.items():
                if key.startswith(fallback_prefix):
                    new_key = key[fallback_len:]
                    project_modules_state_dict[new_key] = value
                    keys_found_count += 1
            if keys_found_count == 0:
                logger.error(f"No keys matching prefixes '{direct_prefix}' or '{sgm_prefix}' found. Cannot load project modules.")
                return False
            else:
                logger.warning(f"Used fallback prefix '{fallback_prefix}' to find {keys_found_count} keys.")

        try:
            load_info = self.project_modules.load_state_dict(project_modules_state_dict, strict=strict)
            if strict:
                logger.info("Loaded project module state dict successfully (strict=True).")
            else:
                logger.warning(
                    f"Loaded project module state dict (strict=False). Missing keys: {load_info.missing_keys}. Unexpected keys: {load_info.unexpected_keys}"
                )
                if load_info.missing_keys:
                    logger.warning(f"Missing project module keys: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    logger.warning(f"Unexpected project module keys found: {load_info.unexpected_keys}")  # Warning instead of error
            return True
        except Exception as e:
            logger.error(f"ERROR loading project module state dict: {e}")
            return False

    @classmethod
    def _load_project_module_weights(
        cls,
        model: torch.nn.Module,
        supir_model_path: str,
        supir_weight_name: str,
        supir_zero_sft_subfolder: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: Optional[str] = None,
        **kwargs,
    ):
        """
        Finds, loads, and sets SUPIR project module weights into an existing model.
        This method directly integrates the weight loading logic from the original
        from_pretrained_supir's "Seção 5".
        """
        logger.debug(f"Loading additional SUPIR weights (project_modules) from: {supir_model_path}")

        supir_weights_path = None
        supir_weight_filename_rel = os.path.join(supir_zero_sft_subfolder, supir_weight_name) if supir_zero_sft_subfolder else supir_weight_name
        hub_download_args = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "repo_type": "model",
        }

        try:
            if os.path.isdir(supir_model_path):
                path = os.path.join(supir_model_path, supir_weight_filename_rel)
                if os.path.exists(path):
                    supir_weights_path = path
                else:
                    alt_filename_rel = supir_weight_filename_rel.replace(".safetensors", ".bin")
                    alt_path = os.path.join(supir_model_path, alt_filename_rel)
                    if os.path.exists(alt_path):
                        supir_weights_path = alt_path
                    else:
                        logger.error(f"SUPIR weights not found locally at '{path}' or '{alt_path}'.")
            elif not local_files_only:
                try:
                    supir_weights_path = hf_hub_download(
                        repo_id=supir_model_path,
                        filename=supir_weight_filename_rel,
                        **hub_download_args,
                    )
                except EntryNotFoundError:
                    alt_filename_rel = supir_weight_filename_rel.replace(".safetensors", ".bin")
                    logger.warning(f"'{supir_weight_filename_rel}' not found on Hub '{supir_model_path}', trying '{alt_filename_rel}'...")
                    try:
                        supir_weights_path = hf_hub_download(
                            repo_id=supir_model_path,
                            filename=alt_filename_rel,
                            **hub_download_args,
                        )
                    except EntryNotFoundError:
                        logger.error(f"Neither '{supir_weight_filename_rel}' nor '{alt_filename_rel}' found in Hub repo {supir_model_path}.")
                except Exception as e_download:
                    logger.error(f"Error downloading SUPIR weights ('{supir_weight_filename_rel}' or fallback): {e_download}")
            else:
                logger.warning(f"SUPIR weights not found locally at '{supir_model_path}' (not a dir) and local_files_only=True.")

            if not supir_weights_path or not os.path.exists(supir_weights_path):
                raise FileNotFoundError(
                    f"Could not find SUPIR project module weights file. Searched in '{supir_model_path}' "
                    f"for '{supir_weight_filename_rel}' (and .bin fallback if applicable)."
                )

            # Load SUPIR state dict
            logger.info(f"Loading SUPIR state dict from {supir_weights_path}")
            if supir_weights_path.endswith(".safetensors"):
                supir_state_dict = load_file(supir_weights_path, device="cpu")
            else:
                supir_state_dict = torch.load(supir_weights_path, map_location="cpu")
            logger.debug(f"Loaded SUPIR state dict from {supir_weights_path} (keys: {len(supir_state_dict)}).")

            # Build and load project modules
            if not hasattr(model, "project_modules") or model.project_modules is None:
                logger.info("Project modules not found on the model. Attempting to build them...")
                try:
                    model._build_project_modules()
                    if not hasattr(model, "project_modules") or model.project_modules is None:
                        raise RuntimeError("Model config lacks 'supir_mode' or build failed.")
                    model.project_modules.to(device=model.device, dtype=model.dtype)
                    logger.info("Project modules built successfully during weight loading.")
                except Exception as e_build:
                    logger.error(f"Failed to build project_modules: {e_build}")
                    raise RuntimeError(f"Could not build project_modules on the model: {e_build}") from e_build

            load_success = model._load_project_modules_state_dict(supir_state_dict, strict=True)
            if not load_success:
                raise RuntimeError(f"Failed to load SUPIR weights into project_modules from {supir_weights_path}.")
            del supir_state_dict
            logger.info("SUPIR project_modules weights loaded successfully into the model.")
            return True

        except Exception as e:
            logger.error(f"An error occurred during SUPIR project_module weight loading: {e}")
            raise e

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        supir_model_path: str,
        subfolder: str = "unet",
        supir_config_name: str = "config.json",
        supir_weight_name: str = "diffusion_pytorch_model.safetensors",
        **kwargs,
    ) -> "UNet2DConditionModel":
        """
        Loads a base UNet model and SUPIR project module weights, supporting advanced loading options via kwargs.

        This method follows the modern `diffusers` loading pipeline. It first instantiates
        the complete SUPIR-enabled UNet structure on the "meta" device. It then calls the
        internal `_load_pretrained_model` method to load only the base UNet weights, which
        correctly handles `device_map` and offloading while ignoring the custom SUPIR modules.

        Finally, it loads the custom SUPIR weights into the remaining modules and finalizes
        the model.

        Args:
            pretrained_model_name_or_path (`str`): Path or Hub ID of the base UNet model.
            supir_model_path (`str`): Path or Hub ID of the SUPIR model components.
            subfolder (`str`, *optional*, defaults to `"unet"`): Subfolder for base UNet weights/config.
            supir_config_name (`str`, *optional*, defaults to `"config.json"`): Filename of the SUPIR config.
            supir_weight_name (`str`, *optional*, defaults to `"diffusion_pytorch_model.safetensors"`):
                Filename of the SUPIR project module weights.
            **kwargs:
                Additional arguments passed to the diffusers loading backend. Key arguments include:
                `device_map`, `max_memory`, `low_cpu_mem_usage`, `torch_dtype`, `variant`, `use_safetensors`,
                `cache_dir`, `force_download`, `token`, `revision`, etc.

        Returns:
            `UNet2DConditionModel`: The loaded and initialized UNet2DConditionModel model.
        """
        # 1. Pop arguments from kwargs
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        quantization_config = kwargs.pop("quantization_config", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        disable_mmap = kwargs.pop("disable_mmap", False)

        # 2. Initial Validations
        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError("device_map requires `accelerate` library. Please install it with `pip install accelerate`.")
        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning("`low_cpu_mem_usage` requires `accelerate`. Disabling it.")
        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError("Cannot set `low_cpu_mem_usage=False` while using `device_map`.")

        logger.debug(f"Starting unified SUPIR loading from base='{pretrained_model_name_or_path}' and supir='{supir_model_path}'")

        # 3. Determine Final SUPIR Configuration
        final_supir_mode = "XL-base"
        final_supir_scale = 2.0
        final_supir_project_type = "ZeroSFT"
        supir_zero_sft_subfolder: str = "zero_sft"
        supir_config_path = None
        supir_token_used = token
        supir_config_filename_rel = os.path.join(supir_zero_sft_subfolder, supir_config_name) if supir_zero_sft_subfolder else supir_config_name

        if os.path.isdir(supir_model_path):
            path = os.path.join(supir_model_path, supir_config_filename_rel)
            if os.path.exists(path):
                supir_config_path = path
        elif not local_files_only:
            try:
                supir_config_path = hf_hub_download(
                    repo_id=supir_model_path,
                    filename=supir_config_filename_rel,
                    repo_type="model",
                    cache_dir=cache_dir,
                    token=supir_token_used,
                    force_download=force_download,
                    revision=revision,
                )
            except EntryNotFoundError:
                logger.warning(f"SUPIR config '{supir_config_filename_rel}' not on Hub '{supir_model_path}'.")
            except Exception as e:
                logger.error(f"Error downloading SUPIR config: {e}")

        if supir_config_path and os.path.exists(supir_config_path):
            try:
                with open(supir_config_path, "r") as f:
                    supir_config_data = json.load(f)
                logger.debug(f"Loaded SUPIR config data: {supir_config_data}")
                final_supir_mode = supir_config_data.get("supir_mode", final_supir_mode)
                final_supir_scale = supir_config_data.get("project_channel_scale", final_supir_scale)
                final_supir_project_type = supir_config_data.get("project_type", final_supir_project_type)
            except Exception as e:
                logger.error(f"Failed to load/parse SUPIR config {supir_config_path}: {e}")
        else:
            logger.warning("SUPIR config not found/loaded. Using defaults.")

        if final_supir_mode is None:
            raise ValueError("Could not determine `supir_mode`.")
        logger.debug(f"Final SUPIR params: mode={final_supir_mode}, scale={final_supir_scale}, type={final_supir_project_type}")

        # 4. Load Base Config and Add SUPIR Params
        logger.debug(f"Loading base UNet config from: {pretrained_model_name_or_path}/{subfolder}")
        config, _, commit_hash = cls.load_config(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            **kwargs,
        )
        config = copy.deepcopy(config)
        logger.debug("Base config loaded.")

        user_agent = {"diffusers": __version__, "file_type": "model", "framework": "pytorch"}

        # determine initial quantization config.
        #######################################
        pre_quantized = "quantization_config" in config and config["quantization_config"] is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config["quantization_config"] = DiffusersAutoQuantizer.merge_quantization_configs(config["quantization_config"], quantization_config)
            else:
                config["quantization_config"] = quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(config["quantization_config"], pre_quantized=pre_quantized)
        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(torch_dtype=torch_dtype, from_flax=from_flax, device_map=device_map)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            device_map = hf_quantizer.update_device_map(device_map)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = cls._keep_in_fp32_modules is not None and (hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False))

        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        else:
            keep_in_fp32_modules = []

        config["supir_mode"] = final_supir_mode
        config["project_channel_scale"] = final_supir_scale
        config["project_type"] = final_supir_project_type
        logger.debug("Final config prepared using base model's structure.")

        # 5. Instantiate Model on Meta Device
        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())

        with ContextManagers(init_contexts):
            model = cls.from_config(config, **kwargs)

        logger.debug("UNet2DConditionModel instance created on meta device.")

        # 6. Load BASE Weights using the official _load_pretrained_model
        logger.debug("Loading base model weights using diffusers' internal loader...")

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        is_local = os.path.isdir(pretrained_model_name_or_path)

        hub_kwargs_fetch = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "user_agent": user_agent,
            "commit_hash": commit_hash,
        }

        # Determine if we're loading from a sharded checkpoint
        is_sharded = False
        sharded_metadata = None
        resolved_model_file = None

        index_file = _fetch_index_file(
            is_local=is_local,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder or "",
            use_safetensors=use_safetensors,
            variant=variant,
            **hub_kwargs_fetch,
        )
        if index_file is not None and index_file.is_file():
            is_sharded = True

        # Load the weights
        if is_sharded:
            resolved_model_file, sharded_metadata = _get_checkpoint_shard_files(
                pretrained_model_name_or_path, index_file, **hub_kwargs_fetch, subfolder=subfolder or ""
            )
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
            logger.debug(f"Found sharded base weights: {resolved_model_file}")
        else:
            weights_name_sf = _add_variant(SAFETENSORS_WEIGHTS_NAME, variant)
            weights_name_bin = _add_variant(WEIGHTS_NAME, variant)
            try:
                if use_safetensors:
                    resolved_model_file = [
                        _get_model_file(pretrained_model_name_or_path, weights_name=weights_name_sf, **hub_kwargs_fetch, subfolder=subfolder)
                    ]
                else:
                    raise IOError("use_safetensors is False, skipping safetensors check.")
            except IOError:
                if allow_pickle:
                    resolved_model_file = [
                        _get_model_file(pretrained_model_name_or_path, weights_name=weights_name_bin, **hub_kwargs_fetch, subfolder=subfolder)
                    ]
                else:
                    raise IOError(f"Could not find .safetensors weights for variant {variant} and pickle loading is disabled.")
            loaded_keys = None

        state_dict = None
        if not is_sharded:
            state_dict = load_state_dict(resolved_model_file[0], disable_mmap=disable_mmap)  # disable_mmap can be a kwarg later
            if loaded_keys is None:
                loaded_keys = list(state_dict.keys())

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules)

        # 7. Determine Device Map
        final_device_map = _determine_device_map(
            model, device_map=device_map, max_memory=max_memory, torch_dtype=torch_dtype, keep_in_fp32_modules=[], hf_quantizer=hf_quantizer
        )
        # The official method will handle loading, device mapping, and offloading for the base UNet parts.
        # It will correctly ignore the `project_modules` as they are not in the `loaded_keys`.
        model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs = cls._load_pretrained_model(
            model,
            state_dict=state_dict,
            resolved_model_file=resolved_model_file,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            loaded_keys=loaded_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=final_device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=[],
        )
        logger.info(f"Base UNet weights loaded via internal method. Mismatched keys: {mismatched_keys}")

        # 8. Load SUPIR Project Module Weights
        if supir_model_path:
            logger.debug("Loading SUPIR project_modules weights...")
            cls._load_project_module_weights(
                model=model,
                supir_model_path=supir_model_path,
                supir_weight_name=supir_weight_name,
                supir_zero_sft_subfolder=supir_zero_sft_subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        else:
            logger.warning("`supir_model_path` not provided. Skipping loading of SUPIR project_modules weights.")

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        # 9. Finalization
        # A final .to(dtype) ensures type consistency.
        if torch_dtype is not None and torch_dtype == getattr(torch, "float8_e4m3fn", None) and hf_quantizer is None and not use_keep_in_fp32_modules:
            model.to(torch_dtype)

        if hf_quantizer is not None:
            # We also make sure to purge `_pre_quantization_dtype` when we serialize
            # the model config because `_pre_quantization_dtype` is `torch.dtype`, not JSON serializable.
            model.register_to_config(_name_or_path=pretrained_model_name_or_path, _pre_quantization_dtype=torch_dtype)
        else:
            model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        model.eval()
        logger.info("UNet2DConditionModel loaded successfully!")

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path_or_dict: str,
        supir_model_path: Optional[str] = None,
        supir_config_name: str = "config.json",
        supir_weight_name: str = "diffusion_pytorch_model.safetensors",
        **kwargs,
    ) -> "UNet2DConditionModel":
        supir_cache_dir = kwargs.get("cache_dir", None)
        supir_token = kwargs.get("token", None)
        supir_local_files_only = kwargs.get("local_files_only", False)
        supir_force_download = kwargs.get("force_download", False)
        supir_revision = kwargs.get("revision", None)

        supir_kwargs_to_pop = ["supir_model_path", "supir_config_name", "supir_weight_name"]

        clean_kwargs = kwargs.copy()
        for skw in supir_kwargs_to_pop:
            clean_kwargs.pop(skw, None)

        supir_zero_sft_subfolder: str = "zero_sft"
        logger.debug(f"Loading base UNet from single file: {pretrained_model_link_or_path_or_dict}")

        model = super(UNet2DConditionModel, cls).from_single_file(pretrained_model_link_or_path_or_dict, **clean_kwargs)

        current_supir_mode = model.config.get("supir_mode", "XL-base")
        current_project_channel_scale = model.config.get("project_channel_scale", 2.0)
        current_project_type = model.config.get("project_type", "ZeroSFT")

        if supir_model_path:
            supir_config_filename_rel = os.path.join(supir_zero_sft_subfolder, supir_config_name) if supir_zero_sft_subfolder else supir_config_name
            if os.path.isdir(supir_model_path):
                path = os.path.join(supir_model_path, supir_config_filename_rel)
                if os.path.exists(path):
                    supir_config_path = path
            elif not supir_local_files_only:
                try:
                    supir_config_path = hf_hub_download(
                        repo_id=supir_model_path,
                        filename=supir_config_filename_rel,
                        repo_type="model",
                        cache_dir=supir_cache_dir,
                        token=supir_token,
                        force_download=supir_force_download,
                        revision=supir_revision,
                    )
                except EntryNotFoundError:
                    logger.warning(f"SUPIR config '{supir_config_filename_rel}' not on Hub '{supir_model_path}'.")
                except Exception as e:
                    logger.error(f"Error downloading SUPIR config: {e}")
            else:
                logger.warning("SUPIR config not found locally and local_files_only=True.")

            if supir_config_path and os.path.exists(supir_config_path):
                try:
                    with open(supir_config_path, "r") as f:
                        supir_config_data = json.load(f)
                    logger.debug(f"Loaded SUPIR-specific config data: {supir_config_data}")
                    current_supir_mode = supir_config_data.get("supir_mode", current_supir_mode)
                    current_project_channel_scale = supir_config_data.get("project_channel_scale", current_project_channel_scale)
                    current_project_type = supir_config_data.get("project_type", current_project_type)
                except Exception as e:
                    logger.error(f"Failed to load/parse SUPIR-specific config {supir_config_path}: {e}")
            else:
                logger.warning("SUPIR-specific config not found. Using existing/default SUPIR parameters.")

        model.config.supir_mode = current_supir_mode
        model.config.project_channel_scale = current_project_channel_scale
        model.config.project_type = current_project_type

        if supir_model_path is not None:
            logger.debug("Loading SUPIR project_modules weights...")
            cls._load_project_module_weights(
                model=model,
                supir_model_path=supir_model_path,
                supir_config_name=supir_config_name,
                supir_weight_name=supir_weight_name,
                supir_zero_sft_subfolder=supir_zero_sft_subfolder,
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                proxies=kwargs.get("proxies"),
                local_files_only=kwargs.get("local_files_only", False),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
            )
        else:
            logger.warning("`supir_model_path` not provided. Skipping loading of SUPIR project_modules weights.")

        model.eval()
        logger.info("UNet2DConditionModel loaded successfully!")
        return model

    def _check_config(
        self,
        down_block_types: Tuple[str],
        up_block_types: Tuple[str],
        only_cross_attention: Union[bool, Tuple[bool]],
        block_out_channels: Tuple[int],
        layers_per_block: Union[int, Tuple[int]],
        cross_attention_dim: Union[int, Tuple[int]],
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
        reverse_transformer_layers_per_block: bool,
        attention_head_dim: int,
        num_attention_heads: Optional[Union[int, Tuple[int]]],
    ):
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

    def _set_time_proj(self, time_embedding_type, block_out_channels, flip_sin_to_cos, freq_shift, time_embedding_dim):
        """Initializes the time projection layer."""
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError("time_embed_dim must be even for Fourier.")
            self.time_proj = GaussianFourierProjection(time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos)
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(f"Unsupported time_embedding_type: {time_embedding_type}")
        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(self, encoder_hid_dim_type, cross_attention_dim, encoder_hid_dim):
        """Initializes the encoder hidden states projection layer."""
        if isinstance(cross_attention_dim, tuple):
            cross_attn_dim_for_proj = cross_attention_dim[-1]
        else:
            cross_attn_dim_for_proj = cross_attention_dim

        self.encoder_hid_proj = None
        if encoder_hid_dim_type is None:
            return

        if encoder_hid_dim is None:
            raise ValueError(f"`encoder_hid_dim` required for type {encoder_hid_dim_type}.")

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attn_dim_for_proj)
        elif encoder_hid_dim_type == "text_image_proj":
            self.encoder_hid_proj = TextImageProjection(encoder_hid_dim, encoder_hid_dim, cross_attn_dim_for_proj)
        elif encoder_hid_dim_type == "image_proj":
            self.encoder_hid_proj = ImageProjection(encoder_hid_dim, cross_attn_dim_for_proj)
        elif encoder_hid_dim_type == "ip_image_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attn_dim_for_proj)
        else:
            raise ValueError(f"Unsupported encoder_hid_dim_type: {encoder_hid_dim_type}")

    def _set_class_embedding(self, class_embed_type, act_fn, num_class_embeds, projection_class_embeddings_input_dim, time_embed_dim, timestep_input_dim):
        """Initializes the class embedding layer."""
        self.class_embedding = None
        if class_embed_type is None:
            pass
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity()
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("'projection' needs 'projection_class_embeddings_input_dim'")
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("'simple_projection' needs 'projection_class_embeddings_input_dim'")
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            raise ValueError(f"Unsupported class_embed_type: {class_embed_type}")

    def _set_add_embedding(
        self,
        addition_embed_type,
        addition_embed_type_num_heads,
        addition_time_embed_dim,
        flip_sin_to_cos,
        freq_shift,
        cross_attention_dim,
        encoder_hid_dim,
        projection_class_embeddings_input_dim,
        time_embed_dim,
    ):
        """Initializes the additional conditioning embedding layer (e.g., time, text)."""
        if isinstance(cross_attention_dim, tuple):
            cross_attn_dim_for_add = cross_attention_dim[-1]
        else:
            cross_attn_dim_for_add = cross_attention_dim

        self.add_embedding = None
        self.add_time_proj = None

        if addition_embed_type is None:
            pass
        elif addition_embed_type == "text":
            if encoder_hid_dim is None:
                raise ValueError("'text' requires 'encoder_hid_dim'")
            self.add_embedding = TextTimeEmbedding(encoder_hid_dim, time_embed_dim, num_heads=addition_embed_type_num_heads)
        elif addition_embed_type == "text_image":
            self.add_embedding = TextImageTimeEmbedding(cross_attn_dim_for_add, cross_attn_dim_for_add, time_embed_dim)
        elif addition_embed_type == "text_time":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("'text_time' requires 'projection_class_embeddings_input_dim'")
            if addition_time_embed_dim is None:
                raise ValueError("'text_time' requires 'addition_time_embed_dim'")
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            if encoder_hid_dim is None:
                raise ValueError("'image' requires 'encoder_hid_dim'")
            self.add_embedding = ImageTimeEmbedding(encoder_hid_dim, time_embed_dim)
        elif addition_embed_type == "image_hint":
            if encoder_hid_dim is None:
                raise ValueError("'image_hint' requires 'encoder_hid_dim'")
            self.add_embedding = ImageHintTimeEmbedding(encoder_hid_dim, time_embed_dim)
        else:
            raise ValueError(f"Unsupported addition_embed_type: {addition_embed_type}")

    def _set_pos_net_if_use_gligen(self, attention_type: str, cross_attention_dim: int):
        """Initializes the GLIGEN position network if required by attention type."""
        self.position_net = None
        if attention_type in ["gated", "gated-text-image"]:
            positive_len = getattr(self.config, "gligen_positive_len", 768)
            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = GLIGENTextBoundingboxProjection(positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """Returns the attention processor dictionary."""
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor") and callable(module.get_processor):
                try:
                    processors[f"{name}.processor"] = module.get_processor()
                except AttributeError as e:
                    logger.warning(f"Could not get processor for {name}: {e}")
                except TypeError as e:
                    logger.warning(f"TypeError calling get_processor for {name}: {e}. Trying without args.")
                    try:
                        processors[f"{name}.processor"] = module.get_processor()
                    except Exception as e2:
                        logger.error(f"Failed to get processor for {name} even without args: {e2}")

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            if name == "project_modules":
                for i, proj_module in enumerate(module):
                    fn_recursive_add_processors(f"{name}.{i}", proj_module, processors)
            else:
                fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        """Sets the attention processor(s)."""

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor") and callable(module.set_processor):
                final_processor = processor
                proc_key = f"{name}.processor"
                if isinstance(processor, dict):
                    final_processor = processor.get(proc_key, processor)
                    if proc_key not in processor:
                        logger.debug(f"Processor key {proc_key} not found in dict. Applying default/global processor.")
                        is_specific_processor_found = False
                        for key_p in processor.keys():
                            if name in key_p:
                                final_processor = processor[key_p]
                                is_specific_processor_found = True
                                break
                        if not is_specific_processor_found:
                            logger.warning(f"Specific processor for {name} not found in dict. Skipping set_processor for this module.")
                            return

                if not isinstance(final_processor, dict):
                    try:
                        module.set_processor(final_processor)
                    except Exception as e:
                        logger.error(f"Failed to set processor for {name}: {e}")
                else:
                    logger.error(f"Cannot set processor for {name}: final_processor is still a dict.")

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            if name == "project_modules":
                for i, proj_module in enumerate(module):
                    fn_recursive_attn_processor(f"project_modules.{i}", proj_module, processor.copy() if isinstance(processor, dict) else processor)
            else:
                fn_recursive_attn_processor(name, module, processor.copy() if isinstance(processor, dict) else processor)

    def set_default_attn_processor(self):
        """Sets the default AttnProcessor."""
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}")

        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size):
        """Enables sliced attention computation."""
        pass

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self):
        """Enables gradient checkpointing."""
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self):
        """Disables gradient checkpointing."""
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        """Enables the FreeU mechanism."""
        for i, up_block in enumerate(self.up_blocks):
            if hasattr(up_block, "set_freeu"):
                up_block.set_freeu(s1, s2, b1, b2)
            else:
                logger.debug(f"UpBlock {i} does not support FreeU.")

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        for up_block in self.up_blocks:
            if hasattr(up_block, "disable_freeu"):
                up_block.disable_freeu()

    def fuse_qkv_projections(self, fuse=True):
        """Fuses QKV projections (if possible)."""
        for module in self.modules():
            if isinstance(module, DiffusersAttention):
                module.fuse_projections(fuse)

    def unfuse_qkv_projections(self):
        """Unfuses QKV projections."""
        self.fuse_qkv_projections(fuse=False)

    def _get_time_embed(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]) -> Optional[torch.Tensor]:
        """Gets the time embedding for the forward pass."""
        if not hasattr(self, "time_proj"):
            return None
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            dtype = torch.int64 if isinstance(timestep, int) else (torch.float32 if (is_mps or is_npu) else torch.float64)
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        return t_emb.to(dtype=sample.dtype)

    def _get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Gets the class embedding for the forward pass."""
        if not hasattr(self, "class_embedding") or self.class_embedding is None:
            return None
        if class_labels is None:
            raise ValueError("class_labels required for class_embedding")
        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    def _get_aug_embed(self, emb, added_cond_kwargs):
        """
        Gets the additional conditioning embeddings for the forward pass.
        Reflects SGM logic: expects 'vector' in added_cond_kwargs for 'text_time' type
        and passes it directly to the 'add_embedding' layer.
        Handles the case where added_cond_kwargs is None or 'vector' is missing for testing.
        """

        if not hasattr(self, "add_embedding") or self.add_embedding is None:
            logger.debug("_get_aug_embed_forward: No 'add_embedding' layer found. Skipping.")
            return None

        if added_cond_kwargs is None:
            logger.debug("_get_aug_embed_forward: added_cond_kwargs is None. Skipping.")
            return None

        add_embed_type = getattr(self.config, "addition_embed_type", None)
        aug_emb = None

        vector_cond = added_cond_kwargs.get("vector")

        if vector_cond is not None:
            logger.debug(f"_get_aug_embed_forward: Found 'vector' in added_cond_kwargs with shape: {vector_cond.shape}")

            if hasattr(self.add_embedding, "linear_1"):
                expected_input_dim = self.add_embedding.linear_1.in_features
                if vector_cond.shape[-1] != expected_input_dim:
                    logger.error(f"UNet ADM error: 'vector' shape[-1] is {vector_cond.shape[-1]}, but add_embedding expects {expected_input_dim}.")
                    return None
                else:
                    logger.debug(f"  Passing vector (shape: {vector_cond.shape}) to self.add_embedding (expects: {expected_input_dim})")
                    aug_emb = self.add_embedding(vector_cond)
                    logger.debug(f"  Output of add_embedding shape: {aug_emb.shape}")
            else:
                logger.warning("Could not verify input dimension for add_embedding. Attempting to process vector.")
                try:
                    aug_emb = self.add_embedding(vector_cond)
                    logger.debug(f"  Output of add_embedding shape (dimension not verified): {aug_emb.shape}")
                except Exception as e:
                    logger.error(f"Error processing vector with add_embedding (dim not verified): {e}")
                    return None
        else:
            logger.debug("_get_aug_embed_forward: 'vector' key not found in added_cond_kwargs. Skipping ADM processing.")

        if aug_emb is None:
            if add_embed_type == "text":
                logger.debug("_get_aug_embed_forward: add_embed_type is 'text', but 'vector' was prioritized/missing. Skipping 'text' logic.")
                text_embeds = added_cond_kwargs.get("text_embeds")
                if text_embeds is not None:
                    aug_emb = self.add_embedding(text_embeds.to(emb.dtype))
                else:
                    logger.warning("'text_embeds' missing for 'text' type.")
            elif add_embed_type is not None and add_embed_type != "text_time":
                logger.warning(f"Unsupported or unhandled addition_embed_type: {add_embed_type} when 'vector' is missing.")

        if aug_emb is None:
            logger.debug("_get_aug_embed_forward: No augmentation embedding generated.")
        else:
            logger.debug(f"_get_aug_embed_forward: Returning augmentation embedding with shape: {aug_emb.shape}")

        return aug_emb

    def _process_encoder_hidden_states(self, encoder_hidden_states, added_cond_kwargs):
        """Processes encoder hidden states (e.g., projection) for the forward pass."""
        if not hasattr(self, "encoder_hid_proj") or self.encoder_hid_proj is None:
            return encoder_hidden_states
        enc_type = self.config.encoder_hid_dim_type
        if added_cond_kwargs is None:
            added_cond_kwargs = {}

        if enc_type == "text_proj":
            return self.encoder_hid_proj(encoder_hidden_states)
        elif enc_type == "text_image_proj":
            img_e = added_cond_kwargs.get("image_embeds")
            if img_e is None:
                raise ValueError("'text_image_proj' needs 'image_embeds'")
            return self.encoder_hid_proj(encoder_hidden_states, img_e)
        elif enc_type == "image_proj":
            img_e = added_cond_kwargs.get("image_embeds")
            if img_e is None:
                raise ValueError("'image_proj' needs 'image_embeds'")
            return self.encoder_hid_proj(img_e)
        elif enc_type == "ip_image_proj":
            img_e = added_cond_kwargs.get("image_embeds")
            if img_e is None:
                raise ValueError("'ip_image_proj' needs 'image_embeds'")
            img_e_proj = self.encoder_hid_proj(img_e)
            txt_e = encoder_hidden_states  # Assume text embeds might not need projection or is handled elsewhere
            return (txt_e, img_e_proj)  # Return as tuple
        else:
            return encoder_hidden_states  # Return original if type not handled

    def _get_dynamic_injection_map(
        self,
        base_map,
        current_timestep_module_scales: Optional[Dict[str, float]],
        flag_overrides: Optional[Dict[str, Any]],
        apply_global_dynamic_scale: Optional[float] = None,
    ):
        if current_timestep_module_scales is None and flag_overrides is None and apply_global_dynamic_scale is None:
            return base_map

        dynamic_map = copy.deepcopy(base_map)
        default_sft_active = flag_overrides.get("sft_all_active", None) if flag_overrides else None
        default_cross_active = flag_overrides.get("cross_all_active", None) if flag_overrides else None

        if flag_overrides is not None or default_sft_active is not None or default_cross_active is not None:
            for block_key, block_config in dynamic_map.items():
                if block_key == "post_mid":
                    stage_map = block_config
                    if default_sft_active is not None:
                        stage_map["sft_active"] = default_sft_active
                    key_specific_active = "sft_postmid_active"
                    if flag_overrides and key_specific_active in flag_overrides:
                        stage_map["sft_active"] = flag_overrides[key_specific_active]
                else:  # Up block
                    for i, stage_map in enumerate(block_config):
                        if default_sft_active is not None:
                            stage_map["sft_active"] = default_sft_active
                        key_sft_specific = f"sft_{block_key}_stage{i}_active"
                        if flag_overrides and key_sft_specific in flag_overrides:
                            stage_map["sft_active"] = flag_overrides[key_sft_specific]
                        if default_cross_active is not None:
                            stage_map["cross_active"] = default_cross_active
                        key_cross_specific = f"cross_{block_key}_stage{i}_active"
                        if flag_overrides and key_cross_specific in flag_overrides:
                            stage_map["cross_active"] = flag_overrides[key_cross_specific]

        if current_timestep_module_scales is not None:
            for block_key, block_config in dynamic_map.items():
                if block_key == "post_mid":
                    stage_map = block_config
                    if stage_map.get("sft_active"):
                        scale_key = "sft_postmid_scale"
                        if scale_key in current_timestep_module_scales:
                            stage_map["sft_scale"] = current_timestep_module_scales[scale_key]
                else:  # Up block
                    for i, stage_map in enumerate(block_config):
                        if stage_map.get("sft_active"):
                            scale_key = f"sft_{block_key}_stage{i}_scale"
                            if scale_key in current_timestep_module_scales:
                                stage_map["sft_scale"] = current_timestep_module_scales[scale_key]

                        if stage_map.get("cross_active"):
                            scale_key = f"cross_{block_key}_stage{i}_scale"
                            if scale_key in current_timestep_module_scales:
                                stage_map["cross_scale"] = current_timestep_module_scales[scale_key]
        elif apply_global_dynamic_scale is not None:
            for block_key, block_config in dynamic_map.items():
                if block_key == "post_mid":
                    stage_map = block_config
                    if stage_map.get("sft_active"):
                        stage_map["sft_scale"] = apply_global_dynamic_scale
                else:  # Up block
                    for i, stage_map in enumerate(block_config):
                        if stage_map.get("sft_active"):
                            stage_map["sft_scale"] = apply_global_dynamic_scale
                        if stage_map.get("cross_active"):
                            stage_map["cross_scale"] = apply_global_dynamic_scale

        return dynamic_map

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,  # ControlNet input
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # ControlNet input
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,  # For T2I Adapters
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        The forward pass of the UNet2DConditionModel model.

        Args:
            sample (`torch.Tensor`): Input noisy latent tensor.
            timestep (`Union[torch.Tensor, float, int]`): Current timestep.
            encoder_hidden_states (`torch.Tensor`): Text conditioning embeddings.
            class_labels (`Optional[torch.Tensor]`, *optional*): Class labels (not typically used with SDXL).
            timestep_cond (`Optional[torch.Tensor]`, *optional*): Additional timestep conditioning.
            attention_mask (`Optional[torch.Tensor]`, *optional*): Attention mask for self-attention.
            cross_attention_kwargs (`Optional[Dict[str, Any]]`, *optional*):
                Dictionary potentially containing SUPIR control info (`control_hs`, `control_scale`)
                and other kwargs for attention layers (like `scale` for LoRA or `gligen` args).
            added_cond_kwargs (`Optional[Dict[str, Any]]`, *optional*):
                Dictionary containing additional conditioning signals like `text_embeds` (pooled) and `time_ids`.
            down_block_additional_residuals (`Optional[Tuple[torch.Tensor]]`, *optional*):
                Residuals from ControlNet down blocks.
            mid_block_additional_residual (`Optional[torch.Tensor]`, *optional*):
                Residual from ControlNet mid block.
            down_intrablock_additional_residuals (`Optional[Tuple[torch.Tensor]]`, *optional*):
                Residuals for T2I-Adapter style injection within down blocks.
            encoder_attention_mask (`Optional[torch.Tensor]`, *optional*): Attention mask for cross-attention.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return a `UNet2DConditionOutput` or a tuple.

        Returns:
            `UNet2DConditionOutput` or `tuple`:
                If `return_dict` is True, returns `UNet2DConditionOutput` containing the predicted sample.
                Otherwise, returns a tuple containing the predicted sample.
        """

        logger.debug("\n=== UNet2DConditionModel Diffusers Forward Start ===")
        logger.debug(f"  Input sample (xt) shape: {sample.shape}, dtype: {sample.dtype}")
        if torch.is_tensor(timestep):
            logger.debug(f"  Input timestep shape: {timestep.shape}")
        else:
            logger.debug(f"  Input timestep: {timestep}")
        logger.debug(f"  Input encoder_hidden_states shape: {encoder_hidden_states.shape}")
        if added_cond_kwargs is not None:
            logger.debug(f"  Input added_cond_kwargs keys: {list(added_cond_kwargs.keys())}")
        else:
            logger.debug("  Input added_cond_kwargs: None")
        if cross_attention_kwargs is not None:
            logger.debug(f"  Input cross_attention_kwargs keys: {list(cross_attention_kwargs.keys())}")
        else:
            logger.debug("  Input cross_attention_kwargs: None")

        # 0. Check inputs
        if down_intrablock_additional_residuals is not None:
            logger.warning("down_intrablock_additional_residuals (T2I-Adapter) not fully implemented in UNet2DConditionModel forward.")

        # 1. Prepare parameters
        supir_mode = self.config.supir_mode is not None
        control_hs = None
        processed_cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        current_injection_module_scales_from_pipeline = processed_cross_attention_kwargs.pop("current_injection_module_scales", None)
        injection_flags_override = processed_cross_attention_kwargs.pop("injection_flags", None)
        control_scale_global_dynamic = processed_cross_attention_kwargs.pop("control_scale_global_dynamic", 1.0)
        global_scale_to_apply_if_needed = control_scale_global_dynamic if current_injection_module_scales_from_pipeline is None else None
        dynamic_injection_map = self._get_dynamic_injection_map(
            self.SUPIR_INJECTION_MAP,
            current_injection_module_scales_from_pipeline,
            injection_flags_override,
            apply_global_dynamic_scale=global_scale_to_apply_if_needed,
        )

        if supir_mode:
            control_hs = processed_cross_attention_kwargs.pop("control_hs", [])
            processed_cross_attention_kwargs.pop("control_scale", None)
            if not hasattr(self, "project_modules") or not self.project_modules:
                raise RuntimeError("SUPIR mode active but project_modules not built.")
            expected_controls = 10
            logger.debug("--- SUPIR Control Info ---")
            if control_hs is not None:
                logger.debug(f"  Received control_hs: List/Tuple of {len(control_hs)} tensors")
                if len(control_hs) != expected_controls:
                    logger.warning(f"Received {len(control_hs)} controls, expected {expected_controls}. Ensure ControlNet matches.")
                    if len(control_hs) > 0 and len(control_hs) < expected_controls:
                        raise ValueError(f"Insufficient controls: {len(control_hs)} < {expected_controls}")
                    if len(control_hs) > 0:
                        control_hs = control_hs[:expected_controls]

        default_overall_up_factor = 2 ** getattr(self, "num_upsamplers", 0)
        forward_upsample_size = False

        if default_overall_up_factor > 1 and any((s % default_overall_up_factor) != 0 for s in sample.shape[-2:]):
            logger.warning(
                f"Input sample size {sample.shape[-2:]} not divisible by default upsampling factor {default_overall_up_factor}. Enabling explicit upsample size."
            )
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(self.dtype)) * -10000.0
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            if encoder_attention_mask.ndim == 2:
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)

        # center input if configured
        if self.config.center_input_sample:
            sample = 2.0 * sample - 1.0

        # 2. Pre-process inputs
        sample = sample

        # 3. Time and conditional embeddings
        t_emb = self._get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        emb = emb.to(dtype=self.dtype)

        logger.debug("--- Timestep/Conditional Embedding ---")
        logger.debug(f"  t_emb shape: {t_emb.shape if t_emb is not None else 'None'}")
        logger.debug(f"  emb (after time_embedding): {emb.shape}")

        # Class embedding
        class_emb = self._get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb
            logger.debug(f"  emb (after class_emb): {emb.shape}")

        # Process added_cond_kwargs
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        aug_emb = self._get_aug_embed(
            emb=emb,
            added_cond_kwargs=added_cond_kwargs,
        )
        if aug_emb is not None:
            emb = emb.to(aug_emb.dtype) + aug_emb
            logger.debug(f"  emb (after aug_emb): {emb.shape}")

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
            logger.debug(f"  emb (after time_embed_act): {emb.shape}")

        # 4. Encoder hidden states processing
        proc_enc_hs = self._process_encoder_hidden_states(encoder_hidden_states, added_cond_kwargs)

        if isinstance(proc_enc_hs, tuple):
            final_encoder_hidden_states, add_hidden_states = proc_enc_hs
            if add_hidden_states is not None:
                logger.warning("IP-Adapter add_hidden_states processing not fully shown here.")
                pass
        else:
            final_encoder_hidden_states = proc_enc_hs
        logger.debug("--- Processed Encoder Hidden States ---")
        logger.debug(f"  final_encoder_hidden_states shape: {final_encoder_hidden_states.shape}")

        # 5. Input convolution
        sample = self.conv_in(sample)
        logger.debug("--- After Input Conv ---")
        logger.debug(f"  sample shape: {sample.shape}")

        # 6. Down Blocks
        lora_scale = processed_cross_attention_kwargs.pop("scale", 1.0)
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_ctrlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        down_block_res_samples = (sample,)
        logger.debug("\n--- Diffusers Down Blocks (Collecting Skips) ---")
        for i, down_block in enumerate(self.down_blocks):
            sample_in_shape = sample.shape
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                down_block_args = {
                    "hidden_states": sample,
                    "temb": emb,
                    "encoder_hidden_states": final_encoder_hidden_states,
                    "attention_mask": attention_mask,
                    "cross_attention_kwargs": processed_cross_attention_kwargs or None,
                    "encoder_attention_mask": encoder_attention_mask,
                }
                if "scale" in inspect.signature(down_block.forward).parameters:
                    down_block_args["scale"] = lora_scale
                sample, res_samples = down_block(**down_block_args)
            else:
                downblock_args = {"hidden_states": sample, "temb": emb}
                if "scale" in inspect.signature(down_block.forward).parameters:
                    downblock_args["scale"] = lora_scale
                sample, res_samples = down_block(**downblock_args)

            down_block_res_samples += res_samples
            logger.debug(f"  Down Block {i} ({type(down_block).__name__}): Input={sample_in_shape}, Output={sample.shape}, Skips Added={len(res_samples)}")

        logger.debug(f"  Total skip connections collected: {len(down_block_res_samples)}")

        if is_ctrlnet:
            logger.debug("--- Applying ControlNet Down Block Residuals ---")
            new_down_block_res_samples = ()
            for i, (res, ctrl_res) in enumerate(zip(down_block_res_samples, down_block_additional_residuals)):
                logger.debug(f"  Adding residual hs[{i}] ({ctrl_res.shape}) to skip ({res.shape})")
                new_down_block_res_samples += (res + ctrl_res,)
            down_block_res_samples = new_down_block_res_samples

        # 7. Mid Block
        logger.debug("\n--- Diffusers Mid Block ---")
        if self.mid_block:
            sample_in_mid_shape = sample.shape
            mid_block_args = {
                "hidden_states": sample,
                "temb": emb,
                "encoder_hidden_states": final_encoder_hidden_states,
                "attention_mask": attention_mask,
                "cross_attention_kwargs": processed_cross_attention_kwargs or None,
                "encoder_attention_mask": encoder_attention_mask,
            }
            if hasattr(self.mid_block, "has_cross_attention") and not self.mid_block.has_cross_attention:
                mid_block_args.pop("encoder_hidden_states", None)
                mid_block_args.pop("attention_mask", None)
                mid_block_args.pop("cross_attention_kwargs", None)
                mid_block_args.pop("encoder_attention_mask", None)
            if "scale" in inspect.signature(self.mid_block.forward).parameters:
                mid_block_args["scale"] = lora_scale

            sample = self.mid_block(**mid_block_args)
            logger.debug(f"  Mid Block Output (Before Injection/Residual): Input={sample_in_mid_shape}, Output={sample.shape}")

            if is_ctrlnet:
                logger.debug("--- Applying ControlNet Mid Block Residual ---")
                logger.debug(f"  Adding residual mid ({mid_block_additional_residual.shape}) to sample ({sample.shape})")
                sample = sample + mid_block_additional_residual
                logger.debug(f"    Shape after adding residual: {sample.shape}")

            if supir_mode and dynamic_injection_map is not None and "post_mid" in dynamic_injection_map:
                map_info = dynamic_injection_map["post_mid"]
                apply_sft_flag = map_info.get("sft_active", False)
                sft_scale = map_info.get("sft_scale", 1.0)
                ctrl_idx = map_info.get("sft_ctrl")
                proj_idx = map_info.get("sft_proj")

                logger.debug("\n--- Diffusers Post-Mid Injection ---")
                logger.debug(f"    Checking Post-Mid: ProjIdx={proj_idx}, CtrlIdx={ctrl_idx}, Scale={sft_scale}, Active={apply_sft_flag}")

                if (
                    apply_sft_flag
                    and ctrl_idx is not None
                    and proj_idx is not None
                    and control_hs is not None
                    and self.project_modules is not None
                    and ctrl_idx < len(control_hs)
                    and proj_idx < len(self.project_modules)
                ):
                    module = self.project_modules[proj_idx]
                    control = control_hs[ctrl_idx]
                    if isinstance(module, ZeroSFT) and control is not None:
                        logger.debug("\n--- Diffusers Post-Mid Injection ---")
                        logger.debug(f"  Calling project_modules[{proj_idx}] ({type(module).__name__}) with control[{ctrl_idx}]")
                        logger.debug(f"    sample shape (input 'h'): {sample.shape}")
                        logger.debug(f"    control shape ('c'): {control.shape}")
                        logger.debug("    h_ori (skip) is None")
                        logger.debug(f"    SFT scale: {sft_scale}")

                        sample = module(c=control.to(sample.dtype), h_skip=sample, h_decoder=None, control_scale=sft_scale)
                        logger.debug(f"    Output shape after injection: {sample.shape}")
                    elif control is None:
                        logger.warning(f"Skipping Post-MidBlock SFT: control_hs[{ctrl_idx}] is None.")
                    elif not isinstance(module, ZeroSFT):
                        logger.warning(f"Skipping Post-MidBlock SFT: project_modules[{proj_idx}] is not ZeroSFT ({type(module).__name__}).")
                else:
                    if not apply_sft_flag:
                        logger.debug("SFT not applied.")
                    else:
                        logger.warning(f"Conditions not met for Post-MidBlock SFT injection. ctrl_idx={ctrl_idx}, proj_idx={proj_idx}")
            else:
                logger.warning("No 'post_mid' key found in dynamic_injection_map.")

        logger.debug("\n--- Diffusers Up Blocks ---")
        for i, up_block in enumerate(self.up_blocks):
            logger.debug(f"\n--- Up Block {i} ---")
            up_block_supir_kwargs = {}
            if supir_mode:
                map_key = f"up_block_{i}"
                if map_key in dynamic_injection_map:
                    block_injection_stages = dynamic_injection_map.get(map_key, []) if supir_mode and dynamic_injection_map else []
                    up_block_supir_kwargs = {"control_hs": control_hs, "injection_stages_map": block_injection_stages, "project_modules": self.project_modules}
                else:
                    logger.warning(f"No injection map found for {map_key} in SUPIR_INJECTION_MAP_V2.")

            # Get skip connections based on the number of resnets in the up block
            res_samples_count = len(getattr(up_block, "resnets", []))
            res_for_block = down_block_res_samples[-res_samples_count:]
            down_block_res_samples = down_block_res_samples[:-res_samples_count]
            logger.debug(f"  Using Skip Connections (res_for_block): {[res.shape for res in res_for_block]}")

            current_upsample_size = None
            if i < len(self.up_blocks) - 1 and forward_upsample_size:
                if down_block_res_samples:
                    current_upsample_size = down_block_res_samples[-1].shape[2:]
                else:
                    logger.warning(f"Cannot determine upsample size for UpBlock {i} based on skips.")

            # Combine kwargs and call up block
            block_kwargs_for_call = processed_cross_attention_kwargs.copy() if processed_cross_attention_kwargs else {}
            block_kwargs_for_call.update(up_block_supir_kwargs)
            logger.debug(f"  Calling {type(up_block).__name__} with cross_attention_kwargs keys: {list(block_kwargs_for_call.keys())}")

            # Call the custom up block (SupirUpBlock2D or SupirCrossAttnUpBlock2D)
            up_block_args = {
                "hidden_states": sample,
                "temb": emb,
                "res_hidden_states_tuple": res_for_block,
                "encoder_hidden_states": final_encoder_hidden_states,
                "cross_attention_kwargs": block_kwargs_for_call if block_kwargs_for_call else None,
                "upsample_size": current_upsample_size,
                "attention_mask": attention_mask,
                "encoder_attention_mask": encoder_attention_mask,
            }
            valid_up_block_args = {k: v for k, v in up_block_args.items() if k in inspect.signature(up_block.forward).parameters}
            sample_in_up_shape = sample.shape
            sample = up_block(**valid_up_block_args)
            logger.debug(f"  Up Block {i} ({type(up_block).__name__}): Input={sample_in_up_shape}, Output={sample.shape}")
            logger.debug(f"--- End Up Block {i} ---")

        # 10. Post-process
        logger.debug("\n--- Diffusers Final Output Layer ---")
        sample_before_out = sample.shape
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        logger.debug(f"  Output Layer: Input={sample_before_out}, Final Output={sample.shape}")

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)
        return UNet2DConditionOutput(sample=sample)
