# This file implements the ControlNet model for the SUPIR pipeline.
#
# The core model architecture is ported from the original SUPIR project. It utilizes
# attention mechanisms from the Diffusers library (HuggingFace) and includes
# custom adaptations to integrate these components.
#
# Copyright (c) 2024 by SupPixel Pty Ltd. for the original ControlNet architecture.
# Copyright (c) 2025 by DEVAIEXP for the adaptation and integration work.
# Copyright 2024, The HuggingFace Team for original Diffusers components.
#
# As this module is a direct derivative of the SUPIR project and is functionally
# dependent on its proprietary components, its use is governed by the
# SUPIR Software License Agreement.
#
# Usage is strictly limited to NON-COMMERCIAL purposes.
#
# For the full license terms, please see the LICENSE_SUPIR.md file in the
# parent pipeline directory.


from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, repeat_only=False):
    if not repeat_only:        
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        half = dim // 2
        freqs = torch.exp(-torch.linspace(0, 1, half) * 9.0).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    else:
        return timesteps.repeat(1, dim)

class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, encoder_hidden_states=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, encoder_hidden_states=encoder_hidden_states)
            else:
                x = layer(x)
        return x

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout=0.0, out_channels=None, dims=2, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = GEGLU(dim, inner_dim) if glu else nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, disable_self_attn=False):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            cross_attention_dim=None if disable_self_attn else dim,
        )
        self.attn2 = Attention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            cross_attention_dim=context_dim,
        )
        self.ff = FeedForward(dim, glu=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.disable_self_attn = disable_self_attn
      
    def forward(self, x, encoder_hidden_states=None):
        x = self.attn1(self.norm1(x), encoder_hidden_states=encoder_hidden_states if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), encoder_hidden_states=encoder_hidden_states) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, context_dim=None, use_linear=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = GroupNorm32(32, in_channels)
        self.use_linear = use_linear
        if use_linear:
            self.proj_in = nn.Linear(in_channels, inner_dim)
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, 1))

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim)
            for _ in range(depth)
        ])

    def forward(self, x, encoder_hidden_states=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.use_linear:
            x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class SUPIRControlNetModel(ModelMixin, ConfigMixin):
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"]
    
    @register_to_config
    def __init__(
        self,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4, 2],
        channel_mult=[1, 2, 4],
        num_head_channels=64,
        transformer_depth=[1, 2, 10],
        context_dim=2048,
        adm_in_channels=2816,
        num_classes="sequential",
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
    ):
        super().__init__()        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        self.attention_resolutions = attention_resolutions
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes == "sequential":
            self.label_emb = nn.Sequential(
                nn.Sequential(
                    nn.Linear(adm_in_channels, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            )

        self.input_hint_block = TimestepEmbedSequential(
            zero_module(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        )
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [ResBlock(ch, time_embed_dim, out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    n_heads = ch // num_head_channels
                    depth = transformer_depth[level]
                    layers.append(
                        SpatialTransformer(ch, n_heads, num_head_channels, depth=depth, context_dim=context_dim, use_linear=use_linear_in_transformer)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, out_channels=ch)))
                input_block_chans.append(ch)
                ds *= 2

        n_heads = ch // num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim),
            SpatialTransformer(ch, n_heads, num_head_channels, depth=transformer_depth[-1], context_dim=context_dim, use_linear=use_linear_in_transformer),
            ResBlock(ch, time_embed_dim),
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,        
        controlnet_conditioning_scale: float = 1.0,
        guess_mode: bool = False,
        return_dict: bool = False,
    ):        
        logger.debug(f"\n=== SUPIRControlNetModel Diffusers Forward Start ===")
        logger.debug(f"  Input sample (xt) shape: {sample.shape}, dtype: {sample.dtype}")
        logger.debug(f"  Input timestep: {timestep}")
        logger.debug(f"  Input encoder_hidden_states shape: {encoder_hidden_states.shape}")
        logger.debug(f"  Input controlnet_cond (zLQ) shape: {controlnet_cond.shape}")
        if added_cond_kwargs is not None and "vector" in added_cond_kwargs:
            logger.debug(f"  Input added_cond_kwargs['vector'] shape: {added_cond_kwargs['vector'].shape}")
        elif added_cond_kwargs is not None:
            logger.debug(f"  Input added_cond_kwargs keys: {list(added_cond_kwargs.keys())} ('vector' missing?)")
        else:
            logger.debug(f"  Input added_cond_kwargs: None")
                
        _dtype = sample.dtype
        if encoder_hidden_states is not None: 
            encoder_hidden_states = encoder_hidden_states.to(_dtype)
        if controlnet_cond is not None: 
            controlnet_cond = controlnet_cond.to(_dtype)
        
        timestep_val = timestep.item() if torch.is_tensor(timestep) and timestep.numel() == 1 else timestep
        t_emb = timestep_embedding(timestep, self.model_channels, repeat_only=False).to(_dtype)
        emb = self.time_embed(t_emb)
        logger.debug(f"--- Timestep Embedding ---")
        logger.debug(f"  t_emb shape: {t_emb.shape}")
        logger.debug(f"  emb (after time_embed) shape: {emb.shape}")
        
        if self.num_classes == "sequential" and added_cond_kwargs:
            vector_input = added_cond_kwargs.get("vector")
            if vector_input is not None:
                logger.debug(f"--- Label Embedding ---")
                logger.debug(f"  Received vector shape: {vector_input.shape}")
                if not hasattr(self.label_emb[0][0], 'in_features'):
                    logger.debug("ERROR: Cannot check label_emb input dimension.")
                elif vector_input.shape[-1] != self.label_emb[0][0].in_features:
                    logger.error(f"Shape mismatch for label_emb: input vector has {vector_input.shape[-1]}, layer expects {self.label_emb[0][0].in_features}")                    
                else:
                    label_emb_out = self.label_emb(vector_input.to(emb.dtype))
                    logger.debug(f"  label_emb output shape: {label_emb_out.shape}")
                    emb = emb + label_emb_out
                    logger.debug(f"  emb (after adding label_emb) shape: {emb.shape}")
            else:
                logger.warning("'vector' not found in added_cond_kwargs for label_emb.")
                logger.debug(f"--- Label Embedding SKIPPED ('vector' not found) ---")
        elif self.num_classes is not None:
            logger.debug(f"--- Label Embedding SKIPPED (num_classes='{self.num_classes}') ---")

        logger.debug(f"--- Hint Processing ---")
        logger.debug(f"  Input to input_hint_block (controlnet_cond=zLQ): {controlnet_cond.shape}")
        
        guided_hint = self.input_hint_block(controlnet_cond, emb, encoder_hidden_states=encoder_hidden_states.to(emb.dtype))
        logger.debug(f"  Output of input_hint_block (guided_hint): {guided_hint.shape}")
        
        hs = []
        h = sample.to(self.dtype)
        logger.debug(f"\n--- ControlNet Input Blocks (Processing Sample/xt) ---")

        # Block 0 (Initial Conv)
        h_in_shape = h.shape
        h = self.input_blocks[0](h, emb, encoder_hidden_states=encoder_hidden_states)
        logger.debug(f"  Input Block 0 ({type(self.input_blocks[0]).__name__}): Input={h_in_shape}, Output={h.shape}")
      
        if guided_hint is not None:
            logger.debug(f"  (Adding guided_hint to h after block 0)")
            h = h + guided_hint.to(h.dtype)
            guided_hint = None
            logger.debug(f"    Shape after adding hint: {h.shape}")

        hs.append(h)
       
        for i, module in enumerate(self.input_blocks[1:], start=1):
            h_in_shape = h.shape            
            h = module(h, emb, encoder_hidden_states=encoder_hidden_states.to(emb.dtype))
            hs.append(h)            
            logger.debug(f"  Input Block {i} ({type(module).__name__}): Input={h_in_shape}, Output={h.shape}")

        # Middle Block
        logger.debug(f"\n--- ControlNet Mid Block ---")
        h_in_mid_shape = h.shape        
        h = self.middle_block(h, emb, encoder_hidden_states=encoder_hidden_states.to(emb.dtype))
        hs.append(h)        
        logger.debug(f"  Mid Block: Input={h_in_mid_shape}, Output={h.shape}")
        
        logger.debug(f"\n--- ControlNet Final Output (hs) ---")
        for i_hs, tensor_hs in enumerate(hs):
            logger.debug(f"  hs[{i_hs}]: {tensor_hs.shape}")
                
        return hs
        
    @property    
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)
     