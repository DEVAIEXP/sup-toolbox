# Copyright 2025, DEVAIEXP
#
# This file is a derivative work based on the FaithDiff project and the original
# Diffusers library by HuggingFace.
#
# Original FaithDiff work is Copyright (c) 2025, Junyang Chen
# and was licensed under the MIT License.
#
# Original Diffusers work is Copyright by The HuggingFace Team,
# and was licensed under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.accelerate_utils import apply_forward_hook
from safetensors.torch import load_file

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


    
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class Encoder(ModelMixin):
    """Encoder layer of a variational autoencoder that encodes input into a latent representation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.use_rgb = False
        self.down_block_type = down_block_types
        self.block_out_channels = block_out_channels

        self.tile_sample_min_size = 1024
        self.tile_latent_min_size = int(self.tile_sample_min_size / 8)
        self.tile_overlap_factor = 0.25
        self.use_tiling = False
        self.use_slicing = False

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.gradient_checkpointing = False

    def to_rgb_init(self):
        """Initialize layers to convert features to RGB."""
        self.to_rgbs = nn.ModuleList([])
        self.use_rgb = True
        for i, down_block_type in enumerate(self.down_block_type):
            output_channel = self.block_out_channels[i]
            self.to_rgbs.append(nn.Conv2d(output_channel, 3, kernel_size=3, padding=1))

    def enable_tiling(self):
        """Enable tiling for large inputs."""
        self.use_tiling = True

    def encoder(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """Encode the input tensor into a latent representation."""
        sample = self.conv_in(sample)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
            return sample
        else:
            for down_block in self.down_blocks:
                sample = down_block(sample)
            sample = self.mid_block(sample)
            
            return sample

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors vertically with a smooth transition."""
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors horizontally with a smooth transition."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode(x)

        enc = self.encoder(x)
        
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        return h
           
    def _tiled_encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Encode the input tensor using tiling for large inputs."""
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        return moments

    # def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
    #     """Forward pass of the encoder, using tiling if enabled for large inputs."""
    #     if self.use_tiling and (
    #         sample.shape[-1] > self.tile_latent_min_size or sample.shape[-2] > self.tile_latent_min_size
    #     ):
    #         return self._tiled_encode(sample)
    #     return self.encoder(sample)

class ControlNetConditioningEmbedding(nn.Module):
    """A small network to preprocess conditioning inputs, inspired by ControlNet."""

    def __init__(self, conditioning_embedding_channels: int, conditioning_channels: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, conditioning_channels, kernel_size=3, padding=1)
        self.norm_in = nn.GroupNorm(num_channels=conditioning_channels, num_groups=32, eps=1e-6)
        self.conv_out = zero_module(
            nn.Conv2d(conditioning_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        """Process the conditioning input through the network."""
        conditioning = self.norm_in(conditioning)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class QuickGELU(nn.Module):
    """A fast approximation of the GELU activation function."""

    def forward(self, x: torch.Tensor):
        """Apply the QuickGELU activation to the input tensor."""
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        """Apply LayerNorm and preserve the input dtype."""
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    """A transformer-style block with self-attention and an MLP."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 2)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 2, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """Apply self-attention to the input tensor."""
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """Forward pass through the residual attention block."""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """The output of UnifiedUNet2DConditionModel."""

    sample: torch.FloatTensor = None


class UNet2DConditionModel(OriginalUNet2DConditionModel, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    """A unified 2D UNet model extending OriginalUNet2DConditionModel with custom functionality."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
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
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        """Initialize the UnifiedUNet2DConditionModel."""
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )

        # Additional attributes
        self.denoise_encoder = None
        self.information_transformer_layes = None
        self.condition_embedding = None
        self.agg_net = None
        self.spatial_ch_projs = None

    def init_vae_encoder(self, dtype):
        self.denoise_encoder = Encoder()
        if dtype is not None:
            self.denoise_encoder.to(dtype)

    def init_information_transformer_layes(self):
        num_trans_channel = 640
        num_trans_head = 8
        num_trans_layer = 2
        num_proj_channel = 320
        self.information_transformer_layes = nn.Sequential(
            *[ResidualAttentionBlock(num_trans_channel, num_trans_head) for _ in range(num_trans_layer)]
        )
        self.spatial_ch_projs = zero_module(nn.Linear(num_trans_channel, num_proj_channel))

    def init_ControlNetConditioningEmbedding(self, channel=512):
        self.condition_embedding = ControlNetConditioningEmbedding(320, channel)

    def init_extra_weights(self):
        self.agg_net = nn.ModuleList()

    def load_additional_layers(
        self, dtype: Optional[torch.dtype] = torch.float16, channel: int = 512, weight_path: Optional[str] = None
    ):
        """Load additional layers and weights from a file.

        Args:
            weight_path (str): Path to the weight file.
            dtype (torch.dtype, optional): Data type for the loaded weights. Defaults to torch.float16.
            channel (int): Conditioning embedding channel out size. Defaults 512.
        """
        if self.denoise_encoder is None:
            self.init_vae_encoder(dtype)

        if self.information_transformer_layes is None:
            self.init_information_transformer_layes()

        if self.condition_embedding is None:
            self.init_ControlNetConditioningEmbedding(channel)

        if self.agg_net is None:
            self.init_extra_weights()

        # Load weights if provided
        if weight_path is not None:
            is_safetensors = weight_path.endswith(".safetensors")
            is_pickle = weight_path.endswith(".ckpt") or weight_path.endswith(".bin")
            if not is_safetensors and not is_pickle: #try set default 
                weight_path = os.path.join(weight_path, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(weight_path):
                raise ValueError(f"`weigth_path` supplied: {weight_path}, not exists!")
            if is_pickle:
                state_dict = torch.load(weight_path, weights_only=False)
            else:
                state_dict = load_file(weight_path, device="cpu")
            self.load_state_dict(state_dict, strict=True)

        # Move all modules to the same device and dtype as the model
        device = next(self.parameters()).device
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype or next(self.parameters()).dtype)

    def to(self, *args, **kwargs):
        """Override to() to move all additional modules to the same device and dtype."""
        super().to(*args, **kwargs)
        for module in [
            self.denoise_encoder,
            self.information_transformer_layes,
            self.condition_embedding,
            self.agg_net,
            self.spatial_ch_projs,
        ]:
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
            strict (bool, optional): Whether to strictly enforce that all keys match. Defaults to True.
        """
        core_dict = {}
        additional_dicts = {
            "denoise_encoder": {},
            "information_transformer_layes": {},
            "condition_embedding": {},
            "agg_net": {},
            "spatial_ch_projs": {},
        }

        for key, value in state_dict.items():
            if key.startswith("denoise_encoder."):
                additional_dicts["denoise_encoder"][key[len("denoise_encoder.") :]] = value
            elif key.startswith("information_transformer_layes."):
                additional_dicts["information_transformer_layes"][key[len("information_transformer_layes.") :]] = value
            elif key.startswith("condition_embedding."):
                additional_dicts["condition_embedding"][key[len("condition_embedding.") :]] = value
            elif key.startswith("agg_net."):
                additional_dicts["agg_net"][key[len("agg_net.") :]] = value
            elif key.startswith("spatial_ch_projs."):
                additional_dicts["spatial_ch_projs"][key[len("spatial_ch_projs.") :]] = value
            else:
                core_dict[key] = value

        super().load_state_dict(core_dict, strict=False)
        for module_name, module_dict in additional_dicts.items():
            module = getattr(self, module_name, None)
            if module is not None and module_dict:
                module.load_state_dict(module_dict, strict=strict)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        input_embedding: Optional[torch.Tensor] = None,
        current_control_scale: float = 1.0,
        add_sample: bool = True,
        return_dict: bool = True,
        use_condition_embedding: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """Forward pass prioritizing the original modified implementation.

        Args:
            sample (torch.FloatTensor): The noisy input tensor with shape `(batch, channel, height, width)`.
            timestep (Union[torch.Tensor, float, int]): The number of timesteps to denoise an input.
            encoder_hidden_states (torch.Tensor): The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (torch.Tensor, optional): Optional class labels for conditioning.
            timestep_cond (torch.Tensor, optional): Conditional embeddings for timestep.
            attention_mask (torch.Tensor, optional): An attention mask of shape `(batch, key_tokens)`.
            cross_attention_kwargs (Dict[str, Any], optional): A kwargs dictionary for the AttentionProcessor.
            added_cond_kwargs (Dict[str, torch.Tensor], optional): Additional embeddings to add to the UNet blocks.
            down_block_additional_residuals (Tuple[torch.Tensor], optional): Residuals for down UNet blocks.
            mid_block_additional_residual (torch.Tensor, optional): Residual for the middle UNet block.
            down_intrablock_additional_residuals (Tuple[torch.Tensor], optional): Additional residuals within down blocks.
            encoder_attention_mask (torch.Tensor, optional): A cross-attention mask of shape `(batch, sequence_length)`.
            input_embedding (torch.Tensor, optional): Additional input embedding for preprocessing.
            add_sample (bool): Whether to add the sample to the processed embedding. Defaults to True.
            return_dict (bool): Whether to return a UNet2DConditionOutput. Defaults to True.
            use_condition_embedding (bool): Whether to use the condition embedding. Defaults to True.

        Returns:
            Union[UNet2DConditionOutput, Tuple]: The processed sample tensor, either as a UNet2DConditionOutput or tuple.
        """
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process (following the original modified logic)
        base_sample  = self.conv_in(sample)  # [B, 4, H, W] -> [B, 320, H, W]
        if (
            input_embedding is not None
            and self.condition_embedding is not None
            and self.information_transformer_layes is not None
        ):
            conditioned_hint: torch.Tensor
            if use_condition_embedding:
                conditioned_hint = self.condition_embedding(input_embedding)  # [B, 320, H, W]
            else:
                conditioned_hint = input_embedding
                
            if base_sample.shape[2:] != conditioned_hint.shape[2:]:                
                conditioned_hint = F.interpolate(conditioned_hint, size=base_sample.shape[2:], mode="bilinear", align_corners=False)

            if base_sample.shape[1] != conditioned_hint.shape[1]:
                raise ValueError(
                    f"Channel mismatch after condition_embedding: base_sample has {base_sample.shape[1]} channels, "
                    f"conditioned_hint has {conditioned_hint.shape[1]} channels. Both should be {self.config.block_out_channels[0]}."
                )

            batch_size, channel_unet, height, width = base_sample.shape
           
            concat_feat_input = torch.cat([base_sample, conditioned_hint], dim=1)
            if concat_feat_input.shape[1] != self.information_transformer_layes[0].attn.embed_dim:
                 raise ValueError(
                    f"Input dimension for information_transformer_layes mismatch. "
                    f"Expected {self.information_transformer_layes[0].attn.embed_dim}, got {concat_feat_input.shape[1]}"
                )
            concat_feat_input_reshaped = concat_feat_input.view(batch_size, 2 * channel_unet, height * width).transpose(1, 2)

            transformed_feat = self.information_transformer_layes(concat_feat_input_reshaped) # [B, H*W, 640]
            feat_alpha = self.spatial_ch_projs(transformed_feat) # [B, H*W, 320]
            feat_alpha = feat_alpha.transpose(1, 2).view(batch_size, channel_unet, height, width) # [B, 320, H, W]            
            
            if add_sample:               
                output_sample = base_sample + feat_alpha * current_control_scale
            else:                
                output_sample = feat_alpha * current_control_scale
        else:
             output_sample = base_sample
        
        sample = output_sample    
        # 2.5 GLIGEN position net (kept from the original version)
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down (continues the standard flow)
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)
            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)
        return UNet2DConditionOutput(sample=sample)

