# This file is part of a pipeline designed to run the SUPIR model.
# It is adapted from original Diffusers code (HuggingFace, Apache 2.0 License).
#
# Copyright 2024, The HuggingFace Team for original Diffusers components.
# Modifications and adaptations to this file are Copyright (c) 2025 by DEVAIEXP.
#
# While the code itself is based on a permissive license, it is specifically
# designed to load and operate the SUPIR model, which is under a restrictive
# non-commercial license.
#
# Consequently, the use of this pipeline is governed by the terms of the
# SUPIR Software License Agreement. Its use is strictly limited to
# NON-COMMERCIAL purposes.
#
# For the full license terms, please see the LICENSE_SUPIR.md file in this directory.

import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, Encoder
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.utils.accelerate_utils import apply_forward_hook


logger = logging.get_logger(__name__)


class AutoencoderKLSUPIROutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: DiagonalGaussianDistribution  # noqa: F821


class AutoencoderKLSUPIR(ModelMixin, ConfigMixin):
    r"""A model implementing the AutoencoderKLSUPIR from SUPIR, designed to encode images into latent representations
    with tiling support. This model does not include a decoder and is intended for use with an external decoder (e.g., SDXL).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int, *optional*, defaults to 4): Number of channels in the latent space.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)` * 4):
            Tuple of downsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(128, 256, 512, 512)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to 2): Number of layers per block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): Number of groups for group normalization.
        sample_size (`int`, *optional*, defaults to 256): Sample input size.
        mid_block_add_attention (`bool`, *optional*, defaults to True):
            If enabled, the mid_block of the Encoder will have attention blocks.
        tile_sample_min_size (`int`, *optional*, defaults to 512): Minimum tile size for tiled encoding.
        tile_overlap_factor (`float`, *optional*, defaults to 0.25): Overlap factor for tiled encoding.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",) * 4,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.13025,
        double_z: bool = True,
        latent_embed_dim: int = 4,
        sample_size=32,
        use_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=double_z,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels if double_z else latent_channels, 2 * latent_embed_dim, 1) if use_quant_conv else None
        self.latent_embed_dim = latent_embed_dim

        if mid_block_add_attention:
            resnet_copy = copy.deepcopy(self.encoder.mid_block.resnets[0])
            self.encoder.mid_block.resnets = nn.ModuleList([self.encoder.mid_block.resnets[0], resnet_copy])
            block_in = block_out_channels[-1]
            self.encoder.mid_block.attentions[0].heads = 1
            self.encoder.mid_block.attentions[0].dim_head = block_in

        self.use_tiling = False
        self.scale_factor = scaling_factor

        self.tile_sample_min_size = self.config.sample_size
        sample_size = self.config.sample_size[0] if isinstance(self.config.sample_size, (list, tuple)) else self.config.sample_size
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent
        self.quant_conv = self.quant_conv.to(x.device)

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                if self.config.use_quant_conv:
                    tile = self.quant_conv(tile)
                    torch.cuda.empty_cache()
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        enc = torch.cat(result_rows, dim=2)
        return enc

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLSUPIROutput, Tuple[torch.Tensor]]:
        """
        Encodes the input tensor `x` into latents distribution.

        Args:
            x (`torch.Tensor`): Input tensor of shape `(N, C, H, W)`.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return `AutoencoderKLSUPIROutput`.

        Returns:
            `AutoencoderKLSUPIROutput` or `tuple`: Encoded latents.
        """

        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            moments = self._tiled_encode(x)
        else:
            h = self.encoder(x)
            self.quant_conv = self.quant_conv.to(x.device)
            moments = self.quant_conv(h)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            z = posterior.mode() * self.scale_factor
            return (z,)

        return AutoencoderKLSUPIROutput(latent_dist=posterior)
