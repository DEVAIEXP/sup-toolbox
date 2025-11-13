# This autoencoder_kl file is a derivative work.
# It originates from Diffusers (The HuggingFace Team, Apache 2.0 License)
# Modifications and adaptations to this file are Copyright (c) 2025 by DEVAIEXP.

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from diffusers.models.autoencoders.vae import (
    Decoder,
    DecoderOutput,
    DiagonalGaussianDistribution,
    Encoder,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import deprecate
from diffusers.utils.accelerate_utils import apply_forward_hook


class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without losing too much precision in which case `force_upcast`
            can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False
        self.run_vae_on_cpu = False
        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = self.config.sample_size[0] if isinstance(self.config.sample_size, (list, tuple)) else self.config.sample_size
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def enable_run_on_cpu(self, run_on_cpu: bool = True):
        """
        Enable/disable running VAE computations on the CPU to save VRAM.

        When this is enabled, the main VAE modules (encoder/decoder) and
        their data will be processed on the CPU. This significantly reduces
        VRAM usage at the cost of processing speed.
        """
        self.run_vae_on_cpu = run_on_cpu

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

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
            raise ValueError(f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}")

        self.set_attn_processor(processor)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode(x)

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
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

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True, generator=None) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

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

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decodes a batch of latent vectors into images using a tiled approach,
        with an optional mode to run all computations on the CPU to save VRAM.

        When tiling is enabled, this method splits the latent tensor `z` into
        overlapping tiles and decodes them sequentially. If `run_vae_on_cpu` is
        enabled, the decoder module and each tile are moved to the CPU before
        computation. This significantly reduces VRAM usage at the cost of
        slower processing speed. The resulting decoded tiles are then moved
        back to the original GPU device for blending to avoid seams.

        Args:
            z (`torch.Tensor`):
                Input batch of latent vectors, expected on the target GPU.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead
                of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If `return_dict` is True, a [`~models.vae.DecoderOutput`]
                is returned, otherwise a plain `tuple` is returned. The `sample`
                field contains the final decoded image on the original device.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        cpu_device = torch.device("cpu")
        original_device = z.device

        # If `run_on_cpu` is enabled, move the relevant modules to the CPU once before the loop.
        if self.run_vae_on_cpu:
            if self.config.use_post_quant_conv:
                self.post_quant_conv.to(cpu_device)
            self.decoder.to(cpu_device)

        # Process each tile.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]

                # If CPU mode is on, move the data tile to the CPU for processing.
                if self.run_vae_on_cpu:
                    tile = tile.to(cpu_device)

                # The main computation happens on the modules' current device (CPU or GPU).
                if self.config.use_post_quant_conv:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                torch.cuda.empty_cache()
                # The `decoded` tile result is kept on the CPU if in CPU mode
                # to minimize VRAM usage until the final blending stage.
                row.append(decoded)
            rows.append(row)

        # After the loop, move modules back to the original device if they were moved.
        if self.run_vae_on_cpu:
            if self.config.use_post_quant_conv:
                self.post_quant_conv.to(original_device)
            self.decoder.to(original_device)

        # --- Blending Stage ---
        # The blending operations are performed on the GPU for performance.
        # We move each tile to the GPU just-in-time for blending.
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # Ensure the current tile is on the GPU for blending operations.
                tile_on_gpu = tile.to(original_device)

                if i > 0:
                    # Ensure the tile from the previous row is also on the GPU.
                    tile_on_gpu = self.blend_v(rows[i - 1][j].to(original_device), tile_on_gpu, blend_extent)
                if j > 0:
                    # Ensure the tile from the previous column is also on the GPU.
                    tile_on_gpu = self.blend_h(row[j - 1].to(original_device), tile_on_gpu, blend_extent)

                result_row.append(tile_on_gpu[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""
        Encodes a batch of images into latents using a tiled approach,
        with an optional mode to run all computations on the CPU to save VRAM.

        When tiling is enabled, this method splits the input image `x` into
        overlapping tiles and encodes them sequentially. If `run_vae_on_cpu` is
        enabled, the encoder module and each image tile are moved to the CPU
        for computation. This significantly reduces VRAM usage at the cost of
        slower processing speed. The resulting latent tiles are then moved back
        to the original GPU device for blending.

        Args:
            x (`torch.Tensor`):
                Input batch of images, expected on the target GPU.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`]
                instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.autoencoder_kl.AutoencoderKLOutput`]
                is returned, otherwise a plain `tuple` is returned. The `latent_dist`
                field contains the distribution of the final encoded latents.
        """
        deprecation_message = "The tiled_encode implementation supporting the `return_dict` parameter is deprecated..."
        deprecate("tiled_encode", "1.0.0", deprecation_message, standard_warn=False)

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        cpu_device = torch.device("cpu")
        original_device = x.device

        if self.run_vae_on_cpu:
            self.encoder.to(cpu_device)
            if self.config.use_quant_conv:
                self.quant_conv.to(cpu_device)

        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]

                if self.run_vae_on_cpu:
                    tile = tile.to(cpu_device)

                tile = self.encoder(tile)
                if self.config.use_quant_conv:
                    tile = self.quant_conv(tile)
                    torch.cuda.empty_cache()
                row.append(tile)
            rows.append(row)

        if self.run_vae_on_cpu:
            self.encoder.to(original_device)
            if self.config.use_quant_conv:
                self.quant_conv.to(original_device)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                tile_on_gpu = tile.to(original_device)
                if i > 0:
                    tile_on_gpu = self.blend_v(rows[i - 1][j].to(original_device), tile_on_gpu, blend_extent)
                if j > 0:
                    tile_on_gpu = self.blend_h(row[j - 1].to(original_device), tile_on_gpu, blend_extent)
                result_row.append(tile_on_gpu[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
