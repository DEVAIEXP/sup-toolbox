# Copyright 2025, DEVAIEXP
#
# This file integrates the "Mixture of Diffusers" technique into a
# ControlNet pipeline from the HuggingFace Diffusers library.
#
# The "Mixture of Diffusers" technique is Copyright (c) 2022, Álvaro Barbero Jiménez,
# and was licensed under the MIT License.
#
# The base ControlNet pipeline and Diffusers library are Copyright 2024,
# The HuggingFace Team, and are licensed under the Apache License, Version 2.0.
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

import gc
import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    ControlNetUnionModel,
    MultiControlNetModel,
    UNet2DConditionModel,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import EulerDiscreteScheduler, LMSDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.import_utils import is_invisible_watermark_available
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from PIL import Image
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from diffusers.utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
"""


# This function was copied and adapted from https://huggingface.co/spaces/gokaygokay/TileUpscalerV2, licensed under Apache 2.0.
def _adaptive_tile_size(image_size, base_tile_size=512, max_tile_size=1280):
    """
    Calculate the adaptive tile size based on the image dimensions, ensuring the tile
    respects the aspect ratio and stays within the specified size limits.
    """
    width, height = image_size
    aspect_ratio = width / height

    if aspect_ratio > 1:
        # Landscape orientation
        tile_width = min(width, max_tile_size)
        tile_height = min(int(tile_width / aspect_ratio), max_tile_size)
    else:
        # Portrait or square orientation
        tile_height = min(height, max_tile_size)
        tile_width = min(int(tile_height * aspect_ratio), max_tile_size)

    # Ensure the tile size is not smaller than the base_tile_size
    tile_width = max(tile_width, base_tile_size)
    tile_height = max(tile_height, base_tile_size)

    return tile_width, tile_height


# Copied and adapted from https://github.com/huggingface/diffusers/blob/main/examples/community/mixture_tiling.py
def _tile2pixel_indices(
    tile_row,
    tile_col,
    tile_width,
    tile_height,
    tile_row_overlap,
    tile_col_overlap,
    image_width,
    image_height,
):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    # Calculate initial indices
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)

    # Calculate end indices
    px_row_end = px_row_init + tile_height
    px_col_end = px_col_init + tile_width

    # Ensure the last tile does not exceed the image dimensions
    px_row_end = min(px_row_end, image_height)
    px_col_end = min(px_col_end, image_width)

    return px_row_init, px_row_end, px_col_init, px_col_end


# Copied and adapted from https://github.com/huggingface/diffusers/blob/main/examples/community/mixture_tiling.py
def _tile2latent_indices(
    tile_row,
    tile_col,
    tile_width,
    tile_height,
    tile_row_overlap,
    tile_col_overlap,
    image_width,
    image_height,
):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    # Get pixel indices
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
        tile_row,
        tile_col,
        tile_width,
        tile_height,
        tile_row_overlap,
        tile_col_overlap,
        image_width,
        image_height,
    )

    # Convert to latent space
    latent_row_init = px_row_init // 8
    latent_row_end = px_row_end // 8
    latent_col_init = px_col_init // 8
    latent_col_end = px_col_end // 8
    latent_height = image_height // 8
    latent_width = image_width // 8

    # Ensure the last tile does not exceed the latent dimensions
    latent_row_end = min(latent_row_end, latent_height)
    latent_col_end = min(latent_col_end, latent_width)

    return latent_row_init, latent_row_end, latent_col_init, latent_col_end


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4

    Args:
        noise_cfg (torch.Tensor): Noise configuration tensor.
        noise_pred_text (torch.Tensor): Predicted noise from text-conditioned model.
        guidance_rescale (float): Rescaling factor for guidance. Defaults to 0.0.

    Returns:
        torch.Tensor: Rescaled noise configuration.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def release_memory(device):
    r"""
    Run torch garbage collector to free up memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class StableDiffusionXLControlNetTileSRPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for image-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetUnionModel`]):
            Provides additional conditioning to the unet during the denoising process.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
            Whether the `unet` requires an `aesthetic_score` condition to be passed during inference. Also see the
            config of `stabilityai/stable-diffusion-xl-refiner-1-0`.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->vae->unet"

    _optional_components = [
        "controlnet",
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
    ]

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetUnionModel,
        scheduler: EulerDiscreteScheduler,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        if not isinstance(controlnet, ControlNetUnionModel):
            raise ValueError("Expected `controlnet` to be of type `ControlNetUnionModel`.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

    class TileWeightingMethod(Enum):
        """Mode in which the tile weights will be generated"""

        COSINE = "Cosine"
        GAUSSIAN = "Gaussian"

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        needs_upcasting = self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False)
        original_vae_dtype = self.vae.dtype

        with torch.inference_mode():
            if needs_upcasting:
                logger.info(f"Upcasting VAE decoder to {torch.float32} for decoding...")
                self.vae.to(dtype=torch.float32)
                latents = latents.to(torch.float32)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            # cast back to original precision if needed
            if needs_upcasting:
                logger.info("Casting VAE back to original dtype.")
                self.vae.to(dtype=original_vae_dtype)

        return image

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """

        device = device or self._execution_device
        with torch.inference_mode():
            # set lora scale so that monkey patched LoRA
            # function of text encoder can correctly access it
            if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
                self._lora_scale = lora_scale

                # dynamically adjust the LoRA scale
                if self.text_encoder is not None:
                    if not USE_PEFT_BACKEND:
                        adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                    else:
                        scale_lora_layers(self.text_encoder, lora_scale)

                if self.text_encoder_2 is not None:
                    if not USE_PEFT_BACKEND:
                        adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                    else:
                        scale_lora_layers(self.text_encoder_2, lora_scale)

            prompt = [prompt] if isinstance(prompt, str) else prompt

            if prompt is not None:
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            # Define tokenizers and text encoders
            tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
            text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
            dtype = text_encoders[0].dtype
            if prompt_embeds is None:
                prompt_2 = prompt_2 or prompt
                prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

                # textual inversion: process multi-vector tokens if necessary
                prompt_embeds_list = []
                prompts = [prompt, prompt_2]
                for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                    if isinstance(self, TextualInversionLoaderMixin):
                        prompt = self.maybe_convert_prompt(prompt, tokenizer)

                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {tokenizer.model_max_length} tokens: {removed_text}"
                        )
                    text_encoder.to(dtype)
                    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    if pooled_prompt_embeds is None and prompt_embeds[0].ndim == 2:
                        pooled_prompt_embeds = prompt_embeds[0]

                    if clip_skip is None:
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                    else:
                        # "2" because SDXL always indexes from the penultimate layer.
                        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                    prompt_embeds_list.append(prompt_embeds)

                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            # get unconditional embeddings for classifier free guidance
            zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
            if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            elif do_classifier_free_guidance and negative_prompt_embeds is None:
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt_2 or negative_prompt

                # normalize str to list
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

                uncond_tokens: List[str]
                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.")
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = [negative_prompt, negative_prompt_2]

                negative_prompt_embeds_list = []
                for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                    if isinstance(self, TextualInversionLoaderMixin):
                        negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    negative_prompt_embeds = text_encoder(
                        uncond_input.input_ids.to(text_encoder.device),
                        output_hidden_states=True,
                    )

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    if negative_pooled_prompt_embeds is None and negative_prompt_embeds[0].ndim == 2:
                        negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                    negative_prompt_embeds_list.append(negative_prompt_embeds)

                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

            if self.text_encoder_2 is not None:
                prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                if self.text_encoder_2 is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
                else:
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
            if do_classifier_free_guidance:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)

            if self.text_encoder is not None:
                if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        image,
        strength,
        num_inference_steps,
        tile_overlap,
        tile_size,
        tile_gaussian_sigma,
        tile_weighting_method,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` cannot be None.")
        elif not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` has to be a positive integer but is {num_inference_steps} of type {type(num_inference_steps)}.")
        if tile_overlap is None:
            raise ValueError("`tile_overlap` cannot be None.")
        elif not isinstance(tile_overlap, int) or tile_overlap < 64:
            raise ValueError(f"`tile_overlap` has to be greater than 64 but is {tile_overlap} of type {type(tile_overlap)}.")
        if tile_size is None:
            raise ValueError("`tile_size` cannot be None.")
        elif not isinstance(tile_size, int) or tile_size not in (1024, 1280):
            raise ValueError(f"`tile_size` has to be in 1024 or 1280 but is {tile_size} of type {type(tile_size)}.")
        if tile_gaussian_sigma is None:
            raise ValueError("`tile_gaussian_sigma` cannot be None.")
        elif not isinstance(tile_gaussian_sigma, float) or tile_gaussian_sigma <= 0:
            raise ValueError(f"`tile_gaussian_sigma` has to be a positive float but is {tile_gaussian_sigma} of type {type(tile_gaussian_sigma)}.")
        if tile_weighting_method is None:
            raise ValueError("`tile_weighting_method` cannot be None.")
        elif not isinstance(tile_weighting_method, str) or tile_weighting_method not in [t.value for t in self.TileWeightingMethod]:
            raise ValueError(
                f"`tile_weighting_method` has to be a string in ({[t.value for t in self.TileWeightingMethod]}) but is {tile_weighting_method} of type"
                f" {type(tile_weighting_method)}."
            )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(self.controlnet, torch._dynamo.eval_frame.OptimizedModule)
        if isinstance(self.controlnet, ControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, ControlNetModel):
            self.check_image(image, prompt)
        elif isinstance(self.controlnet, ControlNetUnionModel) or is_compiled and isinstance(self.controlnet._orig_mod, ControlNetUnionModel):
            self.check_image(image, prompt)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (isinstance(self.controlnet, ControlNetUnionModel) or is_compiled and isinstance(self.controlnet._orig_mod, ControlNetUnionModel)) or (
            isinstance(self.controlnet, MultiControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif isinstance(self.controlnet, MultiControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, MultiControlNetModel):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(self.controlnet.nets):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}.")
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.check_image
    def check_image(self, image, prompt):
        image_is_pil = isinstance(image, Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if not image_is_pil and not image_is_tensor and not image_is_np and not image_is_pil_list and not image_is_tensor_list and not image_is_np_list:
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl.StableDiffusionXLControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.prepare_latents
    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        device,
        generator=None,
        add_noise=True,
    ):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        latents_mean = latents_std = None
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)

        dtype = self.vae.dtype
        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        with torch.inference_mode():
            if image.shape[1] == 4:
                init_latents = image

            else:
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype != torch.float32 and getattr(self.vae.config, "force_upcast", False) and getattr(self.vae, "to", None)
                if needs_upcasting:
                    image = image.float()
                    self.vae.to(dtype=torch.float32)

                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                elif isinstance(generator, list):
                    if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                        image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                    elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                        raise ValueError(f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} ")

                    init_latents = [retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i]) for i in range(batch_size)]
                    init_latents = torch.cat(init_latents, dim=0)
                else:
                    init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

                if needs_upcasting:
                    self.vae.to(dtype)

                if latents_mean is not None and latents_std is not None:
                    latents_mean = latents_mean.to(device=device, dtype=dtype)
                    latents_std = latents_std.to(device=device, dtype=dtype)
                    init_latents = (init_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
                else:
                    init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.")
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    def prepare_tiles_weights(
        self,
        grid_rows: int,
        grid_cols: int,
        tile_height: int,
        tile_width: int,
        tile_overlap: int,
        height: int,
        width: int,
        tile_weighting_method: str,
        tile_gaussian_sigma: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> np.ndarray:
        """
        Pre-calculates the blending weights for each tile in the grid.

        This function iterates through each tile's grid position (row, col) and determines its
        exact pixel dimensions, which is crucial for handling border tiles that may be smaller
        than the standard tile size. It then generates a corresponding weight map (either
        Gaussian or Cosine) for blending the noise predictions during the stitching process.

        Args:
            grid_rows (int): The number of rows in the tile grid.
            grid_cols (int): The number of columns in the tile grid.
            tile_height (int): The standard height of a tile in pixels.
            tile_width (int): The standard width of a tile in pixels.
            tile_overlap (int): The number of overlapping pixels between adjacent tiles.
            height (int): The total height of the full image in pixels.
            width (int): The total width of the full image in pixels.
            tile_weighting_method (str): The method for weighting tiles. Options: "Cosine" or "Gaussian".
            tile_gaussian_sigma (float): The sigma parameter for Gaussian weighting.
            batch_size (int): The batch size to use for the weight tensors (typically 1).
            device (torch.device): The device where the weight tensors will be allocated (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): The data type of the weight tensors (e.g., torch.float32).

        Returns:
            np.ndarray: A 2D numpy array of objects, where each element `tile_weights[row, col]`
                        contains a pre-calculated `torch.Tensor` of blending weights for that specific tile.
        """
        # Create a numpy array to store the torch tensors of weights for each tile.
        # Using dtype=object allows storing tensors of different shapes.
        tile_weights = np.empty((grid_rows, grid_cols), dtype=object)

        # Iterate over the grid indices to generate weights for each tile's specific size.
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate the exact pixel dimensions of the current tile.
                # This helper function accounts for the tile's position and ensures it doesn't exceed image bounds.
                px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
                    row,
                    col,
                    tile_width,
                    tile_height,
                    tile_overlap,  # Use the single overlap value for both dimensions
                    tile_overlap,
                    width,
                    height,
                )
                current_tile_pixel_height = px_row_end - px_row_init
                current_tile_pixel_width = px_col_end - px_col_init

                # Generate the weight map for the current tile's exact dimensions.
                if tile_weighting_method == self.TileWeightingMethod.COSINE.value:
                    tile_weights[row, col] = self._generate_cosine_weights(
                        tile_width=current_tile_pixel_width,
                        tile_height=current_tile_pixel_height,
                        nbatches=batch_size,
                        device=device,
                        dtype=torch.float32,  # Cosine weights can be float32 for precision.
                    )
                else:  # Default to Gaussian
                    tile_weights[row, col] = self._generate_gaussian_weights(
                        tile_width=current_tile_pixel_width,
                        tile_height=current_tile_pixel_height,
                        nbatches=batch_size,
                        device=device,
                        dtype=torch.float32,  # Match the main processing dtype.
                        sigma=tile_gaussian_sigma,
                    )

        return tile_weights

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim > passed_add_embed_dim and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif expected_add_embed_dim < passed_add_embed_dim and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def _generate_cosine_weights(self, tile_width, tile_height, nbatches, device, dtype):
        """
        Generates cosine weights as a PyTorch tensor for blending tiles.

        Args:
            tile_width (int): Width of the tile in pixels.
            tile_height (int): Height of the tile in pixels.
            nbatches (int): Number of batches.
            device (torch.device): Device where the tensor will be allocated (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type of the tensor (e.g., torch.float32).

        Returns:
            torch.Tensor: A tensor containing cosine weights for blending tiles, expanded to match batch and channel dimensions.
        """
        # Convert tile dimensions to latent space
        latent_width = tile_width // 8
        latent_height = tile_height // 8

        # Generate x and y coordinates in latent space
        x = np.arange(0, latent_width)
        y = np.arange(0, latent_height)

        # Calculate midpoints
        midpoint_x = (latent_width - 1) / 2
        midpoint_y = (latent_height - 1) / 2

        # Compute cosine probabilities for x and y
        x_probs = np.cos(np.pi * (x - midpoint_x) / latent_width)
        y_probs = np.cos(np.pi * (y - midpoint_y) / latent_height)

        # Create a 2D weight matrix using the outer product
        weights_np = np.outer(y_probs, x_probs)

        # Convert to a PyTorch tensor with the correct device and dtype
        weights_torch = torch.tensor(weights_np, device=device, dtype=dtype)

        # Expand for batch and channel dimensions
        tile_weights_expanded = torch.tile(weights_torch, (nbatches, self.unet.config.in_channels, 1, 1))

        return tile_weights_expanded

    def _generate_gaussian_weights(self, tile_width, tile_height, nbatches, device, dtype, sigma=0.05):
        """
        Generates Gaussian weights as a PyTorch tensor for blending tiles in latent space.

        Args:
            tile_width (int): Width of the tile in pixels.
            tile_height (int): Height of the tile in pixels.
            nbatches (int): Number of batches.
            device (torch.device): Device where the tensor will be allocated (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type of the tensor (e.g., torch.float32).
            sigma (float, optional): Standard deviation of the Gaussian distribution. Controls the smoothness of the weights. Defaults to 0.05.

        Returns:
            torch.Tensor: A tensor containing Gaussian weights for blending tiles, expanded to match batch and channel dimensions.
        """
        # Convert tile dimensions to latent space
        latent_width = tile_width // 8
        latent_height = tile_height // 8

        # Generate Gaussian weights in latent space
        x = np.linspace(-1, 1, latent_width)
        y = np.linspace(-1, 1, latent_height)
        xx, yy = np.meshgrid(x, y)
        gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Convert to a PyTorch tensor with the correct device and dtype
        weights_torch = torch.tensor(gaussian_weight, device=device, dtype=dtype)

        # Expand for batch and channel dimensions
        weights_expanded = weights_torch.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        weights_expanded = weights_expanded.expand(nbatches, -1, -1, -1)  # Expand to the number of batches

        return weights_expanded

    def _calculate_tile_positions(self, image_dim: int, tile_dim: int, overlap: int) -> List[int]:
        """
        Calculates the starting positions for tiles along one dimension.
        """
        if image_dim <= tile_dim:
            return [0]

        positions = []
        current_pos = 0
        stride = tile_dim - overlap

        while True:
            positions.append(current_pos)
            if current_pos + tile_dim >= image_dim:
                break

            current_pos += stride
            if current_pos > image_dim - tile_dim:
                break

        last_pos = positions[-1]
        if last_pos + tile_dim < image_dim:
            positions.append(image_dim - tile_dim)

        return sorted(set(positions))

    def _call_callback(self, callback_on_step_end, callback_on_step_end_tensor_inputs, i, t, local_scope, advance=1):
        """
        Helper function to safely prepare and execute the user-provided callback.

        Args:
            callback_on_step_end: The callback function itself.
            callback_on_step_end_tensor_inputs: A list of tensor names required by the callback.
            i (int): Current step index.
            t (int): Current timestep.
            local_scope (dict): A dictionary of the caller's local variables (e.g., from locals()).
            advance (int): The amount to advance the progress bar (passed to the callback).

        Returns:
            The output from the callback, or None if no callback is provided.
        """
        if callback_on_step_end is not None:
            callback_kwargs = {"advance": advance}
            for k in callback_on_step_end_tensor_inputs:
                if k in local_scope:
                    callback_kwargs[k] = local_scope[k]
                else:
                    print(f"Warning: Callback input '{k}' not found in the provided scope.")

            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
            return callback_outputs

        return None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def set_interrupt(self, value):
        r"""
        set the _interupt value for cancel process.
        """

        self._interrupt = value

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        strength: float = 0.9999,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_mode: Optional[Union[int, List[int]]] = None,
        original_size: Tuple[int, int] = None,
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        tile_overlap: int = 128,
        tile_size: int = 1024,
        tile_gaussian_sigma: float = 0.05,
        tile_weighting_method: str = "Cosine",
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        execution_device: Optional[torch.device] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`, *optional*):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, if passing latents directly, they will not be encoded again.
            control_image (`PipelineImageInput`, *optional*):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance for Unet.
                If the type is specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also
                be accepted as an image. The dimensions of the output image default to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                init, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            strength (`float`, *optional*, defaults to 0.9999):
                Indicates the extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point, and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum, and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*):
                The height in pixels of the generated image. If not provided, defaults to the height of `control_image`.
            width (`int`, *optional*):
                The width in pixels of the generated image. If not provided, defaults to the width of `control_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages generating
                images closely linked to the text `prompt`, usually at the expense of lower image quality.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original UNet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_mode (`int` or `List[int]`, *optional*):
                The mode of ControlNet guidance. Can be used to specify different behaviors for multiple ControlNets.
            original_size (`Tuple[int, int]`, *optional*):
                If `original_size` is not the same as `target_size`, the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning.
            target_size (`Tuple[int, int]`, *optional*):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified, it will default to `(height, width)`. Part of SDXL's micro-conditioning.
            negative_original_size (`Tuple[int, int]`, *optional*):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning.
            negative_target_size (`Tuple[int, int]`, *optional*):
                To negatively condition the generation process based on a target image resolution. It should be the same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning.
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Used to simulate an aesthetic score of the generated image by influencing the negative text condition.
                Part of SDXL's micro-conditioning.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            tile_overlap (`int`, *optional*, defaults to 128):
                Number of overlapping pixels between tiles.
            tile_size (`int`, *optional*, defaults to 1024):
                Maximum size of a tile in pixels.
            tile_gaussian_sigma (`float`, *optional*, defaults to 0.3):
                Sigma parameter for Gaussian weighting of tiles.
            tile_weighting_method (`str`, *optional*, defaults to "Cosine"):
                Method for weighting tiles. Options: "Cosine" or "Gaussian".
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/):
                `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            execution_device(`torch.device`, *optional*, defaults "None"):
                If defined, it will be used to execute the pipeline components.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`
            containing the output images.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]

        if not isinstance(control_image, list):
            control_image = [control_image]
        else:
            control_image = control_image.copy()

        if control_mode is None or isinstance(control_mode, list) and len(control_mode) == 0:
            raise ValueError("The value for `control_mode` is expected!")

        if not isinstance(control_mode, list):
            control_mode = [control_mode]

        if len(control_image) != len(control_mode):
            raise ValueError("Expected len(control_image) == len(control_mode)")

        num_control_type = controlnet.config.num_control_type

        # 0. Set internal use parameters
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = execution_device or self._execution_device
        self._interrupt = False

        control_type = [0 for _ in range(num_control_type)]
        control_type = torch.Tensor(control_type)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        global_pool_conditions = controlnet.config.global_pool_conditions
        guess_mode = guess_mode or global_pool_conditions
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        negative_original_size = negative_original_size or original_size
        negative_target_size = negative_target_size or target_size

        # 1. Check inputs
        for _image, control_idx in zip(control_image, control_mode):
            control_type[control_idx] = 1
            self.check_inputs(
                prompt,
                height,
                width,
                _image,
                strength,
                num_inference_steps,
                tile_overlap,
                tile_size,
                tile_gaussian_sigma,
                tile_weighting_method,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
            )

        # set batch size
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise TypeError("`prompt` must be a string or a list of strings.")
        if batch_size > 1 and num_images_per_prompt > 1:
            logger.warning("Using batch_size > 1 and num_images_per_prompt > 1 might lead to unexpected behavior.")

        effective_batch_size = batch_size * num_images_per_prompt

        # 2 Prepare tile settings
        # 2.1 Get tile width and tile height size
        tile_row_overlap = tile_overlap
        tile_col_overlap = tile_overlap
        tile_width, tile_height = _adaptive_tile_size((width, height), max_tile_size=tile_size)

        # 2.2 Calculate the number of tiles needed
        y_steps = self._calculate_tile_positions(height, tile_height, tile_row_overlap)
        x_steps = self._calculate_tile_positions(width, tile_width, tile_col_overlap)
        grid_rows = len(y_steps)
        grid_cols = len(x_steps)

        # 3. Encode input prompt
        text_encoder_lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare embeddings for CFG
        if self.do_classifier_free_guidance:
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            prompt_embeds_cfg = prompt_embeds
            pooled_prompt_embeds_cfg = pooled_prompt_embeds

        prompt_embeds_cfg = prompt_embeds_cfg.to(device)
        pooled_prompt_embeds_cfg = pooled_prompt_embeds_cfg.to(device)

        # 5. Prepare added time ids & embeddings (por tile)
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        # 6. Prepare ADM
        crops_coords_top_left = negative_crops_coords_top_left = (tile_width, tile_height)
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        add_time_ids = add_time_ids.repeat(effective_batch_size, 1)
        if self.do_classifier_free_guidance:
            add_neg_time_ids = add_neg_time_ids.repeat(effective_batch_size, 1)
            add_time_ids_cfg = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        else:
            add_time_ids_cfg = add_time_ids

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds_cfg,
            "time_ids": add_time_ids_cfg.to(device),
        }

        # 7. Prepare latent image
        image_tensor = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        # 7.1 Prepare controlnet_conditioning_image
        control_image = self.prepare_control_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=effective_batch_size,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        control_type = control_type.reshape(1, -1).to(device, dtype=controlnet.dtype).repeat(effective_batch_size * 2, 1)

        # 8. Prepare timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        self.scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1].repeat(effective_batch_size)
        self._num_timesteps = len(timesteps)

        # 9. Prepare latent variables
        dtype = prompt_embeds.dtype
        if latents is None:
            latents = self.prepare_latents(
                image_tensor,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                device,
                generator,
                True,
            )

        latents = latents.to(dtype)

        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # 10. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 11. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            controlnet_keep.append(1.0 - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end))

        # 12. Prepare tiles weights and latent overlaps size to denoising process
        tile_weights = self.prepare_tiles_weights(
            grid_rows,
            grid_cols,
            tile_height,
            tile_width,
            tile_overlap,
            height,
            width,
            tile_weighting_method,
            tile_gaussian_sigma,
            prompt_embeds.shape[0],
            device,
            dtype,
        )

        # 13. Forces freeing up resources by moving components to the CPU.
        self.unet.eval()
        if hasattr(self, "_offload_device"):
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
        if hasattr(self.vae, "to"):
            self.vae.encoder.to("cpu")

        release_memory(device)

        # 14. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with torch.inference_mode():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Diffuse each tile
                    noise_preds = []
                    for row_idx, y_start in enumerate(y_steps):
                        noise_preds_row = []
                        for col_idx, x_start in enumerate(x_steps):
                            noise_preds_row.append(None)
                        noise_preds.append(noise_preds_row)

                    num_total_tiles = grid_rows * grid_cols
                    current_tile_idx = 0

                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            current_tile_idx += 1
                            progress_bar.set_postfix_str(f"Tile {current_tile_idx}/{num_total_tiles} ", refresh=True)

                            # update postfix progress
                            self._call_callback(
                                callback_on_step_end,
                                callback_on_step_end_tensor_inputs,
                                i,
                                t,
                                locals(),
                                0,
                            )

                            if self.interrupt:
                                continue

                            lat_row_init, lat_row_end, lat_col_init, lat_col_end = _tile2latent_indices(
                                row,
                                col,
                                tile_width,
                                tile_height,
                                tile_overlap,
                                tile_overlap,
                                width,
                                height,
                            )
                            tile_latents = latents[:, :, lat_row_init:lat_row_end, lat_col_init:lat_col_end]

                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = (
                                torch.cat([tile_latents] * 2) if self.do_classifier_free_guidance else tile_latents  # 1, 4, ...
                            )
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            # controlnet(s) inference
                            if guess_mode and self.do_classifier_free_guidance:
                                # Infer ControlNet only for the conditional batch.
                                control_model_input = tile_latents
                                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                                controlnet_prompt_embeds = prompt_embeds_cfg.chunk(2)[1]
                                controlnet_added_cond_kwargs = {
                                    "text_embeds": added_cond_kwargs["text_embeds"].chunk(2)[1],
                                    "time_ids": added_cond_kwargs["time_ids"].chunk(2)[1],
                                }
                            else:
                                control_model_input = latent_model_input
                                controlnet_prompt_embeds = prompt_embeds_cfg
                                controlnet_added_cond_kwargs = added_cond_kwargs

                            if isinstance(controlnet_keep[i], list):
                                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                            else:
                                controlnet_cond_scale = controlnet_conditioning_scale
                                if isinstance(controlnet_cond_scale, list):
                                    controlnet_cond_scale = controlnet_cond_scale[0]
                                cond_scale = controlnet_cond_scale * controlnet_keep[i]

                            (
                                px_row_init_pixel,
                                px_row_end_pixel,
                                px_col_init_pixel,
                                px_col_end_pixel,
                            ) = _tile2pixel_indices(
                                row,
                                col,
                                tile_width,
                                tile_height,
                                tile_overlap,
                                tile_overlap,
                                width,
                                height,
                            )

                            tile_control_image = control_image[
                                :,
                                :,
                                px_row_init_pixel:px_row_end_pixel,
                                px_col_init_pixel:px_col_end_pixel,
                            ]

                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                control_model_input,
                                t,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=[tile_control_image],
                                control_type=control_type,
                                control_type_idx=control_mode,
                                conditioning_scale=cond_scale,
                                guess_mode=guess_mode,
                                added_cond_kwargs=controlnet_added_cond_kwargs,
                                return_dict=False,
                            )

                            if guess_mode and self.do_classifier_free_guidance:
                                # Inferred ControlNet only for the conditional batch.
                                # To apply the output of ControlNet to both the unconditional and conditional batches,
                                # add 0 to the unconditional batch to keep it unchanged.
                                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                            # predict the noise residual
                            with torch.amp.autocast(device.type, dtype=self.dtype, enabled=self.unet.dtype != self.dtype):
                                noise_pred = self.unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds_cfg,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]

                            # perform guidance
                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred_tile = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                if self.guidance_rescale > 0.0:
                                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                                    noise_pred_tile = rescale_noise_cfg(
                                        noise_pred_tile,
                                        noise_pred_text,
                                        guidance_rescale=self.guidance_rescale,
                                    )
                                noise_preds[row][col] = noise_pred_tile

                    # Stitch noise predictions for all tiles
                    noise_pred = torch.zeros(latents.shape, device=device)
                    contributors = torch.zeros(latents.shape, device=device)

                    # Add each tile contribution to overall latents
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(
                                row,
                                col,
                                tile_width,
                                tile_height,
                                tile_overlap,
                                tile_overlap,
                                width,
                                height,
                            )
                            tile_weights_resized = tile_weights[row, col]

                            noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += noise_preds[row][col] * tile_weights_resized
                            contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights_resized

                    # Average overlapping areas with more than 1 contributor
                    noise_pred /= contributors
                    noise_pred = noise_pred.to(dtype)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_outputs = self._call_callback(
                            callback_on_step_end,
                            callback_on_step_end_tensor_inputs,
                            i,
                            t,
                            locals(),
                            1,
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        pooled_prompt_embeds_cfg = callback_outputs.pop("add_text_embeds", pooled_prompt_embeds_cfg)
                        add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                    # update progress bar
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

        # move to free memory for vae decoding
        self.unet.to("cpu")
        self.controlnet.to("cpu")
        release_memory(device)

        # 15. Post-processing
        if output_type == "latent":
            image = latents
        else:
            image = self.decode_latents(latents)

            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
