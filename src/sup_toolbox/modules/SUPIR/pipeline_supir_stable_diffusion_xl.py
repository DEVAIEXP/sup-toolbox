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
import gc
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.import_utils import is_invisible_watermark_available
from diffusers.utils.torch_utils import randn_tensor

from ..diffusers_local.pag_utils import PAGMixin
from .models.autoencoderKL_SUPIR import AutoencoderKLSUPIR
from .models.controlnet_SUPIR import SUPIRControlNetModel
from .models.unet_2d_condition_SUPIR import UNet2DConditionModel
from .schedulers.scheduling_euler_discrete_restore import EulerDiscreteRestoreScheduler


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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


class LatentTileAttention:
    """Handles latent tiling with overlapping grids and Gaussian blending."""

    def __init__(self, kernel_size=(128, 128), overlap=0.5, device="cuda"):
        super().__init__()
        self.kernel_size = kernel_size  # Tuple (height, width) in latent space
        self.overlap = overlap
        self.device = device
        self.tile_weights = self._gaussian_weights(kernel_size[1], kernel_size[0]).to(self.device)
        self.idxes = []
        self.original_size = None

    def _gaussian_weights(self, tile_width, tile_height):
        """Generate a Gaussian weight mask for tile contributions.

        Args:
            tile_width (int): Width of the tile.
            tile_height (int): Height of the tile.

        Returns:
            torch.Tensor: Gaussian weight tensor of shape (channels, height, width).
        """
        from numpy import exp, pi, sqrt

        latent_width = tile_width
        latent_height = tile_height
        var = 0.05
        midpoint = (latent_width - 1) / 2
        x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for y in range(latent_height)]
        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=torch.device("cuda")), (4, 1, 1))

    def grids(self, x):
        """Split the input tensor into overlapping grid patches and concatenate them.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Concatenated tensor of all grid patches.
        """
        import math

        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        if h < k1:
            k1 = h
        if w < k2:
            k2 = w
        self.tile_weights = self._gaussian_weights(k2, k1)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        step_j = k2 if num_col == 1 else math.ceil(k2 * self.overlap)
        step_i = k1 if num_row == 1 else math.ceil(k1 * self.overlap)
        parts = []
        idxes = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i : i + k1, j : j + k2])
                idxes.append({"i": i, "j": j})
                j = j + step_j
            i = i + step_i
        self.idxes = idxes
        return torch.cat(parts, dim=0)

    def grids_inverse(self, outs):
        """Reconstruct the original tensor from processed grid patches with overlap blending."""
        if self.original_size is None or not self.idxes:
            raise RuntimeError("grids() must be called before grids_inverse()")

        b, c, h, w = self.original_size
        k1, k2 = self.kernel_size
        k1 = min(k1, h)
        k2 = min(k2, w)

        preds = torch.zeros(self.original_size, dtype=torch.float32, device=outs.device)
        count_mt = torch.zeros(self.original_size, dtype=torch.float32, device=outs.device)

        current_k1 = outs.shape[2]
        current_k2 = outs.shape[3]
        tile_weights_ = self.tile_weights.to(dtype=torch.float32, device=outs.device)
        if current_k1 != tile_weights_.shape[1] or current_k2 != tile_weights_.shape[2]:
            logger.warning(f"Inverse tile size mismatch. Recomputing weights for {outs.shape[2:]}")
            tile_weights_ = self._gaussian_weights(current_k2, current_k1).to(dtype=torch.float32, device=outs.device)

        if outs.shape[0] != len(self.idxes):
            raise ValueError(f"Number of tiles in input ({outs.shape[0]}) does not match stored coordinates ({len(self.idxes)}).")

        outs_float32 = outs.to(torch.float32)
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            preds[:, :, i : i + current_k1, j : j + current_k2] += outs_float32[cnt : cnt + 1] * tile_weights_
            count_mt[:, :, i : i + current_k1, j : j + current_k2] += tile_weights_

        min_weight_sum = 1e-6
        count_mt = torch.clamp(count_mt, min=min_weight_sum)

        result_float32 = preds / count_mt
        result = result_float32.to(outs.dtype)

        self.original_size = None
        self.idxes = []
        del outs, outs_float32, preds, count_mt, tile_weights_
        release_memory(self.device)

        return result


class LongPromptWeight(object):
    """
    Copied from https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion_xl.py
    """

    def __init__(self) -> None:
        pass

    def parse_prompt_attention(self, text):
        """
        Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
        Accepted tokens are:
        (abc) - increases attention to abc by a multiplier of 1.1
        (abc:3.12) - increases attention to abc by a multiplier of 3.12
        [abc] - decreases attention to abc by a multiplier of 1.1
        \\( - literal character '('
        \\[ - literal character '['
        \\) - literal character ')'
        \\] - literal character ']'
        \\ - literal character '\'
        anything else - just text

        >>> parse_prompt_attention('normal text')
        [['normal text', 1.0]]
        >>> parse_prompt_attention('an (important) word')
        [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
        >>> parse_prompt_attention('(unbalanced')
        [['unbalanced', 1.1]]
        >>> parse_prompt_attention('\\(literal\\]')
        [['(literal]', 1.0]]
        >>> parse_prompt_attention('(unnecessary)(parens)')
        [['unnecessaryparens', 1.1]]
        >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
        [['a ', 1.0],
        ['house', 1.5730000000000004],
        [' ', 1.1],
        ['on', 1.0],
        [' a ', 1.1],
        ['hill', 0.55],
        [', sun, ', 1.1],
        ['sky', 1.4641000000000006],
        ['.', 1.1]]
        """
        import re

        re_attention = re.compile(
            r"""
                \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
                \)|]|[^\\()\[\]:]+|:
            """,
            re.X,
        )

        re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

        res = []
        round_brackets = []
        square_brackets = []

        round_bracket_multiplier = 1.1
        square_bracket_multiplier = 1 / 1.1

        def multiply_range(start_position, multiplier):
            for p in range(start_position, len(res)):
                res[p][1] *= multiplier

        for m in re_attention.finditer(text):
            text = m.group(0)
            weight = m.group(1)

            if text.startswith("\\"):
                res.append([text[1:], 1.0])
            elif text == "(":
                round_brackets.append(len(res))
            elif text == "[":
                square_brackets.append(len(res))
            elif weight is not None and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), float(weight))
            elif text == ")" and len(round_brackets) > 0:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif text == "]" and len(square_brackets) > 0:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                parts = re.split(re_break, text)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    res.append([part, 1.0])

        for pos in round_brackets:
            multiply_range(pos, round_bracket_multiplier)

        for pos in square_brackets:
            multiply_range(pos, square_bracket_multiplier)

        if len(res) == 0:
            res = [["", 1.0]]

        # merge runs of identical weights
        i = 0
        while i + 1 < len(res):
            if res[i][1] == res[i + 1][1]:
                res[i][0] += res[i + 1][0]
                res.pop(i + 1)
            else:
                i += 1

        return res

    def get_prompts_tokens_with_weights(self, clip_tokenizer: CLIPTokenizer, prompt: str):
        """
        Get prompt token ids and weights, this function works for both prompt and negative prompt

        Args:
            pipe (CLIPTokenizer)
                A CLIPTokenizer
            prompt (str)
                A prompt string with weights

        Returns:
            text_tokens (list)
                A list contains token ids
            text_weight (list)
                A list contains the correspodent weight of token ids

        Example:
            import torch
            from transformers import CLIPTokenizer

            clip_tokenizer = CLIPTokenizer.from_pretrained(
                "stablediffusionapi/deliberate-v2"
                , subfolder = "tokenizer"
                , dtype = torch.float16
            )

            token_id_list, token_weight_list = get_prompts_tokens_with_weights(
                clip_tokenizer = clip_tokenizer
                ,prompt = "a (red:1.5) cat"*70
            )
        """
        texts_and_weights = self.parse_prompt_attention(prompt)
        text_tokens, text_weights = [], []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = clip_tokenizer(word, truncation=False).input_ids[1:-1]  # so that tokenize whatever length prompt
            # the returned token is a 1d list: [320, 1125, 539, 320]

            # merge the new tokens to the all tokens holder: text_tokens
            text_tokens = [*text_tokens, *token]

            # each token chunk will come with one weight, like ['red cat', 2.0]
            # need to expand weight for each token.
            chunk_weights = [weight] * len(token)

            # append the weight back to the weight holder: text_weights
            text_weights = [*text_weights, *chunk_weights]
        return text_tokens, text_weights

    def group_tokens_and_weights(self, token_ids: list, weights: list, pad_last_block=False):
        """
        Produce tokens and weights in groups and pad the missing tokens

        Args:
            token_ids (list)
                The token ids from tokenizer
            weights (list)
                The weights list from function get_prompts_tokens_with_weights
            pad_last_block (bool)
                Control if fill the last token list to 75 tokens with eos
        Returns:
            new_token_ids (2d list)
            new_weights (2d list)

        Example:
            token_groups,weight_groups = group_tokens_and_weights(
                token_ids = token_id_list
                , weights = token_weight_list
            )
        """
        bos, eos = 49406, 49407

        # this will be a 2d list
        new_token_ids = []
        new_weights = []
        while len(token_ids) >= 75:
            # get the first 75 tokens
            head_75_tokens = [token_ids.pop(0) for _ in range(75)]
            head_75_weights = [weights.pop(0) for _ in range(75)]

            # extract token ids and weights
            temp_77_token_ids = [bos] + head_75_tokens + [eos]
            temp_77_weights = [1.0] + head_75_weights + [1.0]

            # add 77 token and weights chunk to the holder list
            new_token_ids.append(temp_77_token_ids)
            new_weights.append(temp_77_weights)

        # padding the left
        if len(token_ids) >= 0:
            padding_len = 75 - len(token_ids) if pad_last_block else 0

            temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
            new_token_ids.append(temp_77_token_ids)

            temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
            new_weights.append(temp_77_weights)

        return new_token_ids, new_weights

    def get_weighted_text_embeddings_sdxl(
        self,
        pipe: StableDiffusionXLPipeline,
        prompt: str = "",
        prompt_2: str = None,
        neg_prompt: str = "",
        neg_prompt_2: str = None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        extra_emb=None,
        extra_emb_alpha=0.6,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        num_images_per_prompt: int = 1,
        device: torch.device = None,
    ):
        """
        Encodes prompts into text embeddings for Stable Diffusion XL, supporting
        weighted prompts (e.g., `(word:1.5)`) and prompts of arbitrary length.

        This method handles the complexities of SDXL's dual-tokenizer and
        dual-text-encoder architecture. It tokenizes and encodes prompts for
        both encoders, applies prompt weights, handles padding, and combines the
        resulting embeddings. It also manages LoRA scaling and optional CLIP skipping.

        If embeddings are already provided (`prompt_embeds`, etc.), this function
        will return them directly, skipping the encoding process.

        Args:
            pipe (`StableDiffusionXLPipeline`):
                The SDXL pipeline instance containing the tokenizers and text encoders.
            prompt (`str`):
                The main positive prompt.
            prompt_2 (`str`, *optional*):
                The secondary positive prompt (for the second text encoder). If provided,
                it will be appended to the main prompt.
            neg_prompt (`str`):
                The main negative prompt.
            neg_prompt_2 (`str`, *optional*):
                The secondary negative prompt. If provided, it will be appended to the
                main negative prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings for the positive prompt. If provided,
                the function will skip the encoding step for the positive prompt.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings for the negative prompt.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings for the positive prompt.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings for the negative prompt.
            extra_emb (`torch.Tensor`, *optional*):
                Extra embeddings (e.g., from IP-Adapters) to concatenate.
            extra_emb_alpha (`float`, *optional*, defaults to 0.6):
                The weight to apply to the `extra_emb`.
            lora_scale (`float`, *optional*):
                A scale factor for LoRA layers.
            clip_skip (`int`, *optional*):
                Number of layers to skip from the end of the CLIP text encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt. Embeddings will be
                duplicated accordingly.
            device (`torch.device`, *optional*):
                The device on which to create the tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing `prompt_embeds`, `negative_prompt_embeds`,
                `pooled_prompt_embeds`, and `negative_pooled_prompt_embeds`.
        """
        # 1. LoRA Scale Management
        # Set the LoRA scale on the pipeline so that the patched LoRA functions
        # within the text encoders can access it during the forward pass.
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

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

        # 2. Early Exit for Pre-computed Embeddings
        # If all necessary embeddings are provided, skip the entire encoding
        # process and return them directly.
        if prompt_embeds is not None and negative_prompt_embeds is not None and pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is not None:
            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

        # 3. Prompt Preparation
        # Consolidate the two-part prompts into single strings for each encoder.
        with torch.inference_mode():
            prompt = prompt[0] if isinstance(prompt, list) else prompt
            if prompt_2:
                prompt = f"{prompt} {prompt_2}"

            if neg_prompt_2:
                neg_prompt = f"{neg_prompt} {neg_prompt_2}"

            prompt_t1 = prompt_t2 = prompt
            neg_prompt_t1 = neg_prompt_t2 = neg_prompt

            # Handle textual inversion tokens if the pipeline supports it.
            if isinstance(pipe, TextualInversionLoaderMixin):
                if pipe.tokenizer is not None:
                    prompt_t1 = pipe.maybe_convert_prompt(prompt_t1, pipe.tokenizer)
                    neg_prompt_t1 = pipe.maybe_convert_prompt(neg_prompt_t1, pipe.tokenizer)

                prompt_t2 = pipe.maybe_convert_prompt(prompt_t2, pipe.tokenizer_2)
                neg_prompt_t2 = pipe.maybe_convert_prompt(neg_prompt_t2, pipe.tokenizer_2)

            # 4. Tokenization and Weight Parsing
            eos = pipe.tokenizer.eos_token_id if pipe.tokenizer is not None else pipe.tokenizer_2.eos_token_id

            embeds = []
            neg_embeds = []

            # Tokenize and parse weights for the first tokenizer (CLIP-L).
            if pipe.tokenizer is not None:
                prompt_tokens, prompt_weights = self.get_prompts_tokens_with_weights(pipe.tokenizer, prompt_t1)
                neg_prompt_tokens, neg_prompt_weights = self.get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt_t1)

                # Pad the shorter prompt to match the length of the longer one.
                prompt_token_len = len(prompt_tokens)
                neg_prompt_token_len = len(neg_prompt_tokens)
                if prompt_token_len > neg_prompt_token_len:
                    neg_prompt_tokens.extend([eos] * (prompt_token_len - neg_prompt_token_len))
                    neg_prompt_weights.extend([1.0] * (prompt_token_len - neg_prompt_token_len))
                else:
                    prompt_tokens.extend([eos] * (neg_prompt_token_len - prompt_token_len))
                    prompt_weights.extend([1.0] * (neg_prompt_token_len - prompt_token_len))

                # Group tokens into chunks of 77 (the max sequence length).
                prompt_token_groups, prompt_weight_groups = self.group_tokens_and_weights(prompt_tokens.copy(), prompt_weights.copy())
                neg_prompt_token_groups, neg_prompt_weight_groups = self.group_tokens_and_weights(neg_prompt_tokens.copy(), neg_prompt_weights.copy())

            # Tokenize and parse weights for the second tokenizer (OpenCLIP-G).
            prompt_tokens_2, prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt_t2)
            neg_prompt_tokens_2, neg_prompt_weights_2 = self.get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt_t2)

            # Pad the shorter prompt to match the length of the longer one.
            prompt_token_len_2 = len(prompt_tokens_2)
            neg_prompt_token_len_2 = len(neg_prompt_tokens_2)
            if prompt_token_len_2 > neg_prompt_token_len_2:
                neg_prompt_tokens_2.extend([eos] * (prompt_token_len_2 - neg_prompt_token_len_2))
                neg_prompt_weights_2.extend([1.0] * (prompt_token_len_2 - neg_prompt_token_len_2))
            else:
                prompt_tokens_2.extend([eos] * (neg_prompt_token_len_2 - prompt_token_len_2))
                prompt_weights_2.extend([1.0] * (neg_prompt_token_len_2 - prompt_token_len_2))

            # Group tokens into chunks of 77.
            prompt_token_groups_2, prompt_weight_groups_2 = self.group_tokens_and_weights(prompt_tokens_2.copy(), prompt_weights_2.copy())
            neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = self.group_tokens_and_weights(neg_prompt_tokens_2.copy(), neg_prompt_weights_2.copy())

            # 5. Encoding and Weight Application
            # This loop processes each chunk of 77 tokens to handle arbitrarily long prompts.
            token_groups = prompt_token_groups if pipe.text_encoder is not None else prompt_token_groups_2
            weight_groups = prompt_weight_groups if pipe.text_encoder is not None else prompt_weight_groups_2

            for i in range(len(token_groups)):
                # 5a. Process Positive Prompts
                token_tensor = torch.tensor([token_groups[i]], dtype=torch.long, device=device)
                weight_tensor = torch.tensor(weight_groups[i], dtype=torch.float16, device=device)
                if pipe.text_encoder is not None:
                    token_tensor_2 = torch.tensor([prompt_token_groups_2[i]], dtype=torch.long, device=device)

                # Get raw embeddings from the text encoders.
                if pipe.text_encoder is not None:
                    prompt_embeds_1 = pipe.text_encoder(token_tensor.to(device), output_hidden_states=True)
                    prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2.to(device), output_hidden_states=True)
                    pooled_prompt_embeds = prompt_embeds_2[0]
                else:  # Handle cases with only one encoder
                    prompt_embeds_1 = pipe.text_encoder_2(token_tensor.to(device), output_hidden_states=True)
                    pooled_prompt_embeds = prompt_embeds_1[0]

                # Apply CLIP skip.
                if clip_skip is None:
                    prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
                    if pipe.text_encoder is not None:
                        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
                else:
                    prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-(clip_skip + 2)]
                    if pipe.text_encoder is not None:
                        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-(clip_skip + 2)]

                # Concatenate the embeddings from both encoders.
                if pipe.text_encoder is not None:
                    prompt_embeds_list = [
                        prompt_embeds_1_hidden_states,
                        prompt_embeds_2_hidden_states,
                    ]
                else:
                    prompt_embeds_list = [prompt_embeds_1_hidden_states]
                token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

                # Apply parsed weights to the token embeddings.
                for j in range(len(weight_tensor)):
                    if weight_tensor[j] != 1.0:
                        token_embedding[j] = token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]

                embeds.append(token_embedding.unsqueeze(0))

                # 5b. Process Negative Prompts
                if pipe.tokenizer is not None:
                    neg_token_tensor = torch.tensor([neg_prompt_token_groups[i]], dtype=torch.long, device=device)
                    neg_weight_tensor = torch.tensor(neg_prompt_weight_groups[i], dtype=torch.float16, device=device)
                else:  # Handle cases with only one tokenizer
                    neg_token_tensor = torch.tensor([neg_prompt_token_groups_2[i]], dtype=torch.long, device=device)
                    neg_weight_tensor = torch.tensor(neg_prompt_weight_groups_2[i], dtype=torch.float16, device=device)
                neg_token_tensor_2 = torch.tensor([neg_prompt_token_groups_2[i]], dtype=torch.long, device=device)

                # Get raw embeddings for the negative prompt.
                if pipe.text_encoder is not None:
                    neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor.to(device), output_hidden_states=True)
                    neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2.to(device), output_hidden_states=True)
                    negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]
                    neg_prompt_embeds_list = [
                        neg_prompt_embeds_1.hidden_states[-2],
                        neg_prompt_embeds_2.hidden_states[-2],
                    ]
                else:
                    neg_prompt_embeds_1 = pipe.text_encoder_2(neg_token_tensor.to(device), output_hidden_states=True)
                    negative_pooled_prompt_embeds = neg_prompt_embeds_1[0]
                    neg_prompt_embeds_list = [neg_prompt_embeds_1.hidden_states[-2]]

                neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)

                # Apply parsed weights to the negative token embeddings.
                for z in range(len(neg_weight_tensor)):
                    if neg_weight_tensor[z] != 1.0:
                        neg_token_embedding[z] = neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]

                neg_embeds.append(neg_token_embedding.unsqueeze(0))

            # 6. Final Processing and Duplication
            # Concatenate the chunks back into a single tensor for each prompt.
            prompt_embeds = torch.cat(embeds, dim=1)
            negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

            # Duplicate embeddings for each image to be generated per prompt.
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, -1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, -1)

            # Apply any extra embeddings (e.g., from IP-Adapter).
            if extra_emb is not None:
                extra_emb = extra_emb.to(prompt_embeds.device, dtype=prompt_embeds.dtype) * extra_emb_alpha
                prompt_embeds = torch.cat([prompt_embeds, extra_emb], 1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, torch.zeros_like(extra_emb)], 1)
                print(f"fix prompt_embeds, extra_emb_alpha={extra_emb_alpha}")

        # 7. LoRA Scale Cleanup
        # Restore the original LoRA scale to prevent side effects on subsequent calls.
        if pipe.text_encoder is not None:
            if isinstance(pipe, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(pipe.text_encoder, lora_scale)

        if pipe.text_encoder_2 is not None:
            if isinstance(pipe, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(pipe.text_encoder_2, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )


@dataclass
class InjectionScaleConfig:
    """Configuration for a single dynamic injection scale."""

    scale_end: float = 1.0
    linear: bool = False
    scale_start: float = 0.0
    reverse: bool = False


@dataclass
class InjectionConfigs:
    """
    A comprehensive configuration for all SFT and Cross-Attention injection scales,
    allowing for fine-grained dynamic control over each injection point.
    """

    sft_post_mid: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_0_stage0: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_0_stage1: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_0_stage2: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_1_stage0: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_1_stage1: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_1_stage2: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_2_stage0: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_2_stage1: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    sft_up_block_2_stage2: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)

    cross_up_block_0_stage1: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    cross_up_block_0_stage2: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    cross_up_block_1_stage1: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)
    cross_up_block_1_stage2: InjectionScaleConfig = field(default_factory=InjectionScaleConfig)

    def to_dict(self) -> Dict[str, Dict[str, any]]:
        """Converts the dataclass instance to a dictionary suitable for the pipeline."""
        from dataclasses import asdict

        return asdict(self)


@dataclass
class InjectionFlags:
    """
    A set of boolean flags to enable or disable individual SFT and Cross-Attention
    injections at runtime.
    """

    # SFT Flags
    sft_post_mid_active: bool = True
    sft_up_block_0_stage0_active: bool = True
    sft_up_block_0_stage1_active: bool = True
    sft_up_block_0_stage2_active: bool = True
    sft_up_block_1_stage0_active: bool = True
    sft_up_block_1_stage1_active: bool = True
    sft_up_block_1_stage2_active: bool = True
    sft_up_block_2_stage0_active: bool = True
    sft_up_block_2_stage1_active: bool = True
    sft_up_block_2_stage2_active: bool = True

    # Cross-Attention Flags
    cross_up_block_0_stage1_active: bool = False
    cross_up_block_0_stage2_active: bool = True
    cross_up_block_1_stage1_active: bool = False
    cross_up_block_1_stage2_active: bool = True

    def to_dict(self) -> Dict[str, bool]:
        """Converts the dataclass instance to a dictionary."""
        from dataclasses import asdict

        return asdict(self)


class SUPIRStableDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionXLLoraLoaderMixin,
    FromSingleFileMixin,
    IPAdapterMixin,
    PAGMixin,
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->denoise_encoder->vae->unet"

    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "unet",
        "controlnet",
    ]

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
        "image",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        denoise_encoder: AutoencoderKLSUPIR,
        controlnet: SUPIRControlNetModel,
        scheduler: EulerDiscreteRestoreScheduler,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet,
            denoise_encoder=denoise_encoder,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self._denoise_encoder_dtype = torch.float32
        self.default_sample_size = self.unet.config.sample_size if unet is not None else 128
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
        self.prompt_proj = None
        self.lpw = LongPromptWeight()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.torch_dtype = self.controlnet.dtype  # use dtype from controlnet

    def set_pag_layers(self, pag_applied_layers: Union[str, List[str]] = "mid"):  # ["mid"],["down.block_1"],["up.block_0.attentions_0"]
        if hasattr(self, "original_attn_proc") and len(self.original_attn_proc) > 0:  # rollback original attn processor
            self.unet.set_attn_processor(self.original_attn_proc)
        self.set_pag_applied_layers(pag_applied_layers)

    def enable_vae_tiling(self):
        """Enables tiling in the denoise_encoder and the original VAE."""
        self.vae.enable_tiling()
        self.denoise_encoder.enable_tiling()

    def disable_vae_tiling(self):
        """Disables tiling in denoise_encoder."""
        self.vae.disable_tiling()
        self.denoise_encoder.disable_tiling()

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        needs_upcasting = self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False) and hasattr(self.vae, "to")
        original_vae_dtype = self.vae.dtype

        if hasattr(self.vae, "device") and self.vae.device.type != latents.device.type:
            self.vae.decoder.to(latents.device)

        with torch.inference_mode():
            if needs_upcasting:
                logger.info(f"Upcasting VAE decoder to {self._denoise_encoder_dtype} for decoding...")
                self.vae.to(dtype=self._denoise_encoder_dtype)
                latents = latents.to(self._denoise_encoder_dtype)

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

    def encode_image_latents(
        self,
        image: torch.Tensor,
        timestep: Optional[torch.Tensor],
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
        use_denoise_encoder: bool,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        add_noise=False,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Encodes the preprocessed image into latents using the appropriate encoder."""

        if use_denoise_encoder:
            encoder_module = self.denoise_encoder
            scale_factor = encoder_module.scale_factor
            force_cast = False
        else:
            encoder_module = self.vae
            scale_factor = self.vae.config.scaling_factor
            force_cast = getattr(self.vae.config, "force_upcast", False)

        # Get initial latents from the core encoding function
        # Note: 'image' here is the preprocessed image tensor from image_processor
        # Its batch size is `effective_batch_size`
        image_latents = self._encode_vae_image(
            image_tensor=image,
            encoder_module=encoder_module,
            scale_factor=scale_factor,
            device=device,
            dtype=encoder_module.dtype,  # Pass encoder's target dtype
            force_vae_upcast=force_cast,
            generator=generator,
        )

        # Handle num_images_per_prompt if a single latent image was input
        # If we have 1 latent but want multiple versions per prompt, repeat the latent.
        if image_latents.shape[0] == 1 and num_images_per_prompt > 1:
            image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

        if image_latents.shape[0] < batch_size:
            # We have fewer prepared latents than the target batch_size.
            # This typically happens if fewer prompts were given than (batch_size * num_images_per_prompt)
            # OR if 1 image was given, and batch_size is (1_prompt * num_images_per_prompt).
            # We need to duplicate the existing latents to match the target batch_size.
            if batch_size % image_latents.shape[0] == 0:
                deprecation_message = (
                    f"You have passed {batch_size} total samples to generate (e.g., via prompts * num_images_per_prompt), "
                    f"but only {image_latents.shape[0]} initial latent samples are available after processing input images "
                    f"and num_images_per_prompt. Initial latents are now being duplicated to match the target batch size. "
                    f"Ensure your input image batch size and num_images_per_prompt align with the number of prompts if this is not intended."
                )
                # Using a generic key for deprecate as the exact source of 'batch_size' isn't solely 'len(prompt)' anymore
                deprecate("latent_batch_size_mismatch", "1.0.0", deprecation_message, standard_warn=False)

                multiples = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * multiples, dim=0)
            else:
                # This case should ideally be rare if num_images_per_prompt logic is handled correctly
                # with the number of original prompts to calculate 'batch_size'.
                raise ValueError(
                    f"Target batch_size {batch_size} is not a multiple of the number of available "
                    f"latents {image_latents.shape[0]} after considering num_images_per_prompt. Cannot duplicate evenly."
                )
        elif image_latents.shape[0] > batch_size:
            # We have more prepared latents than the target batch_size.
            # This could happen if more input images were provided than (prompts * num_images_per_prompt).
            # We should use only the first 'batch_size' latents to match the prompts.
            logger.warning(
                f"More initial latents ({image_latents.shape[0]}) are available than the target batch_size ({batch_size}). "
                f"Using the first {batch_size} latents to match the number of generation instances."
            )
            image_latents = image_latents[:batch_size]

        if add_noise:
            shape = image_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=image_latents.dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            latents = image_latents

        if return_noise or return_image_latents:
            outputs = (latents,)
            if return_noise:
                outputs += (noise,)

            if return_image_latents:
                outputs += (image_latents,)
        else:
            outputs = latents

        return outputs

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
                    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

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
                        uncond_input.input_ids.to(device),
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

    # Copied and adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, z_center, restoration_scale, s_churn, s_noise):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature

        extra_step_kwargs = {}
        accepts_z_center = "z_center" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_z_center:
            extra_step_kwargs["z_center"] = z_center

        accepts_s_churn = "s_churn" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_s_churn:
            extra_step_kwargs["s_churn"] = s_churn

        accepts_s_noise = "s_noise" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_s_noise:
            extra_step_kwargs["s_noise"] = s_noise

        accepts_restoration_scale = "restoration_scale" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_restoration_scale:
            extra_step_kwargs["restoration_scale"] = restoration_scale

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        image,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if image is None:
            raise ValueError("`image` must be provided!")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.")

        if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.")
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)

            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)

            return timesteps, num_inference_steps - t_start

        else:
            # Strength is irrelevant if we directly request a timestep to start at;
            # that is, strength is determined by the denoising_start instead.
            discrete_timestep_cutoff = int(round(self.scheduler.config.num_train_timesteps - (denoising_start * self.scheduler.config.num_train_timesteps)))

            num_inference_steps = (self.scheduler.timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            t_start = len(self.scheduler.timesteps) - num_inference_steps
            timesteps = self.scheduler.timesteps[t_start:]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start)
            return timesteps, num_inference_steps

    def _encode_vae_image(
        self,
        image_tensor: torch.Tensor,
        encoder_module: Union[AutoencoderKL, AutoencoderKLSUPIR],
        scale_factor: float,
        device: torch.device,
        dtype: torch.dtype,
        force_vae_upcast: bool,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        """
        Core VAE encoding logic: encodes an image tensor into latents and applies scaling_factor.
        Handles potential upcasting for VAE.
        """
        original_input_dtype = image_tensor.dtype
        target_encoder_dtype = encoder_module.dtype

        # Ensure encoder is on the correct device
        if hasattr(encoder_module, "device") and encoder_module.device.type != device.type:
            encoder_module.encoder.to(device)

        needs_upcasting = force_vae_upcast and image_tensor.dtype != torch.float32 and hasattr(encoder_module, "to")

        image_tensor = image_tensor.to(device=device)

        if needs_upcasting:
            logger.info(f"VAE encoder requires float32. Upcasting image and VAE from {original_input_dtype}...")
            image_tensor_for_encode = image_tensor.to(torch.float32)
            encoder_module.to(dtype=torch.float32)
        else:
            image_tensor_for_encode = image_tensor.to(dtype=target_encoder_dtype)

        with torch.inference_mode():
            if isinstance(generator, list) and image_tensor_for_encode.shape[0] > 1:
                encoded_latents_list = [
                    retrieve_latents(
                        encoder_module.encode(image_tensor_for_encode[i : i + 1]),
                        generator=generator[i],
                    )
                    for i in range(image_tensor_for_encode.shape[0])
                ]
                encoded_latents = torch.cat(encoded_latents_list, dim=0)
            else:
                encoded_latents = retrieve_latents(encoder_module.encode(image_tensor_for_encode), generator=generator)

        # Cast VAE back if it was upcasted
        if needs_upcasting:
            encoder_module.to(dtype=target_encoder_dtype)

        # Apply scaling factor
        scaled_latents = encoded_latents * scale_factor

        return scaled_latents.to(dtype=original_input_dtype)

    def _get_aug_emb(self, time_ids, text_embeds):
        time_embeds = self.unet.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        return add_embeds

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

    def _calculate_dynamic_scale(self, config: Dict[str, Any], interpolation_factor: float) -> float:
        """Calculates a single dynamic scale based on its config and interpolation_factor."""

        scale_end = config.get("scale_end", 1.0)
        is_linear = config.get("linear", False)

        if not is_linear:
            return scale_end

        scale_start = config.get("scale_start", 0.0)
        is_reverse = config.get("reverse", False)

        if is_reverse:
            dynamic_scale = scale_start + (scale_end - scale_start) * (1.0 - interpolation_factor)
            dynamic_scale = round(scale_end + (scale_start - scale_end) * interpolation_factor, 2)
            dynamic_scale = round(np.clip(dynamic_scale, min(scale_start, scale_end), max(scale_start, scale_end)), 2)
        else:
            dynamic_scale = scale_start + (scale_end - scale_start) * interpolation_factor
            dynamic_scale = round(scale_end + (scale_start - scale_end) * interpolation_factor, 2)
            dynamic_scale = round(np.clip(dynamic_scale, min(scale_start, scale_end), max(scale_start, scale_end)), 2)

        return dynamic_scale

    def _get_current_dynamic_injection_scales(
        self, injection_configs: Optional[Dict[str, Dict[str, Any]]], interpolation_factor: float
    ) -> Optional[Dict[str, float]]:
        """
        Calculates the current dynamic scales for SFT/CrossAttn modules
        based on their individual linear configurations.
        """
        if injection_configs is None:
            return None

        current_dynamic_scales = {}
        for module_key_base, config in injection_configs.items():
            output_scale_key = f"{module_key_base}_scale"
            current_dynamic_scales[output_scale_key] = self._calculate_dynamic_scale(config, interpolation_factor)

        return current_dynamic_scales

    def _set_dynamic_scales(
        self,
        guidance_scale,
        controlnet_conditioning_scale,
        use_linear_CFG,
        reverse_linear_CFG,
        guidance_scale_start,
        use_linear_control_scale,
        reverse_linear_control_scale,
        control_scale_start,
        use_linear_PAG,
        reverse_linear_PAG,
        pag_scale,
        pag_scale_start,
        injection_configs: Optional[Dict[str, Dict[str, Any]]],
        interpolation_factor,
    ):
        # Linear CFG
        if use_linear_CFG:
            if reverse_linear_CFG:
                cfg_scale_dynamic = round(
                    guidance_scale + (guidance_scale_start - guidance_scale) * interpolation_factor,
                    2,
                )
                cfg_scale_dynamic = round(
                    np.clip(
                        cfg_scale_dynamic,
                        min(guidance_scale_start, guidance_scale),
                        max(guidance_scale_start, guidance_scale),
                    ),
                    2,
                )
            else:
                cfg_scale_dynamic = round(
                    guidance_scale_start + (guidance_scale - guidance_scale_start) * interpolation_factor,
                    2,
                )
                cfg_scale_dynamic = round(
                    np.clip(
                        cfg_scale_dynamic,
                        min(guidance_scale_start, guidance_scale),
                        max(guidance_scale_start, guidance_scale),
                    ),
                    2,
                )
        else:
            cfg_scale_dynamic = guidance_scale

        # Linear PAG
        if use_linear_PAG and self.do_perturbed_attention_guidance:
            if reverse_linear_PAG:
                pag_scale_dynamic = round(pag_scale + (pag_scale_start - pag_scale) * interpolation_factor, 2)
                pag_scale_dynamic = round(
                    np.clip(
                        pag_scale_dynamic,
                        min(pag_scale_start, pag_scale),
                        max(pag_scale_start, pag_scale),
                    ),
                    2,
                )
            else:
                pag_scale_dynamic = round(pag_scale_start + (pag_scale - pag_scale_start) * interpolation_factor, 2)
                pag_scale_dynamic = round(
                    np.clip(
                        pag_scale_dynamic,
                        min(pag_scale_start, pag_scale),
                        max(pag_scale_start, pag_scale),
                    ),
                    2,
                )
        else:
            pag_scale_dynamic = 0.0 if not self.do_perturbed_attention_guidance else pag_scale

        # Linear Control Scale
        if use_linear_control_scale and injection_configs is None:
            if reverse_linear_control_scale:
                control_scale_dynamic = round(
                    controlnet_conditioning_scale + (control_scale_start - controlnet_conditioning_scale) * interpolation_factor,
                    2,
                )
                control_scale_dynamic = round(
                    np.clip(
                        control_scale_dynamic,
                        min(control_scale_start, controlnet_conditioning_scale),
                        max(control_scale_start, controlnet_conditioning_scale),
                    ),
                    2,
                )
            else:
                control_scale_dynamic = round(
                    control_scale_start + (controlnet_conditioning_scale - control_scale_start) * interpolation_factor,
                    2,
                )
                control_scale_dynamic = round(
                    np.clip(
                        control_scale_dynamic,
                        min(control_scale_start, controlnet_conditioning_scale),
                        max(control_scale_start, controlnet_conditioning_scale),
                    ),
                    2,
                )
        else:
            control_scale_dynamic = controlnet_conditioning_scale

        # Control Scale by module
        current_injection_module_scales = self._get_current_dynamic_injection_scales(injection_configs, interpolation_factor)

        return (
            cfg_scale_dynamic,
            pag_scale_dynamic,
            control_scale_dynamic,
            current_injection_module_scales,
        )

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

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0 and self.unet.config.time_cond_proj_dim is None

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
        strength: float = 1.0,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        start_point: Optional[str] = "noise",
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 9.0,
        guidance_rescale: float = 0.5,
        use_linear_CFG: bool = False,
        reverse_linear_CFG: bool = False,
        guidance_scale_start: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        restoration_scale: float = 4.0,
        s_churn: float = 0.0,
        s_noise: float = 1.003,
        controlnet_conditioning_scale: float = 1.0,
        use_linear_control_scale: bool = False,
        reverse_linear_control_scale: bool = False,
        control_scale_start: float = 0.0,
        use_linear_PAG: bool = False,
        reverse_linear_PAG: bool = False,
        pag_scale: float = 3.0,
        pag_scale_start: float = 1.0,
        pag_adaptive_scale: float = 0.0,
        use_lpw_prompt: bool = False,
        tile_size: int = 1024,
        tile_overlap: float = 0.5,
        original_size: Tuple[int, int] = None,
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        zero_sft_injection_configs: Optional[Union[Dict[str, Dict[str, Any]], InjectionConfigs]] = None,
        zero_sft_injection_flags: Optional[Union[Dict[str, bool], InjectionFlags]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        output_type: str = "pil",
        return_dict: bool = True,
        execution_device: Optional[torch.device] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[StableDiffusionXLPipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The low-quality (LQ) image to be restored or upscaled. This image serves as the primary
                source for the restoration process, providing the content and structure.
            strength (`float`, *optional*, defaults to 1.0):
                Determines the number of denoising steps to run. A value of 1.0 runs for the full
                `num_inference_steps`. Lower values skip initial steps, relevant when starting from a
                partially noised state. For pure restoration, 1.0 is typical.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            start_point (`str`, *optional*, defaults to "noise"):
                The starting point for the generation process. Can be "noise" (starts from pure random noise, guided
                by `zLQ` and `z_center`) or "lr" (starts from a noised version of the low-resolution input image).
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 9.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            guidance_rescale (`float`, *optional*, defaults to 0.5):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            use_linear_CFG (`bool`, *optional*, defaults to `False`):
                Whether to linearly interpolate the `guidance_scale` (CFG) during denoising.
            reverse_linear_CFG (`bool`, *optional*, defaults to `False`):
                If `True`, reverses the direction of the linear CFG interpolation.
            guidance_scale_start (`float`, *optional*, defaults to 1.0):
                The starting value for the CFG scale when `use_linear_CFG` is `True`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
            restoration_scale (`float`, *optional*, defaults to 4.0):
                The scale of the restoration guidance applied in the scheduler. A higher value forces the result
                closer to `z_center`, enhancing fidelity but potentially reducing detail.
            s_churn (`float`, *optional*, defaults to 0.0):
                Corresponds to the `s_churn` parameter in a Karras-style scheduler, adding stochasticity.
                Usually set to 0 for restoration tasks.
            s_noise (`float`, *optional*, defaults to 1.003):
                Corresponds to the `s_noise` parameter in a Karras-style scheduler.
            controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
                The global conditioning scale for the SUPIR ControlNet, controlling the influence of the low-quality
                image structure (`zLQ`) on the UNet's SFT injections. If `use_linear_control_scale` is `True`, this
                value serves as the end point of the interpolation.
            use_linear_control_scale (`bool`, *optional*, defaults to `False`):
                Whether to linearly interpolate the `controlnet_conditioning_scale` during denoising.
            reverse_linear_control_scale (`bool`, *optional*, defaults to `False`):
                If `True`, reverses the direction of the linear SUPIR ControlNet scale interpolation.
            control_scale_start (`float`, *optional*, defaults to 0.0):
                The starting value for the SUPIR ControlNet scale when `use_linear_control_scale` is `True`.
            use_linear_PAG (`bool`, *optional*, defaults to `False`):
                Whether to linearly interpolate the Perceptual-Attention-Guidance (PAG) scale during denoising.
            reverse_linear_PAG (`bool`, *optional*, defaults to `False`):
                If `True`, reverses the direction of the linear PAG scale interpolation.
            pag_scale (`float`, *optional*, defaults to 3.0):
                The scale for Perceptual-Attention-Guidance (PAG). Used as the end point for linear scaling or as
                the fixed scale.
            pag_scale_start (`float`, *optional*, defaults to 1.0):
                The starting scale for PAG when `use_linear_PAG` is True.
            pag_adaptive_scale (`float`, *optional*, defaults to 0.0):
                Adaptive scale factor for PAG, if implemented.
            use_lpw_prompt (`bool`, *optional*, defaults to `False`):
                Whether to use the Long Prompt Weighting (LPW) system for handling prompts longer than the
                tokenizer's limit.
            tile_size (`int`, *optional*, defaults to 1024):
                The size (in pixels) of the square tiles to use when latent tiling is enabled.
            tile_overlap (`float`, *optional*, defaults to 0.5):
                The overlap fraction between adjacent tiles (e.g., 0.5 for 50% overlap).
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            zero_sft_injection_configs (`dict` or `InjectionConfigs`, *optional*):
                A dictionary or an `InjectionConfigs` object to configure individual SFT/CrossAttn injection
                scales dynamically. If a dictionary, keys are module names (e.g., "sft_postmid") and values are
                sub-dictionaries with keys like "scale_end", "linear", "scale_start", "reverse".
                Using the `InjectionConfigs` class is recommended for clarity and safety.
            zero_sft_injection_flags (`dict` or `InjectionFlags`, *optional*):
                A dictionary or an `InjectionFlags` object to enable or disable individual SFT/CrossAttn
                injections at runtime. Keys are module names (e.g., "sft_postmid_active") and values are booleans.
                Using the `InjectionFlags` class is recommended.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            execution_device(`torch.device`, *optional*, defaults "None"):
                If defined, it will be used to execute the pipeline components.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
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

        # 0. Set values for local parameters
        # Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        device = execution_device or self._execution_device
        self._interrupt = False
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale
        original_size = original_size or (height, width)
        target_size = target_size or original_size

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

        # Normalize injection_configs to a dictionary
        injection_configs_dict = None
        if isinstance(zero_sft_injection_configs, InjectionConfigs):
            injection_configs_dict = zero_sft_injection_configs.to_dict()
        elif isinstance(zero_sft_injection_configs, dict):
            injection_configs_dict = zero_sft_injection_configs

        # Normalize injection_flags to a dictionary
        injection_flags_dict = None
        if isinstance(zero_sft_injection_flags, InjectionFlags):
            injection_flags_dict = zero_sft_injection_flags.to_dict()
        elif isinstance(zero_sft_injection_flags, dict):
            injection_flags_dict = zero_sft_injection_flags

        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image,
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # 2. Prepare image latents with denoise encoder
        processed_image_lq = self.image_processor.preprocess(image, height=height, width=width).to(device)
        zLQ = self.encode_image_latents(
            image=processed_image_lq,
            batch_size=effective_batch_size,
            num_images_per_prompt=num_images_per_prompt,
            timestep=None,
            device=device,
            use_denoise_encoder=True,
            return_image_latents=False,
            return_noise=False,
        )

        # 2.1 Calculate z_center (z_stage1 in SGM SUPIR)
        # Decode zLQ [-1, 1]
        z_center = None
        if restoration_scale > 0:
            x_stage1_rec = self.decode_latents(zLQ)

            # Encode with deftault VAE
            z_center = self.encode_image_latents(
                image=x_stage1_rec,
                batch_size=effective_batch_size,
                num_images_per_prompt=num_images_per_prompt,
                timestep=None,
                device=device,
                use_denoise_encoder=False,
                return_image_latents=False,
                return_noise=False,
            )

        release_memory(device)

        # 3. Encode input prompt
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

        if not use_lpw_prompt:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.lpw.get_weighted_text_embeddings_sdxl(
                pipe=self,
                prompt=prompt,
                neg_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=self.clip_skip,
                device=device,
            )

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        # Apply CFG or PAG to embeddings
        if self.do_perturbed_attention_guidance:
            prompt_embeds = self._prepare_perturbed_attention_guidance(prompt_embeds, negative_prompt_embeds, self.do_classifier_free_guidance)
            pooled_prompt_embeds = self._prepare_perturbed_attention_guidance(
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                self.do_classifier_free_guidance,
            )
        elif self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=None,
        )
        latent_timestep = timesteps[:1].repeat(effective_batch_size)

        release_memory(device)
        # 5. Prepare latents
        if start_point == "lr":
            latents_outputs = self.encode_image_latents(
                processed_image_lq,
                latent_timestep,
                batch_size,
                device=device,
                use_denoise_encoder=False,
                generator=generator,
                num_images_per_prompt=num_images_per_prompt,
                add_noise=True,
                return_image_latents=False,
                return_noise=False,
            )
            latents = latents_outputs
        else:
            shape = zLQ.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=self.torch_dtype)

            if sigmas is not None and len(sigmas) > 0:
                initial_sigma = sigmas[0].expand(shape[0])
                latents = noise * initial_sigma[:, None, None, None]
            elif hasattr(self.scheduler, "init_noise_sigma"):
                latents = noise * self.scheduler.init_noise_sigma
            else:
                latents = noise
                logger.warning("Initial noise sigma not found in scheduler and sigmas not provided. Using unscaled noise.")

        latents = latents.to(self.torch_dtype)

        del processed_image_lq

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, z_center, restoration_scale, s_churn, s_noise)

        # 7. Prepare tile settings if enabled
        latents_tiles_batch = None
        zLQ_tiles_batch = None
        z_center_tiles_batch = None
        tiler_latents = None
        tiler_zLQ = None
        tiler_zcenter = None
        num_tiles = 1  # Default for non-tiling
        do_latent_tiling = height > tile_size or width > tile_size

        if do_latent_tiling:
            logger.info(f"Latent Tiling Enabled: Tile Size={tile_size}px, Overlap={tile_overlap}")
            latent_tile_h = tile_size // self.vae_scale_factor
            latent_tile_w = tile_size // self.vae_scale_factor
            kernel_size = (latent_tile_h, latent_tile_w)
            tiler_latents = LatentTileAttention(kernel_size=kernel_size, overlap=tile_overlap, device=device)
            tiler_zLQ = LatentTileAttention(kernel_size=kernel_size, overlap=tile_overlap, device=device)

            if restoration_scale > 0:
                tiler_zcenter = LatentTileAttention(kernel_size=kernel_size, overlap=tile_overlap, device=device)
                if tiler_zcenter is not None:
                    z_center_tiles_batch = tiler_zcenter.grids(z_center)

            latents_tiles_batch = tiler_latents.grids(latents)
            zLQ_tiles_batch = tiler_zLQ.grids(zLQ)
            num_tiles = latents_tiles_batch.shape[0]
            logger.info(f"Number of latent tiles created: {num_tiles}")
            scheduler_states = [copy.deepcopy(self.scheduler.__dict__) for _ in range(num_tiles)]

        # 8. Apply PAG attention if enabled
        if self.do_perturbed_attention_guidance:
            self.original_attn_proc = self.unet.attn_processors
            self._set_pag_attn_processor(
                pag_applied_layers=self._pag_applied_layers,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

        # 9. Forces freeing up resources by moving components to the CPU.
        self.unet.eval()
        self.controlnet.eval()
        if hasattr(self, "_offload_device"):
            self.denoise_encoder.to("cpu")
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
        if hasattr(self.vae, "to"):
            self.vae.encoder.to("cpu")

        release_memory(device)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with torch.inference_mode():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    processed_latent_tiles_for_step = []
                    num_steps = len(timesteps)
                    interpolation_factor = i / (num_steps - 1) if num_steps > 1 else 1.0

                    (
                        cfg_scale_dynamic,
                        pag_scale_dynamic,
                        control_scale_dynamic,
                        current_injection_module_scales,
                    ) = self._set_dynamic_scales(
                        guidance_scale,
                        controlnet_conditioning_scale,
                        use_linear_CFG,
                        reverse_linear_CFG,
                        guidance_scale_start,
                        use_linear_control_scale,
                        reverse_linear_control_scale,
                        control_scale_start,
                        use_linear_PAG,
                        reverse_linear_PAG,
                        pag_scale,
                        pag_scale_start,
                        injection_configs_dict,
                        interpolation_factor,
                    )
                    postfix_str = f"[CFG: {cfg_scale_dynamic:.2f} | PAG: {pag_scale_dynamic:.2f} | CTRL: {control_scale_dynamic:.2f}]"
                    if do_latent_tiling:
                        progress_bar.set_postfix_str(f"Tile 1/{num_tiles} " + postfix_str, refresh=True)
                    else:
                        progress_bar.set_postfix_str(postfix_str, refresh=True)

                    # update postfix progress
                    self._call_callback(callback_on_step_end, callback_on_step_end_tensor_inputs, i, t, locals(), 0)

                    for j in range(num_tiles):
                        if do_latent_tiling:
                            progress_bar.set_postfix_str(f"Tile {j + 1}/{num_tiles} " + postfix_str, refresh=True)

                            # update postfix progress
                            self._call_callback(
                                callback_on_step_end,
                                callback_on_step_end_tensor_inputs,
                                i,
                                t,
                                locals(),
                                0,
                            )

                            self.scheduler.__dict__.update(scheduler_states[j])
                            tile_coords = tiler_latents.idxes[j]
                            current_latent_tile = latents_tiles_batch[j : j + 1]
                            current_zlq_tile = zLQ_tiles_batch[j : j + 1]
                            current_zcenter_tile = z_center_tiles_batch[j : j + 1] if z_center_tiles_batch is not None else None
                            adjusted_batch_size = current_latent_tile.shape[0]
                        else:
                            tile_coords = {"i": 0, "j": 0}
                            current_latent_tile = latents
                            current_zlq_tile = zLQ
                            current_zcenter_tile = z_center if restoration_scale > 0 else None
                            adjusted_batch_size = effective_batch_size

                        # Prepare ADM by Tile
                        tile_crops_coords = (
                            tile_coords["i"] * self.vae_scale_factor,
                            tile_coords["j"] * self.vae_scale_factor,
                        )
                        if self.do_perturbed_attention_guidance or self.do_classifier_free_guidance:
                            adjusted_batch_size = prompt_embeds.shape[0] // current_latent_tile.shape[0]

                        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                            original_size,
                            tile_crops_coords,
                            target_size,
                            aesthetic_score,
                            negative_aesthetic_score,
                            negative_original_size,
                            tile_crops_coords,
                            negative_target_size,
                            dtype=current_latent_tile.dtype,
                            text_encoder_projection_dim=text_encoder_projection_dim,
                        )

                        if self.do_perturbed_attention_guidance:
                            add_time_ids = self._prepare_perturbed_attention_guidance(add_time_ids, add_neg_time_ids, self.do_classifier_free_guidance)
                        elif self.do_classifier_free_guidance:
                            # add_neg_time_ids = add_neg_time_ids.repeat(effective_batch_size, 1)
                            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                        add_time_ids = add_time_ids.repeat(effective_batch_size, 1)
                        add_time_ids = add_time_ids.to(prompt_embeds.device)
                        # Prepare vector by tile
                        vector_final_tile = self._get_aug_emb(add_time_ids, pooled_prompt_embeds).to(self.torch_dtype)
                        added_conditions_tile = {"vector": vector_final_tile}

                        # Expand Tile Inputs for CFG or PAG
                        latent_tile_model_input = (
                            torch.cat([current_latent_tile] * adjusted_batch_size)
                            if self.do_classifier_free_guidance or self.do_perturbed_attention_guidance
                            else current_latent_tile
                        )
                        zlq_tile_model_input = (
                            torch.cat([current_zlq_tile] * adjusted_batch_size)
                            if self.do_classifier_free_guidance or self.do_perturbed_attention_guidance
                            else current_zlq_tile
                        )

                        # Scale model input
                        latent_tile_model_input = self.scheduler.scale_model_input(latent_tile_model_input, t)

                        # SUPIR ControlNet prediction
                        if self.controlnet:
                            control_hs_tile = self.controlnet(
                                sample=latent_tile_model_input,
                                timestep=t,
                                encoder_hidden_states=prompt_embeds,
                                controlnet_cond=zlq_tile_model_input,
                                added_cond_kwargs=added_conditions_tile,
                                return_dict=False,
                            )
                        else:
                            control_hs_tile = None

                        # UNet prediction
                        supir_cross_attention_kwargs = {
                            "control_hs": control_hs_tile,
                            "control_scale_global_dynamic": control_scale_dynamic,
                            "current_injection_module_scales": current_injection_module_scales,
                            "injection_flags": injection_flags_dict,
                            **(cross_attention_kwargs or {}),
                        }

                        with torch.amp.autocast(
                            device.type,
                            dtype=self.torch_dtype,
                            enabled=self.unet.dtype != self.torch_dtype,
                        ):
                            noise_pred_tile = self.unet(
                                sample=latent_tile_model_input,
                                timestep=t,
                                encoder_hidden_states=prompt_embeds,
                                added_cond_kwargs=added_conditions_tile,
                                cross_attention_kwargs=supir_cross_attention_kwargs,
                            )[0]

                        # Apply CFG or PAG
                        if self.do_perturbed_attention_guidance or self.do_classifier_free_guidance:
                            noise_pred_uncond_tile, noise_pred_text_tile = noise_pred_tile.chunk(2)

                        if self.do_perturbed_attention_guidance:
                            noise_pred_final_tile = self._apply_perturbed_attention_guidance(
                                noise_pred_tile,
                                self.do_classifier_free_guidance,
                                self.guidance_scale,
                                self.guidance_rescale,
                                i,
                                pag_scale=pag_scale_dynamic,
                            )
                        elif self.do_classifier_free_guidance:
                            noise_pred_final_tile = noise_pred_uncond_tile + cfg_scale_dynamic * (noise_pred_text_tile - noise_pred_uncond_tile)
                            if self.guidance_rescale > 0.0:
                                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                                noise_pred_final_tile = rescale_noise_cfg(
                                    noise_pred_final_tile,
                                    noise_pred_text_tile,
                                    guidance_rescale=self.guidance_rescale,
                                )
                        else:
                            noise_pred_final_tile = noise_pred_tile

                        extra_step_kwargs_tile = extra_step_kwargs.copy()

                        # Pass the z_center tile IF restore is active AND the tile exists
                        if do_latent_tiling:
                            accepts_z_center = "z_center" in set(inspect.signature(self.scheduler.step).parameters.keys())
                            if accepts_z_center:
                                if restoration_scale > 0 and current_zcenter_tile is not None:
                                    extra_step_kwargs_tile["z_center"] = current_zcenter_tile
                                else:
                                    extra_step_kwargs_tile.pop("z_center", None)

                        # Calculate previous sample
                        scheduler_output_tile = self.scheduler.step(
                            noise_pred_final_tile,
                            t,
                            current_latent_tile,
                            **extra_step_kwargs_tile,
                            return_dict=True,
                        )
                        prev_latent_tile = scheduler_output_tile.prev_sample

                        if prev_latent_tile.dtype != self.torch_dtype:
                            prev_latent_tile = prev_latent_tile.to(self.torch_dtype)

                        if do_latent_tiling:  # get initial scheduler states to next step
                            scheduler_states[j] = copy.deepcopy(self.scheduler.__dict__)
                        processed_latent_tiles_for_step.append(prev_latent_tile)

                    # Rebuild entire latent
                    if do_latent_tiling:
                        processed_tiles_batch = torch.cat(processed_latent_tiles_for_step, dim=0)
                        latents = tiler_latents.grids_inverse(processed_tiles_batch)
                        latents_tiles_batch = tiler_latents.grids(latents)
                        logger.debug(f"Step {i}: Reconstructed latent shape: {latents.shape}")
                    else:
                        latents = processed_latent_tiles_for_step[0]

                    # Callback
                    if callback_on_step_end is not None:
                        callback_outputs = self._call_callback(
                            callback_on_step_end,
                            callback_on_step_end_tensor_inputs,
                            i,
                            t,
                            locals(),
                            1,
                        )
                        new_latents = callback_outputs.pop("latents", latents)
                        # IF callback changes latents, re-tile
                        if do_latent_tiling and (new_latents != latents).any():
                            latents = new_latents
                            latents_tiles_batch = tiler_latents.grids(latents)
                        else:
                            latents = new_latents

                    # Update progress bar
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

        # move to free memory for vae decoding
        self.unet.to("cpu")
        self.controlnet.to("cpu")
        release_memory(device)

        # 11. Post-processing
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

        # Rollback attn processors
        if self.do_perturbed_attention_guidance:
            self.unet.set_attn_processor(self.original_attn_proc)

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
