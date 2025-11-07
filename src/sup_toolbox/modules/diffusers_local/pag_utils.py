# This pag_utils file is a derivative work.
# It originates from Diffusers (The HuggingFace Team, Apache 2.0 License)
# Modifications and adaptations to this file are Copyright (c) 2025 by DEVAIEXP.

import re
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
)
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PAGMixin:
    r"""Mixin class for [Pertubed Attention Guidance](https://arxiv.org/abs/2403.17377v1)."""

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
    def _rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def _set_pag_attn_processor(self, pag_applied_layers, do_classifier_free_guidance):
        r"""
        Set the attention processor for the PAG layers.
        """
        pag_attn_processors = self._pag_attn_processors
        if pag_attn_processors is None:
            raise ValueError(
                "No PAG attention processors have been set. Set the attention processors by calling `set_pag_applied_layers` and passing the relevant parameters."
            )

        pag_attn_proc = pag_attn_processors[0] if do_classifier_free_guidance else pag_attn_processors[1]

        if hasattr(self, "unet"):
            model: nn.Module = self.unet
        else:
            model: nn.Module = self.transformer

        def is_self_attn(module: nn.Module) -> bool:
            r"""
            Check if the module is self-attention module based on its name.
            """
            return isinstance(module, Attention) and not module.is_cross_attention

        def is_fake_integral_match(layer_id, name):
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        for layer_id in pag_applied_layers:
            # for each PAG layer input, we find corresponding self-attention layers in the unet model
            target_modules = []

            for name, module in model.named_modules():
                # Identify the following simple cases:
                #   (1) Self Attention layer existing
                #   (2) Whether the module name matches pag layer id even partially
                #   (3) Make sure it's not a fake integral match if the layer_id ends with a number
                #       For example, blocks.1, blocks.10 should be differentiable if layer_id="blocks.1"
                if is_self_attn(module) and re.search(layer_id, name) is not None and not is_fake_integral_match(layer_id, name):
                    logger.debug(f"Applying PAG to layer: {name}")
                    target_modules.append(module)

            if len(target_modules) == 0:
                raise ValueError(f"Cannot find PAG layer to set attention processor for: {layer_id}")

            for module in target_modules:
                module.processor = pag_attn_proc

    def _get_pag_scale(self, t):
        r"""
        Get the scale factor for the perturbed attention guidance at timestep `t`.
        """

        if self.do_pag_adaptive_scaling:
            # signal_scale = self.pag_scale - self.pag_adaptive_scale * (1000 - t)
            # if signal_scale < 0:
            signal_scale = self.pag_scale
            if t / self.num_timesteps > self.pag_adaptive_scale:
                signal_scale = 0
            return signal_scale
        else:
            return self.pag_scale

    def _apply_perturbed_attention_guidance(
        self,
        noise_pred,
        do_classifier_free_guidance,
        guidance_scale,
        guidance_rescale,
        t,
        pag_scale,
    ):
        r"""
        Apply perturbed attention guidance to the noise prediction.

        Args:
            noise_pred (torch.Tensor): The noise prediction tensor.
            do_classifier_free_guidance (bool): Whether to apply classifier-free guidance.
            guidance_scale (float): The scale factor for the guidance term.
            t (int): The current time step.

        Returns:
            torch.Tensor: The updated noise prediction tensor after applying perturbed attention guidance.
        """
        # pag_scale = self._get_pag_scale(t)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(2) if noise_pred.shape[0] < 3 else noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) + pag_scale * (noise_pred_text - noise_pred_perturb)
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = self._rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        else:
            noise_pred_text, noise_pred_perturb = noise_pred.chunk(noise_pred.shape[0])
            noise_pred = noise_pred_text + pag_scale * (noise_pred_text - noise_pred_perturb)

        return noise_pred

    def _prepare_perturbed_attention_guidance(self, cond, uncond, do_classifier_free_guidance):
        """
        Prepares the perturbed attention guidance for the PAG model.

        Args:
            cond (torch.Tensor): The conditional input tensor.
            uncond (torch.Tensor): The unconditional input tensor.
            do_classifier_free_guidance (bool): Flag indicating whether to perform classifier-free guidance.

        Returns:
            torch.Tensor: The prepared perturbed attention guidance tensor.
        """

        cond = torch.cat([cond] * 2, dim=0)

        if do_classifier_free_guidance:
            cond = torch.cat([uncond, cond], dim=0)
        return cond

    def set_pag_applied_layers(
        self,
        pag_applied_layers: Union[str, List[str], None],
        pag_attn_processors: Tuple[AttentionProcessor, AttentionProcessor] = (
            PAGCFGIdentitySelfAttnProcessor2_0(),
            PAGIdentitySelfAttnProcessor2_0(),
        ),
    ):
        r"""
        Set the the self-attention layers to apply PAG. Raise ValueError if the input is invalid.

        Args:
            pag_applied_layers (`str` or `List[str]` or `None`):
                One or more strings identifying the layer names, or a simple regex for matching multiple layers, where
                PAG is to be applied. A few ways of expected usage are as follows:
                  - Single layers specified as - "blocks.{layer_index}"
                  - Multiple layers as a list - ["blocks.{layers_index_1}", "blocks.{layer_index_2}", ...]
                  - Multiple layers as a block name - "mid"
                  - Multiple layers as regex - "blocks.({layer_index_1}|{layer_index_2})"
                  - None to disable PAG
            pag_attn_processors:
                (`Tuple[AttentionProcessor, AttentionProcessor]`, defaults to `(PAGCFGIdentitySelfAttnProcessor2_0(),
                PAGIdentitySelfAttnProcessor2_0())`): A tuple of two attention processors. The first attention
                processor is for PAG with Classifier-free guidance enabled (conditional and unconditional). The second
                attention processor is for PAG with CFG disabled (unconditional only).
        """

        if not hasattr(self, "_pag_attn_processors"):
            self._pag_attn_processors = None

        if pag_applied_layers is None:
            self._pag_applied_layers = None
            self._pag_attn_processors = None
            return

        if not isinstance(pag_applied_layers, list):
            pag_applied_layers = [pag_applied_layers]
        if pag_attn_processors is not None:
            if not isinstance(pag_attn_processors, tuple) or len(pag_attn_processors) != 2:
                raise ValueError("Expected a tuple of two attention processors")

        for i in range(len(pag_applied_layers)):
            if not isinstance(pag_applied_layers[i], str):
                raise ValueError(f"Expected either a string or a list of string but got type {type(pag_applied_layers[i])}")

        self._pag_applied_layers = pag_applied_layers
        self._pag_attn_processors = pag_attn_processors

    @property
    def pag_scale(self):
        """
        Get the scale factor for the perturbed attention guidance.
        """
        return self._pag_scale

    @property
    def pag_adaptive_scale(self):
        """
        Get the adaptive scale factor for the perturbed attention guidance.
        """
        return self._pag_adaptive_scale

    @property
    def do_pag_adaptive_scaling(self):
        """
        Check if the adaptive scaling is enabled for the perturbed attention guidance.
        """
        return (
            self._pag_adaptive_scale > 0
            and self._pag_scale > 0
            and hasattr(self, "_pag_applied_layers")
            and self._pag_applied_layers is not None
            and len(self._pag_applied_layers) > 0
        )

    @property
    def do_perturbed_attention_guidance(self) -> bool:
        r"""Check if the perturbed attention guidance is enabled."""
        return self._pag_scale > 0 and hasattr(self, "_pag_applied_layers") and self._pag_applied_layers is not None and len(self._pag_applied_layers) > 0

    @property
    def pag_attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of PAG attention processors: A dictionary contains all PAG attention processors used in the model
            with the key as the name of the layer.
        """

        if self._pag_attn_processors is None:
            return {}

        valid_attn_processors = {x.__class__ for x in self._pag_attn_processors}

        processors = {}
        # We could have iterated through the self.components.items() and checked if a component is
        # `ModelMixin` subclassed but that can include a VAE too.
        if hasattr(self, "unet"):
            denoiser_module = self.unet
        elif hasattr(self, "transformer"):
            denoiser_module = self.transformer
        else:
            raise ValueError("No denoiser module found.")

        for name, proc in denoiser_module.attn_processors.items():
            if proc.__class__ in valid_attn_processors:
                processors[name] = proc

        return processors

    @staticmethod
    def _check_input_pag_applied_layer(layer):
        r"""
        Check if each layer input in `applied_pag_layers` is valid. It should be either one of these 3 formats:
        "{block_type}", "{block_type}.{block_index}", or "{block_type}.{block_index}.{attention_index}". `block_type`
        can be "down", "mid", "up". `block_index` should be in the format of "block_{i}". `attention_index` should be
        in the format of "attentions_{j}".
        """

        layer_splits = layer.split(".")

        if len(layer_splits) > 3:
            raise ValueError(f"pag layer should only contains block_type, block_index and attention_index{layer}.")

        if len(layer_splits) >= 1:
            if layer_splits[0] not in ["down", "mid", "up"]:
                raise ValueError(f"Invalid block_type in pag layer {layer}. Accept 'down', 'mid', 'up', got {layer_splits[0]}")

        if len(layer_splits) >= 2:
            if not layer_splits[1].startswith("block_"):
                raise ValueError(f"Invalid block_index in pag layer: {layer}. Should start with 'block_'")

        if len(layer_splits) == 3:
            if not layer_splits[2].startswith("attentions_"):
                raise ValueError(f"Invalid attention_index in pag layer: {layer}. Should start with 'attentions_'")
