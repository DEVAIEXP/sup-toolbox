# Copyright 2025 The DEVAIEXP Team. All rights reserved.
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

import dataclasses
import json
import logging
import math
import os
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

# Diffusers imports
from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from diffusers.models.controlnets import ControlNetUnionModel
from diffusers.utils import load_image
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

from sup_toolbox.config import (
    CONTROLNET_TILE_PIPELINE_NAME,
    FAITHDIFF_PIPELINE_NAME,
    MODEL_DEFAULT_CONFIG,
    PAG_LAYERS,
    PIPELINE_PARAM_LABELS,
    RESTORER,
    SUPIR_ADVANCED_LABELS,
    SUPIR_MODELS,
    SUPIR_PIPELINE_NAME,
    UPSCALER,
    UPSCALER_SCALES,
    VALID_CONTROLNETTILE_PARAMS,
    VALID_FAITHDIFF_PARAMS,
    VALID_SUPIR_PARAMS,
    Config,
)
from sup_toolbox.enums import (
    ColorFix,
    ImageSizeFixMode,
    MemoryAttention,
    ModelType,
    QuantizationMethod,
    QuantizationMode,
    RestorerEngine,
    RuntimeDevice,
    UpscalerEngine,
    UpscalingMode,
)
from sup_toolbox.modules.diffusers_local.models.autoencoder_kl_cpu import AutoencoderKL
from sup_toolbox.modules.FaithDiff.models.faithdiff_unet import (
    UNet2DConditionModel as FaithDiffUNet,
)
from sup_toolbox.modules.FaithDiff.pipeline_faithdiff_stable_diffusion_xl import (
    FaithDiffStableDiffusionXLPipeline,
)
from sup_toolbox.modules.IPAdapter.ip_load import IPLoad
from sup_toolbox.modules.MoDControlTile.mod_controlnet_tile_sr_sdxl import (
    StableDiffusionXLControlNetTileSRPipeline,
)
from sup_toolbox.modules.model_manager import ModelManager
from sup_toolbox.modules.SUPIR.models.autoencoderKL_SUPIR import (
    AutoencoderKLSUPIR as SUPIRDenoiseEncoder,
)
from sup_toolbox.modules.SUPIR.models.controlnet_SUPIR import (
    SUPIRControlNetModel as SUPIRControlNet,
)
from sup_toolbox.modules.SUPIR.models.unet_2d_condition_SUPIR import (
    UNet2DConditionModel as SUPIRUNet,
)
from sup_toolbox.modules.SUPIR.pipeline_supir_stable_diffusion_xl import (
    InjectionConfigs,
    InjectionFlags,
    SUPIRStableDiffusionXLPipeline,
)
from sup_toolbox.pipeline_util import (
    MAX_SEED,
    PipelineDeviceManager,
    load_scheduler,
    quantize_FP8,
    quantize_NF4,
    quantize_quanto,
)
from sup_toolbox.tools.groundingdino_sam2.groundingdino_sam2 import GroundingDinoSAM2
from sup_toolbox.tools.model_toolkit.model_toolkit import do_load
from sup_toolbox.utils.colorfix import adain_color_fix, wavelet_color_fix
from sup_toolbox.utils.image import (
    check_image,
    dilate_mask,
    padding_image_size,
    save_image_with_metadata,
)
from sup_toolbox.utils.logging import logger
from sup_toolbox.utils.system import release_memory


class PipelineCancelationRequested(Exception):
    """Exception raised within the SUPToolBoxPipeline to signal graceful termination."""

    pass


class SUPToolBoxPipeline:
    def __init__(
        self,
        config: Config,
        log_callback: Callable = None,
        progress_bar_handler: any = None,
        cancel_event: threading.Event = None,
    ):
        self._log_num_divisor = 160
        self.log_callback: Callable = log_callback
        self.progress_bar_handler = progress_bar_handler
        self.config = config
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.model_manager = ModelManager(config)
        self.image_tools_model_path = os.path.join(
            self.config.models_root_path,
            self.model_manager.get_model_settings(ModelType.ImageTools.value, None, "image-tools", download_in_root=True)["model_path"],
        )
        self.pipeline_managers: Dict[str, PipelineDeviceManager] = {}

        # Tracks the specific checkpoint model loaded for each role
        self.current_prior_restorer_model: Optional[str] = None
        self.current_prior_upscaler_model: Optional[str] = None
        self.current_restorer_supir_model: Optional[str] = None
        self.current_upscaler_supir_model: Optional[str] = None
        self.current_vae_model: Optional[str] = None

        self.current_prior_restorer_base = None
        self.current_prior_upscaler_base = None

        self.active_restorer_engine: Optional[RestorerEngine] = None
        self.active_upscaler_engine: Optional[UpscalerEngine] = None

        self.pipe_reload_is_needed: bool = False
        self.vae_onnx: bool = False
        self.grounding_dino_sam2_model: GroundingDinoSAM2 = None
        self.llava_model: LlavaForConditionalGeneration = None
        self.llava_processor: AutoProcessor = None
        self.images = []
        self.ip_model = None
        self.image_prompt_embeds = None
        self.uncond_image_prompt_embeds = None

    # region public API
    def initialize(self):
        """
        Initializes the pipeline by performing the following steps:
        1. Checks for cancellation requests.
        2. Prints the current parameters.
        3. Loads the specified models (Restorer and Upscaler).
        4. Applies optimizations to the pipeline.
        If a cancellation request is detected, it raises a PipelineCancelationRequested exception.
        If any other exception occurs during initialization, it logs the error and returns False.
        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        self._cancel_checker()

        try:
            self.print_params()
            self.load_models(roles_to_load=["Restorer", "Upscaler"])
            self.apply_optimizations()
        except PipelineCancelationRequested:
            raise
        except Exception as e:
            self._do_logger("Initialization failed", e, level=logging.ERROR, exc_info=True)
            return False

        return True

    def predict(self, metadata=None):
        """
        Predicts the output for the given image or images based on the provided metadata.
        This method checks if an image path or image is provided, creates necessary directories for
        storing restored and upscaled images, and processes each image accordingly. It supports both
        PIL images and NumPy arrays, as well as directories containing image files.
        Args:
            metadata (optional): Additional metadata that may be used during the prediction process.
        Returns:
            tuple: A tuple containing:
                - first_generated_image (Image or None): The first generated image from the processing,
                  or None if no images were processed.
                - bool: True if the processing was successful, False otherwise.
        Raises:
            Exception: If an error occurs during image processing.
        """

        self._cancel_checker()
        self.metadata = metadata

        if not bool(self.config.image_path) or self.config.image_path is None:
            self._do_logger("An image or image path must be provided!", level=logging.ERROR)
            return None, False

        restored_dir = os.path.join(self.config.output_dir, "restored")
        os.makedirs(restored_dir, exist_ok=True)
        upscaled_dir = os.path.join(self.config.output_dir, "upscaled")
        os.makedirs(upscaled_dir, exist_ok=True)

        image_is_pil = isinstance(self.config.image_path, Image.Image)
        image_is_np = isinstance(self.config.image_path, np.ndarray)
        is_loaded_image = image_is_pil or image_is_np
        list_files = []
        is_dir = False
        if is_loaded_image:
            if image_is_np:
                list_files = [Image.fromarray(self.config.image_path)]
            else:
                list_files = [self.config.image_path]
        else:
            is_dir = os.path.isdir(self.config.image_path)
            if is_dir:
                image_path = Path(self.config.image_path)
                ext_filter = ["*.jpg", "*.jpeg", "*.png"]
                for ext in ext_filter:
                    list_files.extend(image_path.glob(ext))
            else:
                list_files = [Path(self.config.image_path)]

        first_generated_image = None
        for idx, file_path_or_pil in enumerate(sorted(list_files)):
            self._cancel_checker()

            try:
                initial_image = load_image(str(file_path_or_pil)) if not is_loaded_image else file_path_or_pil
                prompt = self._get_prompt_for_image(file_path_or_pil, is_dir, is_loaded_image)

                if prompt is None and is_dir:
                    continue

                result_image, status = self._process_single_image(
                    initial_image=initial_image,
                    prompt=prompt,
                    image_index=idx,
                    restored_dir=restored_dir,
                    upscaled_dir=upscaled_dir,
                )
                if idx == 0:
                    first_generated_image = result_image
            except PipelineCancelationRequested as ce:
                raise ce
            except Exception as e:
                self._do_logger(
                    f"Failed to process {file_path_or_pil}",
                    level=logging.ERROR,
                    exc_info=True,
                )
                raise e

        return first_generated_image, True

    def generate_caption(self, image: Image.Image) -> Optional[str]:
        """
        Generates a caption for a given image using the Llava model.
        This method ensures that the Llava model is loaded and available for use.
        It constructs a prompt that includes the image and a predefined question prompt
        from the configuration. The method then processes the image and prompt,
        runs inference using the Llava model, and decodes the generated tokens into
        a human-readable caption.
        Args:
            image (Image.Image): The input image for which the caption is to be generated.
        Returns:
            Optional[str]: The generated caption as a string if successful,
                            or None if the model is not loaded or an error occurs during inference.
        """

        self._cancel_checker()

        loaded = self._load_llava_model()
        if not loaded or self.llava_model is None or self.llava_processor is None:
            self._do_logger("Llava model is not available for caption generation.", level=logging.ERROR)
            return None

        try:
            self._do_logger("--- Generating Caption ---")
            prompt = f"USER: <image>\n{self.config.llava_question_prompt}\nASSISTANT:"
            with torch.inference_mode():
                inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")
                input_token_len = inputs["input_ids"].shape[1]
                output_tokens = self.llava_model.generate(**inputs, max_new_tokens=77, do_sample=False)
                generated_tokens = output_tokens[0, input_token_len:]
                outputs = self.llava_processor.decode(generated_tokens, skip_special_tokens=True)

            return outputs
        except Exception as e:
            self._do_logger("Llava inference failed", e, level=logging.ERROR, exc_info=True)
            return None

    def generate_prompt_mask(self, image_for_detection: Image.Image, mask_prompt="") -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Detects objects in an image and generates processed masks for composition.

        Args:
            image_for_detection: The PIL Image to run detection on.

        Returns:
            A tuple containing:
            - composition_mask (PIL.Image): A feathered mask for pasting the restored area.
            - background_mask (PIL.Image): An inverted mask for preserving the original background.
            Returns (None, None) if no object is detected or an error occurs.
        """
        self._cancel_checker()
        self._do_logger("--- Generating Object Mask ---")

        try:
            self._load_grounding_dino_sam_model()
            if self.grounding_dino_sam2_model is None:
                raise RuntimeError("GroundingDinoSAM2 model is not available.")

            self._do_logger("Detecting mask with prompt", self.config.mask_prompt)
            _, combined_mask_cv, _ = self.grounding_dino_sam2_model(
                input_image=image_for_detection,
                prompt=mask_prompt,
                invert_mask_sequence=False,
                invert_mask=False,
                apply_blur=True,
                blur_ksize=self.config._gaussian_blur_ksize_for_control_mask_fill,
                output_type="cv",
            )

            if combined_mask_cv is None or np.sum(combined_mask_cv) == 0:
                self._do_logger("No objects detected for the given prompt.", level=logging.WARNING)
                return None, None

            # Create the feathered mask for pasting the restored content
            self._do_logger("Processing composition mask...")
            dilated_mask = dilate_mask(combined_mask_cv, self.config._ksize_for_dilate_binary_mask_fill)
            ksize = self.config._gaussian_blur_ksize_for_control_mask_fill
            dilated_mask_blur = cv2.GaussianBlur(dilated_mask, (ksize, ksize), 0)
            composition_mask = Image.fromarray(dilated_mask_blur).convert("L")

            # Create the inverted mask for preserving the original background
            self._do_logger("Processing background mask...")
            dilated_mask_for_invert = dilate_mask(
                combined_mask_cv,
                self.config._ksize_for_dilate_binary_mask_fill_inverted,
            )
            inverted_mask_cv = cv2.bitwise_not(dilated_mask_for_invert)
            background_mask = Image.fromarray(inverted_mask_cv).convert("L")

            return composition_mask, background_mask

        except Exception as e:
            self._do_logger(
                "An error occurred during mask generation.",
                e,
                level=logging.ERROR,
                exc_info=True,
            )
            return None, None

    def print_params(self):
        """
        Logs the current pipeline configuration in a formatted, readable way.
        It uses inclusion sets (VALID_*_PARAMS) to display only the relevant
        parameters for each selected engine (SUPIR, FaithDiff, ControlNetTile).
        """
        if self._cancel_checker():
            return False

        # Internal helper function for detailed logging
        def _log_pipeline_settings(title: str, params_obj, include_params: Set[str]):
            """
            Logs the parameters of a specific pipeline configuration object,
            filtering by a set of valid parameter names and handling complex types.

            Args:
                title (str): The section title for the log output.
                params_obj: The pipeline parameters object (e.g., self.config.restorer_pipeline_params).
                include_params (Set[str]): A set of parameter names to include in the log.
            """
            self._do_logger("-" * self._log_num_divisor)
            self._do_logger(title)
            self._do_logger("-" * self._log_num_divisor)

            for param_name, param_value in params_obj.__dict__.items():
                # Only log parameters that are valid for the current engine.
                if param_name not in include_params:
                    continue

                # Get the friendly label from the mapping, or format the attribute name as a fallback.
                friendly_name = PIPELINE_PARAM_LABELS.get(param_name, param_name.replace("_", " ").title())

                # Special handling for complex SUPIR configuration objects
                if isinstance(param_value, InjectionConfigs):
                    self._do_logger("-" * self._log_num_divisor)
                    self._do_logger(friendly_name)  # Print the main title for this section
                    self._do_logger("-" * self._log_num_divisor)
                    # Get the corresponding flags to check if a stage is active.
                    flags = getattr(params_obj, "zero_sft_injection_flags", None)

                    for stage_name, stage_config in dataclasses.asdict(param_value).items():
                        stage_label = SUPIR_ADVANCED_LABELS.get(stage_name, stage_name)

                        # Check if the stage is active before logging it.
                        flag_name = f"{stage_name.replace('sft_', 'sft_')}_active"
                        is_active = flags.to_dict().get(flag_name, True) if flags else True

                        # Format a concise string with the stage's configuration.
                        config_str = (
                            f"Active: {str(is_active):5},"  # Fixed-width 5 chars, no trailing space
                            f" Control Scale: {stage_config['scale_end']}"
                            f", Use Linear Control: {str(stage_config['linear']):5},"  # Fixed-width 5 chars
                            f" Guidance Scale Start: {stage_config['scale_start']}"
                            f", Reverse Linear Control: {str(stage_config['reverse']):5}"  # Fixed-width 5 chars
                        )
                        # Indent the stage for readability.
                        self._do_logger(f"  - {stage_label:<20}", config_str)
                    continue  # Move to the next main parameter

                elif isinstance(param_value, InjectionFlags):
                    # Skip logging the flags object directly, as its information is used above.
                    continue

                # Skip logging long prompts, None values, empty strings, or other complex objects.
                if param_name in ("prompt", "prompt_2") or isinstance(param_value, (dict, type(None))) or param_value == "":
                    continue

                # If the value is an Enum, print its string value.
                if hasattr(param_value, "value"):
                    param_value = param_value.value

                self._do_logger(friendly_name, param_value)

        # Main Logic of print_params
        self._do_logger("-" * self._log_num_divisor)
        self._do_logger("Pipeline Configuration")
        self._do_logger("-" * self._log_num_divisor)

        # Log general, non-pipeline-specific settings.
        self._do_logger("Execution Device", self.config.device.value)
        self._do_logger("Generator Device", self.config.generator_device.value)
        self._do_logger("Checkpoint Precision Type", self.config.weight_dtype.value)
        self._do_logger("VAE Precision Type", self.config.vae_weight_dtype.value)
        self._do_logger(
            "Memmapping Enabled",
            not self.config.get_bool_value(self.config.disable_mmap),
        )
        self._do_logger(
            "Always Offload Models",
            self.config.get_bool_value(self.config.always_offload_models),
        )
        self._do_logger(
            "Selected VAE Model",
            (self.config.selected_vae_model if bool(self.config.selected_vae_model) else "Default"),
        )
        self._do_logger("Save Image in Upscaling Passes", self.config.save_image_on_upscaling_passes)
        # Map each engine to its corresponding set of valid parameters.
        inclusion_map = {
            RestorerEngine.SUPIR: VALID_SUPIR_PARAMS,
            RestorerEngine.FaithDiff: VALID_FAITHDIFF_PARAMS,
            UpscalerEngine.SUPIR: VALID_SUPIR_PARAMS,
            UpscalerEngine.FaithDiff: VALID_FAITHDIFF_PARAMS,
            UpscalerEngine.ControlNetTile: VALID_CONTROLNETTILE_PARAMS,
        }

        # Log Restorer settings if a restorer engine is active.
        if self.config.restorer_engine != RestorerEngine.Nothing:
            self._do_logger(
                "Selected Restorer Model",
                self.config.selected_restorer_checkpoint_model,
            )
            if self.config.restorer_engine == RestorerEngine.SUPIR:
                self._do_logger(
                    "Selected SUPIR Model",
                    self.config.restorer_pipeline_params.supir_model.value,
                )
            self._do_logger(
                f"Selected {RESTORER} Sampler",
                self.config.selected_restorer_sampler.value,
            )
            self._do_logger(
                "Sampler Scale Linear Expoent",
                self.config.restorer_sampler_config.scale_linear_exponent,
            )
            self._do_logger(
                "Face Restoration",
                self.config.get_bool_value(self.config.restore_face),
            )
            if self.config.restore_face:
                self._do_logger("Face Restoration Prompt Mask", self.config.mask_prompt)
            # Get the correct set of parameters to include.
            params_to_include = inclusion_map.get(self.config.restorer_engine, set())

            _log_pipeline_settings(
                title=f"Restorer Settings ({self.config.restorer_engine.value})",
                params_obj=self.config.restorer_pipeline_params,
                include_params=params_to_include,
            )

        # Log Upscaler settings if an upscaler engine is active.
        if self.config.upscaler_engine != UpscalerEngine.Nothing:
            self._do_logger("-" * self._log_num_divisor)
            self._do_logger(
                "Selected Upscaler Model",
                self.config.selected_upscaler_checkpoint_model,
            )
            if self.config.upscaler_engine == UpscalerEngine.SUPIR:
                self._do_logger(
                    "Selected SUPIR Model",
                    self.config.upscaler_pipeline_params.supir_model.value,
                )
            self._do_logger(
                f"Selected {UPSCALER} Sampler",
                self.config.selected_upscaler_sampler.value,
            )
            self._do_logger(
                "Sampler Scale Linear Expoent",
                self.config.upscaler_sampler_config.scale_linear_exponent,
            )
            # Get the correct set of parameters to include.
            params_to_include = inclusion_map.get(self.config.upscaler_engine, set())

            _log_pipeline_settings(
                title=f"Upscaler Settings ({self.config.upscaler_engine.value})",
                params_obj=self.config.upscaler_pipeline_params,
                include_params=params_to_include,
            )

        if self.config.restorer_engine == RestorerEngine.Nothing and self.config.upscaler_engine == UpscalerEngine.Nothing:
            self._do_logger("No pipeline engine selected.", level=logging.WARNING)
            return False

        self._do_logger("-" * self._log_num_divisor)
        return True

    def load_models(self, roles_to_load: List[str] = ["Restorer", "Upscaler"]) -> bool:
        """
        Loads the necessary models for the specified roles (Restorer and Upscaler) while avoiding redundant loads
        if both roles share the same engine. The function also applies Variational Autoencoder (VAE) and samplers
        only if the pipelines have changed.
        Args:
            roles_to_load (List[str], optional): A list of roles to load. Defaults to ["Restorer", "Upscaler"].
        Returns:
            bool: True if any pipelines were changed during the loading process, False otherwise.
        Notes:
            - The function first cleans up any unused pipelines.
            - It checks if the engines for the Restorer and Upscaler are the same and handles loading accordingly.
            - If the engines are the same, the Upscaler state is updated to reflect the Restorer's state without reloading.
            - If the pipelines have changed or the selected VAE model differs from the current one, VAE is applied to the pipelines.
            - Memory is released at the end of the function to optimize resource usage.
        """

        self._cleanup_unused_pipelines()

        pipelines_were_changed = False

        # Determine if the engines are the same
        restorer_engine = self.config.restorer_engine
        upscaler_engine = self.config.upscaler_engine
        engines_are_the_same = restorer_engine != RestorerEngine.Nothing and restorer_engine.value == upscaler_engine.value

        # Process the Restorer if it's in the list of roles to load
        if "Restorer" in roles_to_load and restorer_engine != RestorerEngine.Nothing:
            if self._load_pipeline_for_role("Restorer"):
                pipelines_were_changed = True

        # Process the Upscaler if it's in the list AND if it is not the same engine as the Restorer
        # (or if the Restorer was not loaded in this call)
        if "Upscaler" in roles_to_load and upscaler_engine != UpscalerEngine.Nothing:
            # Only load the upscaler separately if the engines are different.
            # If they are the same, the Restorer loading already handled it.
            # The check in _process_single_image will force a reload if needed (e.g., supir_model changed).
            if not engines_are_the_same:
                if self._load_pipeline_for_role("Upscaler"):
                    pipelines_were_changed = True
            else:
                # If the engines are the same, just ensure the upscaler state reflects the restorer.
                # The actual reload logic will occur in _process_single_image.
                self.active_upscaler_engine = self.active_restorer_engine
                self.current_prior_upscaler_model = self.current_prior_restorer_model
                self.current_upscaler_supir_model = self.current_restorer_supir_model

        if pipelines_were_changed or self.current_vae_model != self.config.selected_vae_model:
            all_managed_pipes = [mgr.pipe for mgr in self.pipeline_managers.values() if mgr.pipe is not None]
            if all_managed_pipes:
                self._apply_vae_to_pipelines(all_managed_pipes)
                if self.config.restorer_engine == RestorerEngine.FaithDiff:
                    self._check_ipa("Restorer")
                elif self.config.upscaler_engine == UpscalerEngine.FaithDiff:
                    self._check_ipa("Upscaler")
                self.current_vae_model = self.config.selected_vae_model

        release_memory()
        return pipelines_were_changed

    def apply_optimizations(self):
        """
        Apply a set of performance and memory optimizations to all managed pipelines.
        This method orchestrates multiple runtime and model-level optimizations based on
        the current self.config and the set of pipeline managers available in
        self.pipeline_managers. It performs logging and cancellation checks as it goes.
        Behavior and operations performed
        - Global torch backend flags:
            - Sets torch.backends.cudnn.allow_tf32 according to config.allow_cudnn_tf32.
            - Sets torch.backends.cuda.matmul.allow_tf32 according to config.allow_cuda_tf32.
        - Weight dtype:
            - Resolves the target weight dtype via self.config.get_weight_dtype and uses it
              unless overridden by a quantization mode that sets a different internal dtype.
        - For each pipeline manager (iterating over self.pipeline_managers.values()):
            - Runs cancellation checks between long-running operations.
            - Optionally enables xFormers memory-efficient attention if configured and not
              already active on the pipeline's UNet.
            - Applies quantization according to config.quantization_method and
              config.quantization_mode:
                * Layerwise: sets self.config._quantization_dtype to the layerwise dtype
                  (e.g., float8_e4m3fn) and calls quantize_FP8 or quantize_NF4 on the UNet
                  and, when applicable, on text encoders.
                * Quanto: calls quantize_quanto on the UNet with the selected mode.
                * Otherwise: uses the previously resolved weight dtype.
              Logging of quantization method/mode is performed once (for the first pipeline).
            - VAE configuration:
                * Enables/disables VAE tiling according to config.enable_vae_tiling.
                * Enables/disables VAE slicing according to config.enable_vae_slicing.
                * Enables running the VAE on CPU when config.run_vae_on_cpu is set and the
                  VAE implementation supports it.
            - Device / offload handling:
                * If config.enable_cpu_offload is true and the configured device is not CPU,
                  enables model CPU offload for the pipeline.
                * Otherwise, moves active pipelines to the configured device (self.config.device).
            - Saves the manager state (mgr.save_state()) if no saved_state exists.
        - After iterating all pipelines:
            - Logs the effective memory attention configuration.
            - Calls release_memory() to free unused resources.
            - Logs completion and returns True on success.
        Return
            bool: True if optimizations were applied successfully.
        Side effects
        - Modifies global torch backend flags (TF32 settings).
        - May mutate pipeline models (quantization, attention processors, VAE settings).
        - May move model parameters between devices or enable CPU offloading.
        - Alters self.config._quantization_dtype in certain quantization modes.
        - Persists pipeline manager state via mgr.save_state().
        - Produces log output via self._do_logger and may raise or propagate exceptions
          coming from torch operations, quantization helpers, or device transfers.
        Exceptions
        - May raise exceptions from torch, the quantization helpers (quantize_FP8,
          quantize_NF4, quantize_quanto), or any model/device operations.
        - May raise a cancellation-related exception if self._cancel_checker indicates
          the operation should be aborted.
        Notes
        - The first pipeline processed (index 0) is used as the canonical source for
          logging global quantization/mode information and CPU offload state.
        - Quantization is skipped for a component if its dtype already matches the
          target quantization dtype.
        - This method assumes pipelines expose attributes and methods referenced
          here (e.g. unet, text_encoder, vae, enable_vae_tiling, enable_model_cpu_offload).
        """
        self._cancel_checker()

        self._do_logger("-" * self._log_num_divisor)
        self._do_logger("Applying optimizations")
        self._do_logger("-" * self._log_num_divisor)
        torch.backends.cudnn.allow_tf32 = self.config.allow_cudnn_tf32
        self._do_logger(
            "CUDNN TF32 enabled",
            self.config.get_bool_value(self.config.allow_cudnn_tf32),
        )
        torch.backends.cuda.matmul.allow_tf32 = self.config.allow_cuda_tf32
        self._do_logger(
            "CUDA MatMul TF32 enabled",
            self.config.get_bool_value(self.config.allow_cuda_tf32),
        )

        weight_dtype = self.config.get_weight_dtype(self.config.weight_dtype)

        all_managed_pipes = list(self.pipeline_managers.values())

        for i, mgr in enumerate(all_managed_pipes):
            if mgr.pipe is not None:
                self._cancel_checker()
                self._do_logger("Optimizing pipeline for role", mgr.pipe.mode)
                if self.config.memory_attention == MemoryAttention.Xformers:
                    current_attention = next(iter(mgr.pipe.unet.attn_processors.values())).__class__.__name__.lower()
                    if MemoryAttention.Xformers.value not in current_attention:
                        mgr.pipe.enable_xformers_memory_efficient_attention()

                if self.config.quantization_method == QuantizationMethod.Layerwise:
                    if i == 0:
                        self._do_logger("Quantization method", self.config.quantization_method.value)
                        self._do_logger("Quantization mode", self.config.quantization_mode.value)
                    self.config._quantization_dtype = torch.float8_e4m3fn
                    if self.config.quantization_mode == QuantizationMode.FP8 and mgr.pipe.unet.dtype != self.config._quantization_dtype:
                        quantize_FP8(mgr.pipe.unet)
                    elif self.config.quantization_mode == QuantizationMode.NF4 and mgr.pipe.unet.dtype != self.config._quantization_dtype:
                        quantize_NF4(mgr.pipe.unet)
                        if mgr.pipe.text_encoder is not None:
                            quantize_NF4(mgr.pipe.text_encoder)
                        if mgr.pipe.text_encoder_2 is not None:
                            quantize_NF4(mgr.pipe.text_encoder_2)
                elif self.config.quantization_method == QuantizationMethod.Quanto:
                    if i == 0:
                        self._do_logger("Quantization method", self.config.quantization_method.value)
                        self._do_logger("Quantization mode", self.config.quantization_mode.value)
                        quantize_quanto(mgr.pipe.unet, self.config.quantization_mode.value)
                else:
                    self.config._quantization_dtype = weight_dtype

                if hasattr(mgr.pipe, "vae"):
                    if self.config.enable_vae_tiling:
                        if not mgr.pipe.vae.use_tiling:
                            self._do_logger(
                                "VAE tiling enabled",
                                self.config.get_bool_value(self.config.enable_vae_tiling),
                            )
                            mgr.pipe.enable_vae_tiling()
                    else:
                        if mgr.pipe.vae.use_tiling:
                            mgr.pipe.disable_vae_tiling()

                    if hasattr(mgr.pipe.vae, "run_vae_on_cpu"):
                        if self.config.run_vae_on_cpu:
                            self._do_logger(
                                "VAE run on CPU",
                                self.config.get_bool_value(self.config.run_vae_on_cpu),
                            )
                            mgr.pipe.vae.enable_run_on_cpu(True)
                        else:
                            mgr.pipe.vae.enable_run_on_cpu(False)

                    if self.config.enable_vae_slicing:
                        if not mgr.pipe.vae.use_slicing:
                            self._do_logger(
                                "VAE slicing enabled",
                                self.config.get_bool_value(self.config.enable_vae_slicing),
                            )
                            mgr.pipe.enable_vae_slicing()
                    else:
                        if mgr.pipe.vae.use_slicing:
                            mgr.pipe.disable_vae_slicing()

                if i == 0:
                    self._do_logger(
                        "CPU offload enabled",
                        self.config.get_bool_value(self.config.enable_cpu_offload),
                    )
                if self.config.enable_cpu_offload and self.config.device != RuntimeDevice.CPU:
                    mgr.pipe.enable_model_cpu_offload()
                else:
                    if mgr.is_active():
                        mgr.pipe.to(self.config.device.value)

                if mgr.saved_state is None:
                    mgr.save_state()

        self._do_logger("Memory attention", self.config.memory_attention.value)

        release_memory()
        self._do_logger("Applying optimizations... OK!")
        self._do_logger("-" * self._log_num_divisor)
        return True

    # endregion

    # region workflow
    def _process_single_image(self, initial_image, prompt, image_index, restored_dir, upscaled_dir):
        """
        Process a single input image through the configured restoration and upscaling pipelines.
        This method orchestrates the per-image workflow that may include a restoration phase
        (facial or general image restoration) followed by an upscaling phase. Behavior is
        controlled by self.config (restorer/upscaler engines and upscaler pipeline parameters).
        The method handles model pipeline activation/deactivation, model loading/reloading,
        optimizations, seed management, logging, cancellation checks, and optional saving of
        intermediate/final results to the provided directories.
        Parameters
        - initial_image (PIL.Image.Image):
            The original image to be processed.
        - prompt (str):
            A textual prompt passed to the upscaler (or other pipelines that make use of prompts).
        - image_index (int):
            Zero-based index of the current image in a larger batch; used for logging and
            for composing filenames when saving results.
        - restored_dir (str | pathlib.Path):
            Directory path where restored images (if any) should be saved by the restoration phase.
        - upscaled_dir (str | pathlib.Path):
            Directory path where upscaled images (if any) should be saved by the upscaling phase.
        Behavior / Side effects
        - If a restorer engine is configured, runs the restoration phase via
          self._run_restoration_phase. On restoration success, the returned restored images and
          their seeds are used as inputs to the subsequent upscaling phase.
        - If no restorer is configured, the original image is duplicated according to the
          upscaler_pipeline_params.num_images value and seeds are generated:
            - If upscaler_pipeline_params.seed != -1, the first seed is that fixed seed and the
              remainder are random.
            - Otherwise, all seeds are random.
        - If an upscaler engine is configured, runs the upscaling phase for each image to
          process via self._run_upscaling_phase. The method may:
            - Reload upscaler models (self.load_models) and reapply optimizations when reloaded.
            - Apply image size fixes when no restorer was used (e.g., progressive resize).
            - Continue to the next sub-image if an upscaling attempt fails (logs a warning).
        - Device and pipeline management:
            - Ensures the appropriate pipeline is active for each phase and deactivates other
              engines if engines differ.
            - If config.always_offload_models is set, ensures both restorer and upscaler pipelines
              are inactive before returning.
        - Cancellation:
            - Periodically calls self._cancel_checker() to allow the operation to be canceled.
        - Logging:
            - Emits informative and error logs about phase starts, pipeline changes, failures,
              and other notable events via self._do_logger.
        Return value
        - (final_result_image, success) where:
            - final_result_image (PIL.Image.Image | None):
                The resulting image chosen as the final output. This is typically the first
                restored image (if a restorer ran) or the first successfully upscaled image.
                If nothing was done (no restorer and no upscaler), None is returned.
            - success (bool):
                True indicates the workflow for this image completed (even if nothing was done).
                False indicates an unrecoverable failure (for example, when the restoration
                phase was configured and failed), and no further processing was performed.
        Notes and edge cases
        - If the restoration phase is configured and fails, the method logs an error and
          returns (None, False) to indicate workflow abortion for this image.
        - If neither restorer nor upscaler is selected, the method logs a warning and returns
          (None, True) (nothing to do but not an error).
        - If multiple images are produced by the restorer, they will be fed individually to
          the upscaler; the method tracks seeds per-sub-image and attempts upscaling for each.
        - The method may write intermediate/final outputs to restored_dir and/or upscaled_dir
          depending on the implementation details of the restoration/upscaling helper methods.
        """

        self._cancel_checker()

        # unload llava if loaded
        self._unload_llava_model()

        has_restorer = self.config.restorer_engine != RestorerEngine.Nothing
        has_upscaler = self.config.upscaler_engine != UpscalerEngine.Nothing
        are_engines_different = has_restorer and has_upscaler and self.config.restorer_engine.value != self.config.upscaler_engine.value

        images_to_upscale = []
        seeds_for_upscale = []
        original_w, original_h = initial_image.size

        if has_restorer:
            self._do_logger(f"---- Starting Restoration Phase with {self.config.restorer_engine.value} ----")
            self._do_logger("Preparing devices for Restoration phase...")
            if are_engines_different:
                self._ensure_pipeline_is_inactive(self.config.upscaler_engine)
            self._ensure_pipeline_is_active(self.config.restorer_engine)

            restored_images, _, _, success, restoration_seeds = self._run_restoration_phase(initial_image, prompt, image_index, restored_dir)
            if success and restored_images:
                images_to_upscale = restored_images
                seeds_for_upscale = restoration_seeds
            else:
                self._do_logger(
                    "Restoration phase failed. Aborting workflow for this image.",
                    level=logging.ERROR,
                )
                return None, False
        else:
            images_to_upscale = [initial_image] * self.config.upscaler_pipeline_params.num_images
            if self.config.upscaler_pipeline_params.seed != -1:
                seeds_for_upscale = [self.config.upscaler_pipeline_params.seed] + [random.randint(0, MAX_SEED) for _ in range(len(images_to_upscale) - 1)]
            else:
                seeds_for_upscale = [random.randint(0, MAX_SEED) for _ in range(len(images_to_upscale))]

        first_image_upscaled = None
        if has_upscaler:
            self._do_logger(f"---- Starting Upscaling Phase with {self.config.upscaler_engine.value} ----")
            self._do_logger("Checking and preparing Upscaler pipeline...")
            upscaler_was_reloaded = self.load_models(roles_to_load=["Upscaler"])
            if upscaler_was_reloaded:
                self._do_logger("Upscaler pipeline was changed, reapplying optimizations...")
                self.apply_optimizations()

            self._do_logger("Preparing devices for Upscaling phase...")
            if are_engines_different:
                self._ensure_pipeline_is_inactive(self.config.restorer_engine)
            self._ensure_pipeline_is_active(self.config.upscaler_engine)

            for sub_index, image_to_process in enumerate(images_to_upscale):
                self._cancel_checker()
                current_seed = seeds_for_upscale[sub_index]

                if not has_restorer:
                    self._do_logger("---- Fixing initial image size for upscaler phase ----")
                    if self.config.upscaler_pipeline_params.image_size_fix_mode == ImageSizeFixMode.ProgressiveResize:
                        self._do_logger("Applying ProgressiveResize to initial image...")
                        image_to_process = check_image(image_to_process, scale_factor=1.0)
                    self._do_logger("-" * self._log_num_divisor)

                final_image, status = self._run_upscaling_phase(
                    initial_image=initial_image,
                    image_to_upscale=image_to_process,
                    prompt=prompt,
                    initial_width=original_w,
                    initial_height=original_h,
                    image_index=image_index,
                    upscaled_dir=upscaled_dir,
                    base_seed=current_seed,
                    image_sub_index=sub_index,
                )
                if not status:
                    self._do_logger(
                        f"Upscaling failed for image {image_index + 1}_{sub_index + 1}. Continuing to next.",
                        level=logging.WARNING,
                    )
                    continue

                if sub_index == 0:
                    first_image_upscaled = final_image

        if not has_restorer and not has_upscaler:
            self._do_logger(
                "No restorer or upscaler selected. Nothing to do.",
                level=logging.WARNING,
            )
            return None, True

        final_result_image = images_to_upscale[0] if first_image_upscaled is None else first_image_upscaled

        if self.config.always_offload_models:
            self._ensure_pipeline_is_inactive(self.config.restorer_engine)
            self._ensure_pipeline_is_inactive(self.config.upscaler_engine)

        return final_result_image, True

    def _run_restoration_phase(
        self,
        image_to_restore: Image.Image,
        prompt: str,
        image_index: int,
        restored_dir: str,
    ):
        """
        Run the image restoration phase for a single input image.
        This method prepares and runs the configured 'restorer' pipeline on a single
        input image and performs the post-processing, optional face restoration and
        color correction, and finally saves the restored outputs to disk.
        Behavior summary
        - Ensures the image meets the target minimum size according to the configured
            image size fix mode (ProgressiveResize, Padding, or none).
        - Optionally prepares IPA image prompt embeddings.
        - Prepares and logs the final prompt / negative prompt used by the restorer.
        - Selects the active restorer pipeline and configures progress bar and PAG
            layers if enabled.
        - Builds deterministic or random seeds and corresponding torch.Generator(s).
        - Calls the restorer pipeline in inference mode and collects returned images.
        - For each restored image:
            - Checks for cancellation via self._cancel_checker().
            - Crops padded outputs back to original dimensions when Padding was used.
            - Resizes 1.0x outputs back to the original size when necessary.
            - Optionally applies selective face restoration (only when target scale == 1.0).
            - Applies configured color-fix method (Wavelet or Adain) using the original
                image as color reference when requested.
            - Saves the final image with metadata and collects the saved image result.
        - Returns the list of saved outputs, original image dimensions, a success flag,
            and the list of seeds used (or None when no restorer pipeline was active).
        Parameters
        - image_to_restore (PIL.Image.Image):
                The input low-quality image to restore.
        - prompt (str | None):
                Optional prompt override to use for this run. If None, the prompt from
                self.config.restorer_pipeline_params.prompt is used.
        - image_index (int):
                Index of the image in the batch (used for logging and naming).
        - restored_dir (str):
                Directory path where restored images will be saved.
        Returns
        - tuple:
            - images_for_next_stage (list):
                    A list containing the result of save_image_with_metadata(...) for each
                    restored image. The concrete type depends on that helper's return
                    value (commonly a PIL.Image or a path/metadata wrapper).
            - initial_w (int):
                    Original input image width (before any resizing/padding).
            - initial_h (int):
                    Original input image height (before any resizing/padding).
            - success (bool):
                    True if the restorer pipeline executed and produced outputs; False if no
                    valid restorer pipeline was active (in that case images_for_next_stage is []).
            - seeds (list[int] | None):
                    List of integer seeds used to produce each output image. If no pipeline
                    was active, this is None.
        Side effects and important notes
        - Calls self._cancel_checker() periodically; a cancellation mechanism may stop
            execution by raising or returning control (depending on its implementation).
        - Logs details about pipeline configuration, prompts, sizes, and each
            post-processing step via self._do_logger(...).
        - May modify internal state:
            - self.image_prompt_embeds and self.uncond_image_prompt_embeds are set when
                apply_ipa_embeds is enabled; otherwise they are cleared.
            - The active restorer pipeline may have its PAG layers set via
                restorer_pipe.set_pag_layers(...).
            - The pipeline's progress bar may be configured with self.progress_bar_handler.
        - If a grounding_dino_sam2_model is present it will be moved to CPU and
            release_memory() is invoked before running the pipeline and again after face
            restoration to free GPU memory.
        - The function runs the restorer pipeline under torch.inference_mode() and
            therefore does not compute gradients.
        Errors and exceptions
        - The method may raise exceptions from:
            - PIL image operations (resize, crop, save).
            - The restorer pipeline execution.
            - Helper utilities such as prepare_pipeline_params, save_image_with_metadata,
                wavelet_color_fix, adain_color_fix, or face restoration helpers.
        - If no active restorer pipeline is found, the method will log an error and
            return ([], initial_w, initial_h, False, None) rather than raising.
        Concurrency / device notes
        - Generators are created on the device returned by self.config.get_device(
            self.config.generator_device ) and seeded deterministically when a specific
            seed is provided; otherwise seeds are random.
        - This method expects appropriate model and device setup to be handled elsewhere
            (e.g., model loading, moving models to GPU) and attempts to reduce memory use
            for certain auxiliary models by moving them to CPU and calling release_memory().
        """
        self._cancel_checker()

        self._prepare_scheduler("Restorer")
        initial_w, initial_h = image_to_restore.size
        target_scale_factor = float(UPSCALER_SCALES[self.config.restorer_pipeline_params.upscale_factor])

        if self.config.restorer_pipeline_params.image_size_fix_mode == ImageSizeFixMode.ProgressiveResize:
            self._do_logger("Applying ProgressiveResize to ensure minimum dimensions...")
            input_image = check_image(image_to_restore, scale_factor=target_scale_factor)
            width_now, height_now = input_image.size
            width_init, height_init = initial_w, initial_h
        elif self.config.restorer_pipeline_params.image_size_fix_mode == ImageSizeFixMode.Padding:
            self._do_logger("Applying pre-resize and Padding to ensure minimum dimensions...")
            target_w = int(initial_w * target_scale_factor)
            target_h = int(initial_h * target_scale_factor)
            image_for_padding = image_to_restore.copy()
            image_for_padding = image_for_padding.resize((target_w, target_h), Image.Resampling.LANCZOS)
            input_image, width_init, height_init, width_now, height_now = padding_image_size(image_for_padding, 32)
        else:
            input_image = image_to_restore
            width_now, height_now = initial_w, initial_h
            width_init, height_init = initial_w, initial_h
            self._do_logger(
                "No image size fix mode applied. Using original size",
                f"{width_now}x{height_now}",
            )

        if self.config.restorer_pipeline_params.apply_ipa_embeds:
            self._prepare_ipa(input_image)
        else:
            self.image_prompt_embeds = None
            self.uncond_image_prompt_embeds = None

        final_prompt = self.config.restorer_pipeline_params.prompt if prompt is None else prompt
        self.config.restorer_pipeline_params.prompt = self._prepare_prompt(
            final_prompt,
            self.config.restorer_pipeline_params.prompt_2,
            self.config.restorer_pipeline_params.invert_prompts,
            self.config.restorer_pipeline_params.apply_prompt_2,
        )
        self._do_logger("Target scale", f"{target_scale_factor}x")
        self._do_logger("Restorer Prompt", self.config.restorer_pipeline_params.prompt[0])
        self._do_logger(
            "Restorer Negative Prompt",
            self.config.restorer_pipeline_params.negative_prompt,
        )
        self._do_logger("-" * self._log_num_divisor)

        restorer_pipe = self._get_active_restorer_pipe()
        if not restorer_pipe:
            self._do_logger("No valid restorer pipeline is active.", level=logging.ERROR)
            return [], initial_w, initial_h, False, None

        if self.progress_bar_handler:
            restorer_pipe.set_progress_bar_config(
                file=self.progress_bar_handler,
                disable=False,
                ncols=100,
                dynamic_ncols=True,
                ascii=" █",
            )

        if self.config.restorer_pipeline_params.seed != -1:
            seeds = [self.config.restorer_pipeline_params.seed] + [
                random.randint(0, MAX_SEED) for _ in range(self.config.restorer_pipeline_params.num_images - 1)
            ]
        else:
            seeds = [random.randint(0, MAX_SEED) for _ in range(self.config.restorer_pipeline_params.num_images)]

        generators = [torch.Generator(self.config.get_device(self.config.generator_device)).manual_seed(s) for s in seeds]
        if len(generators) == 1:
            generators = generators[0]

        pipeline_params = self._prepare_pipeline_params(
            restorer_pipe.__class__.__name__,
            self.config.restorer_pipeline_params,
            initial_w,
            initial_h,
            width_now,
            height_now,
            input_image,
        )

        if self.config.restorer_pipeline_params.enable_PAG:
            pag_layers = [PAG_LAYERS[l] for l in self.config.restorer_pipeline_params.pag_layers]
            restorer_pipe.set_pag_layers(pag_layers)
        else:
            restorer_pipe.set_pag_layers(None)

        if self.grounding_dino_sam2_model is not None:
            self.grounding_dino_sam2_model.to("cpu")
            release_memory()

        with torch.inference_mode():
            restored_images = restorer_pipe(generator=generators, **pipeline_params).images

        images_for_next_stage = []
        for i, restored_image in enumerate(restored_images):
            self._cancel_checker()

            image_rank = f"{image_index + 1}_{i + 1}"
            self._do_logger(f"Post-processing with restored image {image_rank}:")
            image_name = f"{self._get_image_name(seeds[i], f'{image_rank}_{int(target_scale_factor)}x')}"
            restored_image_file = os.path.join(restored_dir, f"{image_name}{self.config.save_image_format}")
            image_to_save_and_forward = restored_image

            if self.config.restorer_pipeline_params.image_size_fix_mode == ImageSizeFixMode.Padding and (
                image_to_save_and_forward.width != width_init or image_to_save_and_forward.height != height_init
            ):
                self._do_logger(
                    "Cropping padded image",
                    f"from {image_to_save_and_forward.width}x{image_to_save_and_forward.height} to {width_init}x{height_init}",
                )
                image_to_save_and_forward = image_to_save_and_forward.crop((0, 0, width_init, height_init))

            if target_scale_factor == 1.0:
                if image_to_save_and_forward.size != (initial_w, initial_h) and (
                    (initial_w >= 1024 and initial_h >= 1024) or (initial_w >= 1024 and initial_h >= 768) or (initial_w >= 768 and initial_h >= 1024)
                ):
                    self._do_logger(
                        "Resizing 1x restoration to original size",
                        f"from {image_to_save_and_forward.width}x{image_to_save_and_forward.height} to {initial_w}x{initial_h}",
                    )
                    image_to_save_and_forward = image_to_save_and_forward.resize((initial_w, initial_h), Image.Resampling.LANCZOS)

            final_image_to_save = image_to_save_and_forward

            if self.config.restore_face:
                if target_scale_factor == 1.0:
                    self._do_logger("Applying selective face restoration for 1x scale factor...")
                    final_image_to_save = self._apply_face_restoration_mask(
                        original_lq_image=image_to_restore,
                        restored_image=image_to_save_and_forward,
                    )
                    self.grounding_dino_sam2_model.to("cpu")
                    release_memory()
                else:
                    self._do_logger(
                        f"Skipping selective face restoration because target scale factor is {target_scale_factor}x (must be 1.0x).",
                        level=logging.WARNING,
                    )

            # apply color fix if enabled
            if self.config.restorer_pipeline_params.color_fix_mode != ColorFix.Nothing:
                image_color_reference = image_to_restore.copy()
                if image_color_reference.width != final_image_to_save.width or image_color_reference.height != final_image_to_save.height:
                    image_color_reference = image_color_reference.resize(final_image_to_save.size)

                if self.config.restorer_pipeline_params.color_fix_mode == ColorFix.Wavelet:
                    self._do_logger(
                        f"Applying {self.config.restorer_pipeline_params.color_fix_mode.value} color fix to image {image_rank}",
                        restored_image_file,
                    )
                    final_image_to_save = wavelet_color_fix(final_image_to_save, image_color_reference)
                elif self.config.restorer_pipeline_params.color_fix_mode == ColorFix.Adain:
                    self._do_logger(
                        f"Applying {self.config.restorer_pipeline_params.color_fix_mode.value} color fix to image {image_rank}",
                        restored_image_file,
                    )
                    final_image_to_save = adain_color_fix(final_image_to_save, image_color_reference)

            self._do_logger(f"Saving restored image {image_rank}", restored_image_file)
            if not self.config.running_on_spaces:
                saved_image = save_image_with_metadata(final_image_to_save, restored_image_file, self.metadata)
                images_for_next_stage.append(saved_image)
            else:
                images_for_next_stage.append(final_image_to_save)

        self._do_logger("-" * self._log_num_divisor)

        return images_for_next_stage, initial_w, initial_h, True, seeds

    def _run_upscaling_phase(
        self,
        initial_image: Image.Image,  # only for color reference
        image_to_upscale: Image.Image,  # this is the image used for upscale
        prompt: str,
        initial_width: int,
        initial_height: int,
        image_index: int,
        upscaled_dir: str,
        base_seed: int,
        image_sub_index: int,
    ):
        """
        Run the upscaling phase for a single image.

        This method performs upscaling using the active upscaler pipeline configured
        in self.config.upscaler_pipeline_params. It supports two upscaling modes:

        - Progressive: performs multiple 2x passes until the target scale is reached,
          applying optional per-pass CFG/strength decay and optionally saving
          intermediate pass outputs.
        - Direct: performs a single pass to reach the final target resolution.

        Key behaviors:
        - Resolves the target scale from UPSCALER_SCALES and immediately skips (returns
          (False, False)) when the target scale is <= 1.0.
        - Obtains the active upscaler pipeline and configures progress bar and PAG
          layers (if supported).
        - Applies IPA embeddings when configured; otherwise clears image prompt embeds.
        - Prepares the prompt with _prepare_prompt and logs prompts and negative prompts.
        - For each pass:
            - Uses a torch.Generator seeded with base_seed for deterministic results.
            - Resizes the image for the pass, optionally pads to a multiple of 64
              (padding_image_size), and prepares pipeline parameters via
              _prepare_pipeline_params.
            - Executes the pipeline under torch.inference_mode().
            - Crops padded outputs back to the original content region.
            - Optionally saves intermediate pass images using _save_pass_image.
            - For Progressive mode, decays guidance_scale (CFG) and strength according
              to cfg_decay_rate and strength_decay_rate between passes.
        - After upscaling completes, ensures the final image matches the expected
          resolution (initial_width/initial_height * target_scale_factor) and resizes
          with LANCZOS if necessary.
        - The final image is saved via _save_pass_image and returned.

        Cancellation and logging:
        - The method periodically calls self._cancel_checker() to support cooperative
          cancellation.
        - Progress and errors are logged via self._do_logger().

        Parameters:
        - initial_image (PIL.Image.Image):
            Reference image used for color/context when saving or post-processing.
        - image_to_upscale (PIL.Image.Image):
            The actual image passed to the upscaler pipeline.
        - prompt (str | None):
            Optional prompt override. If None, the configured upscaler prompt is used.
        - initial_width (int), initial_height (int):
            Original image dimensions (used to compute the expected final resolution).
        - image_index (int):
            Zero-based index of the source image (for logging/filenames).
        - upscaled_dir (str):
            Directory where intermediate/final upscaled images may be saved.
        - base_seed (int):
            RNG seed used to initialize torch.Generator for deterministic outputs.
        - image_sub_index (int):
            Sub-index for multi-result images (used for saving/logging).

        Returns:
        - Tuple(final_image_or_flag, success: bool)
            - On success: (final_image: PIL.Image.Image, True).
            - On failure: (None, False) for pipeline or processing failures.
            - If the configured target scale factor is <= 1.0 the function returns
              (False, False) and performs no upscaling.

        Notes:
        - Padded areas (when using Padding image_size_fix_mode) are cropped out of
          pipeline outputs before further processing.
        - Intermediate pass images are saved only when
          config.save_image_on_upscaling_passes is True.
        """

        self._cancel_checker()
        upscaling_mode = self.config.upscaler_pipeline_params.upscaling_mode

        self._prepare_scheduler("Upscaler")
        target_scale_factor = float(UPSCALER_SCALES[self.config.upscaler_pipeline_params.upscale_factor])
        if target_scale_factor <= 1.0:
            self._do_logger("Upscaler scale factor is <= 1.0, skipping.")
            return False, False

        upscaler_pipe = self._get_active_upscaler_pipe()
        if not upscaler_pipe:
            self._do_logger("No valid upscaler pipeline is active.", level=logging.ERROR)
            return None, False

        if self.progress_bar_handler:
            upscaler_pipe.set_progress_bar_config(
                file=self.progress_bar_handler,
                disable=False,
                ncols=100,
                dynamic_ncols=True,
                ascii=" █",
            )

        if self.config.upscaler_pipeline_params.apply_ipa_embeds:
            self._prepare_ipa(image_to_upscale)
        else:
            self.image_prompt_embeds = None
            self.uncond_image_prompt_embeds = None

        final_prompt = self.config.upscaler_pipeline_params.prompt if prompt is None else prompt
        self.config.upscaler_pipeline_params.prompt = self._prepare_prompt(
            final_prompt,
            self.config.upscaler_pipeline_params.prompt_2,
            self.config.upscaler_pipeline_params.invert_prompts,
            self.config.upscaler_pipeline_params.apply_prompt_2,
        )
        self._do_logger("Target scale", f"{target_scale_factor}x")
        self._do_logger("Upscaler Prompt", self.config.upscaler_pipeline_params.prompt[0])
        self._do_logger(
            "Upscaler Negative Prompt",
            self.config.upscaler_pipeline_params.negative_prompt,
        )

        current_image_for_pass = image_to_upscale
        final_image = None

        if hasattr(upscaler_pipe, "set_pag_layers"):
            if self.config.upscaler_pipeline_params.enable_PAG:
                pag_layers = [PAG_LAYERS[l] for l in self.config.upscaler_pipeline_params.pag_layers]
                upscaler_pipe.set_pag_layers(pag_layers)
            else:
                upscaler_pipe.set_pag_layers(None)

        if upscaling_mode == UpscalingMode.Progressive:
            num_2x_passes = math.ceil(math.log2(target_scale_factor))
            self._do_logger("Performing progressive upscale", f"{num_2x_passes} pass(es) of 2x")

            current_pass_scale_factor = 2
            initial_cfg = self.config.upscaler_pipeline_params.guidance_scale
            initial_strength = self.config.upscaler_pipeline_params.strength

            cfg_decay_rate = self.config.upscaler_pipeline_params.cfg_decay_rate
            strength_decay_rate = self.config.upscaler_pipeline_params.strength_decay_rate

            num_decay_steps = num_2x_passes - 1
            if num_decay_steps > 0:
                total_cfg_decay = initial_cfg * cfg_decay_rate
                cfg_decay_per_step = total_cfg_decay / num_decay_steps

                total_strength_decay = initial_strength * strength_decay_rate
                strength_decay_per_step = total_strength_decay / num_decay_steps
            else:
                cfg_decay_per_step = 0
                strength_decay_per_step = 0

            current_cfg = initial_cfg
            current_strength = initial_strength

            for pass_num in range(num_2x_passes):
                self._cancel_checker()

                if pass_num > 0 and not self.config.enable_cpu_offload:
                    self._ensure_pipeline_is_active(self.config.upscaler_engine)

                input_for_this_pass = current_image_for_pass
                current_w, current_h = input_for_this_pass.size
                target_pass_h, target_pass_w = current_h * 2, current_w * 2

                generator = torch.Generator(self.config.get_device(self.config.generator_device)).manual_seed(base_seed)
                image_for_padding = input_for_this_pass.resize((target_pass_w, target_pass_h), Image.Resampling.LANCZOS)

                if self.config.upscaler_pipeline_params.image_size_fix_mode == ImageSizeFixMode.Padding and pass_num == 0:
                    self._do_logger("Applying Padding to initial image...")

                input_image_padded, width_init, height_init, width_now, height_now = padding_image_size(image_for_padding, 64)

                pipeline_params = self._prepare_pipeline_params(
                    upscaler_pipe.__class__.__name__,
                    self.config.upscaler_pipeline_params,
                    current_w,
                    current_h,
                    width_now,
                    height_now,
                    input_image_padded,
                    input_for_this_pass,
                )

                if pass_num > 0:
                    current_cfg -= cfg_decay_per_step
                    current_strength -= strength_decay_per_step

                current_cfg = round(max(current_cfg, 1.1), 2)
                current_strength = round(max(current_strength, 0.1), 2)

                pipeline_params["guidance_scale"] = current_cfg
                pipeline_params["strength"] = current_strength

                self._do_logger(
                    f"Upscaling Pass {pass_num + 1}/{num_2x_passes}",
                    f"Target: {width_now}x{height_now}, CFG: {current_cfg}, Strength: {current_strength}",
                )

                with torch.inference_mode():
                    upscaled_pass_results = upscaler_pipe(generator=generator, **pipeline_params).images

                if upscaled_pass_results:
                    self._do_logger(
                        "Cropping padded image",
                        f"from {upscaled_pass_results[0].width}x{upscaled_pass_results[0].height} to {width_init}x{height_init}",
                    )
                    cropped_image = upscaled_pass_results[0].crop((0, 0, width_init, height_init))
                    current_image_for_pass = cropped_image
                else:
                    self._do_logger(
                        "Upscaling pass failed. Stopping process for this image.",
                        level=logging.ERROR,
                    )
                    return None, False

                if self.config.save_image_on_upscaling_passes and current_pass_scale_factor < target_scale_factor:
                    self._save_pass_image(
                        initial_image,
                        image_index,
                        upscaled_dir,
                        base_seed,
                        image_sub_index,
                        current_pass_scale_factor,
                        current_image_for_pass,
                    )

                current_pass_scale_factor += 2

            final_image = current_image_for_pass

        elif upscaling_mode == UpscalingMode.Direct:
            self._do_logger("Performing a single direct upscale pass.")

            current_w, current_h = current_image_for_pass.size
            target_w = int(current_w * target_scale_factor)
            target_h = int(current_h * target_scale_factor)

            generator = torch.Generator(self.config.get_device(self.config.generator_device)).manual_seed(base_seed)
            image_for_padding = current_image_for_pass.resize((target_w, target_h), Image.Resampling.LANCZOS)

            if self.config.upscaler_pipeline_params.image_size_fix_mode == ImageSizeFixMode.Padding:
                self._do_logger("Applying Padding to initial image...")

            input_image_padded, width_init, height_init, width_now, height_now = padding_image_size(image_for_padding, 64)

            pipeline_params = self._prepare_pipeline_params(
                upscaler_pipe.__class__.__name__,
                self.config.upscaler_pipeline_params,
                current_w,
                current_h,
                width_now,
                height_now,
                input_image_padded,
                current_image_for_pass,
            )

            self._do_logger(
                "Direct Upscaling",
                f"Target: {width_now}x{height_now}, CFG: {pipeline_params['guidance_scale']}, Strength: {pipeline_params['strength']}",
            )

            upscaled_pass_results = upscaler_pipe(generator=generator, **pipeline_params).images

            if upscaled_pass_results:
                self._do_logger(
                    "Cropping padded image",
                    f"from {upscaled_pass_results[0].width}x{upscaled_pass_results[0].height} to {width_init}x{height_init}",
                )
                cropped_image = upscaled_pass_results[0].crop((0, 0, width_init, height_init))
                final_image = cropped_image
            else:
                self._do_logger(
                    "Direct upscaling pass failed. Stopping process for this image.",
                    level=logging.ERROR,
                )
                return None, False

        if final_image is None:
            self._do_logger("Upscaling failed, no final image was produced.", level=logging.ERROR)
            return None, False

        self._do_logger("-" * self._log_num_divisor)
        self._do_logger(f"Post-processing with upscaled image {image_index + 1}_{image_sub_index + 1}:")
        final_upscaled_w, final_upscaled_h = final_image.size
        target_final_w = int(initial_width * target_scale_factor)
        target_final_h = int(initial_height * target_scale_factor)

        if final_upscaled_w != target_final_w or final_upscaled_h != target_final_h:
            self._do_logger(
                "Final resolution adjustment",
                f"from {final_upscaled_w}x{final_upscaled_h} to {target_final_w}x{target_final_h}",
            )
            final_image = final_image.resize((target_final_w, target_final_h), Image.Resampling.LANCZOS)

        final_image = self._save_pass_image(
            initial_image,
            image_index,
            upscaled_dir,
            base_seed,
            image_sub_index,
            target_scale_factor,
            final_image,
        )
        return final_image, True

    # endregion

    # region load and setup models
    def _load_pipeline_for_role(self, role: str) -> bool:
        """
        Load or reuse a processing pipeline for a given role.
        This internal helper attempts to ensure the pipeline used for the specified
        role is available and up-to-date. It will either reuse an existing pipeline
        manager or load a new pipeline and register it with the manager registry.
        Behavior summary
        - Performs an early cancellation check via self._cancel_checker().
        - Determines whether the request is for the "Upscaler" or "Restorer" role.
        - Chooses the configured engine and checkpoint for the role.
        - If the configured engine indicates "None", the method returns False.
        - Compares the desired engine/checkpoint (and SUPIR sub-model when applicable)
            to the currently active/recorded values to decide whether a reload is required.
        - If no reload is needed and a pipeline manager already exists, the method
            logs that the pipeline is up-to-date and returns False.
        - Otherwise, it resolves the checkpoint model name (file vs registered model),
            sets config._selected_model_path, and:
                - If the checkpoint comes from a path, attempts to load checkpoint info
                    (via do_load) and enforces supported checkpoint types (e.g. "SD-XL" or
                    "VAE-v1-BROKEN"); on failure returns False.
                - If the checkpoint is a registered model, queries model settings from the
                    model manager; if missing, returns False.
        - Delegates actual pipeline construction to engine-specific loaders:
            _load_SUPIR_model, _load_FaithDiff_model, or _load_ControlNetTile_model.
        - If a new pipeline is returned:
                - If the upscaler and restorer share the same engine, the existing
                    PipelineDeviceManager for that engine is reused (its .pipe replaced).
                - Otherwise, a new PipelineDeviceManager is created and registered.
                - Updates internal state attributes: active_*_engine, current_prior_*_model,
                    and current_*_supir_model (for SUPIR engines).
                - Returns True to indicate a pipeline was loaded/changed.
        - If no pipeline is created or an error occurs, returns False.
        Parameters
        - role (str): Role identifier; expected values are "Restorer" or "Upscaler".
            Behavior differs only by selecting the corresponding engine, checkpoint and
            SUPIR sub-model config.
        Return
        - bool: True if a pipeline was newly loaded or replaced; False if no change was
            necessary or if loading failed/was not applicable.
        Side effects
        - Calls self._cancel_checker() and self._do_logger() (logging).
        - May call do_load() (to inspect a checkpoint file) and model_manager.get_model_settings().
        - Updates attributes on self and self.config, including:
            - config._selected_model_path
            - self.checkpoint_info (when loading from file)
            - self.current_prior_restorer_base (when loading from file)
            - active_upscaler_engine / active_restorer_engine
            - current_prior_upscaler_model / current_prior_restorer_model
            - current_upscaler_supir_model / current_restorer_supir_model (when applicable)
            - self.pipeline_managers (adding or updating PipelineDeviceManager entries)
        - On error paths it logs the exception and returns False instead of raising.
        Notes and constraints
        - This method is an internal routine and assumes callers manage concurrency.
        - Supported engine-specific loaders are invoked; if a loader returns None, the
            method treats this as a failed load and returns False.
        - File-based checkpoints are validated for supported types; unsupported types
            cause a logged message and a False return.
        """

        self._cancel_checker()
        is_upscaler = role == "Upscaler"

        engine = self.config.upscaler_engine if is_upscaler else self.config.restorer_engine
        if engine.value == "None":
            return False

        checkpoint = self.config.selected_upscaler_checkpoint_model if is_upscaler else self.config.selected_restorer_checkpoint_model

        current_prior_attr = "current_prior_upscaler_model" if is_upscaler else "current_prior_restorer_model"
        active_engine_attr = "active_upscaler_engine" if is_upscaler else "active_restorer_engine"

        # Reload Logic
        needs_reload = False
        if getattr(self, active_engine_attr) != engine or getattr(self, current_prior_attr) != checkpoint:
            needs_reload = True

        is_supir_engine = engine == RestorerEngine.SUPIR or engine == UpscalerEngine.SUPIR
        if is_supir_engine:
            current_supir_attr = "current_upscaler_supir_model" if is_upscaler else "current_restorer_supir_model"
            target_supir_model = self.config.upscaler_pipeline_params.supir_model if is_upscaler else self.config.restorer_pipeline_params.supir_model
            if getattr(self, current_supir_attr) != target_supir_model:
                needs_reload = True

        if not needs_reload and len(self.pipeline_managers) > 0:
            self._do_logger(f"{role} pipeline is up-to-date. Reusing.")
            return False

        # Loading Logic
        self._do_logger(f"Loading pipeline for {role} role...")

        model_name, is_from_file = self.config.get_checkpoint_model_name(checkpoint)
        prior_model_settings = None
        if not is_from_file:
            prior_model_settings = self.model_manager.get_model_settings(ModelType.Diffusers.value, None, model_name)
            if prior_model_settings is None:
                return False
            self.config._selected_model_path = prior_model_settings["model_path"]
        else:
            self.config._selected_model_path = model_name

            # get checkpoint info
            try:
                self.checkpoint_info = do_load(self.config._selected_model_path)
                self.current_prior_restorer_base = "SD-XL"
                logger.info(f"Checkpoint type: {self.current_prior_restorer_base}")
            except Exception as e:
                logger.error(e)
                return False

            if self.current_prior_restorer_base not in ("SD-XL", "VAE-v1-BROKEN"):
                print("Only SDXL model type are supported!")
                return False

        new_pipe = None
        if engine == RestorerEngine.SUPIR or engine == UpscalerEngine.SUPIR:
            new_pipe = self._load_SUPIR_model(prior_model_settings, is_from_file, role)
        elif engine == RestorerEngine.FaithDiff or engine == UpscalerEngine.FaithDiff:
            new_pipe = self._load_FaithDiff_model(prior_model_settings, is_from_file, role)
        elif engine == UpscalerEngine.ControlNetTile:
            new_pipe = self._load_ControlNetTile_model(prior_model_settings, is_from_file, role)

        if new_pipe:
            # Manage the case where Restorer and Upscaler share the same engine
            # If the upscaler is the same engine as the restorer, do not create a new manager, reuse it
            if is_upscaler and self.config.restorer_engine == self.config.upscaler_engine:
                if self.config.restorer_engine.name in self.pipeline_managers:
                    self.pipeline_managers[self.config.restorer_engine.name].pipe = new_pipe
            else:
                self.pipeline_managers[engine.name] = PipelineDeviceManager(new_pipe, is_active=True)

            # Update the state
            setattr(self, active_engine_attr, engine)
            setattr(self, current_prior_attr, checkpoint)
            if is_supir_engine:
                current_supir_attr = "current_upscaler_supir_model" if is_upscaler else "current_restorer_supir_model"
                target_supir_model = self.config.upscaler_pipeline_params.supir_model if is_upscaler else self.config.restorer_pipeline_params.supir_model
                setattr(self, current_supir_attr, target_supir_model)

            return True

        return False

    def _load_SUPIR_model(self, prior_model_settings, is_from_file, role):
        """
        Loads the SUPIR model components based on the specified role (Restorer or Upscaler).
        Parameters:
        - prior_model_settings (dict): A dictionary containing prior model settings, including variant and safetensor usage.
        - is_from_file (bool): A flag indicating whether to load the model from a file or not.
        - role (str): The role for which the model is being loaded. It can be either "Restorer" or "Upscaler".
        Returns:
        - current_pipeline (SUPIRStableDiffusionXLPipeline or None): The loaded pipeline for the specified role, or None if loading fails.
        Raises:
        - Exception: Logs an error message if the model loading fails for any reason.
        """
        self._cancel_checker()
        controlnet = None
        denoise_encoder = None
        unet = None
        current_pipeline = None

        try:
            weight_dtype = self.config.get_weight_dtype(self.config.weight_dtype)
            device = self.config.device.value

            if role == "Restorer":
                supir_model_enum = self.config.restorer_pipeline_params.supir_model
            elif role == "Upscaler":
                supir_model_enum = self.config.upscaler_pipeline_params.supir_model

            supir_model_name_key = SUPIR_MODELS[supir_model_enum.value]["model_name"]
            model_name, _ = self.config.get_model_name(supir_model_name_key)
            model_settings = self.model_manager.get_model_settings(ModelType.Restorer.value, None, model_name)

            model_path = model_settings["model_path"]
            supir_controlnet_path = os.path.join(model_path, "controlnet")
            self._do_logger("Loading SUPIR ControlNet from", supir_controlnet_path)
            controlnet = SUPIRControlNet.from_pretrained(
                supir_controlnet_path,
                variant=model_settings["variant"],
                torch_dtype=weight_dtype,
                use_safetensors=model_settings["use_safetensors"],
                disable_mmap=self.config.disable_mmap,
            )

            supir_denoise_path = os.path.join(model_path, "denoise_encoder")
            self._do_logger("Loading SUPIR Denoise encoder from", supir_denoise_path)
            denoise_encoder = SUPIRDenoiseEncoder.from_pretrained(
                supir_denoise_path,
                torch_dtype=torch.float32,
                disable_mmap=self.config.disable_mmap,
            ).to(device)

            if not is_from_file:
                unet_params = {
                    "pretrained_model_name_or_path": self.config._selected_model_path,
                    "torch_dtype": weight_dtype,
                    "subfolder": "unet",
                    "variant": prior_model_settings["variant"],
                }
                if not prior_model_settings["use_safetensors"]:
                    unet_params["use_safetensors"] = False

                self._do_logger("Loading UNet from", self.config._selected_model_path)
                unet_params["supir_model_path"] = model_settings["model_path"]
                unet = SUPIRUNet.from_pretrained(**unet_params)
                pipe_params = {
                    "pretrained_model_name_or_path": self.config._selected_model_path,
                    "torch_dtype": weight_dtype,
                    "add_watermarker": False,
                    "controlnet": controlnet,
                    "denoise_encoder": denoise_encoder,
                    "variant": prior_model_settings["variant"],
                    "unet": unet,
                }

                if not prior_model_settings["use_safetensors"]:
                    pipe_params["use_safetensors"] = False

                self._do_logger(f"Loading {role} pipeline...")
                current_pipeline = SUPIRStableDiffusionXLPipeline.from_pretrained(**pipe_params)
            else:
                self._do_logger("Loading UNet from", self.config._selected_model_path)
                unet = SUPIRUNet.from_single_file(
                    self.config._selected_model_path,
                    torch_dtype=weight_dtype,
                    subfolder="unet",
                    supir_model_path=model_settings["model_path"],
                    disable_mmap=self.config.disable_mmap,
                )
                self._do_logger(f"Loading {role} pipeline...")
                current_pipeline = SUPIRStableDiffusionXLPipeline.from_single_file(
                    self.config._selected_model_path,
                    torch_dtype=weight_dtype,
                    cache_dir=None,
                    controlnet=controlnet,
                    denoise_encoder=denoise_encoder,
                    unet=None,
                    add_watermarker=False,
                    disable_mmap=self.config.disable_mmap,
                )
                current_pipeline.unet = unet
            current_pipeline.mode = role
            self._do_logger(f"Loading {role} pipeline... OK!")
            return current_pipeline
        except Exception as e:
            self._do_logger(
                f"Failed to load SUPIR model for mode '{role}'",
                e,
                level=logging.ERROR,
                exc_info=True,
            )
            return None

        finally:
            del controlnet
            del denoise_encoder
            del unet
            release_memory()

    def _load_FaithDiff_model(self, prior_model_settings, is_from_file, role):
        """
        Loads the FaithDiff model based on the specified role and model settings.
        Parameters:
            prior_model_settings (dict): A dictionary containing prior model settings, including 'variant' and 'use_safetensors'.
            is_from_file (bool): A flag indicating whether to load the model from a file or not.
            role (str): The role for which the model is being loaded. It can be either 'Restorer' or 'Upscaler'.
        Returns:
            current_pipeline (FaithDiffStableDiffusionXLPipeline or None):
                The loaded pipeline for the specified role if successful, otherwise None.
        Raises:
            Exception: Logs an error if the model fails to load, including the exception details.
        """
        self._cancel_checker()
        unet = None
        current_pipeline = None

        try:
            weight_dtype = self.config.get_weight_dtype(self.config.weight_dtype)

            if role == "Restorer":
                selected_model = self.config.restorer_engine.value
            elif role == "Upscaler":
                selected_model = self.config.upscaler_engine.value

            model_name, _ = self.config.get_model_name(selected_model)

            model_settings = self.model_manager.get_model_settings(ModelType.Restorer.value, None, model_name)
            if not is_from_file:
                unet_params = {
                    "pretrained_model_name_or_path": self.config._selected_model_path,
                    "torch_dtype": weight_dtype,
                    "subfolder": "unet",
                    "variant": prior_model_settings["variant"],
                }
                if not prior_model_settings["use_safetensors"]:
                    unet_params["use_safetensors"] = False

                self._do_logger("Loading UNet from", self.config._selected_model_path)
                unet = FaithDiffUNet.from_pretrained(**unet_params)
                unet.load_additional_layers(weight_path=model_settings["model_path"], dtype=weight_dtype)
                pipe_params = {
                    "pretrained_model_name_or_path": self.config._selected_model_path,
                    "torch_dtype": weight_dtype,
                    "add_watermarker": False,
                    "variant": prior_model_settings["variant"],
                    "unet": unet,
                }

                if not prior_model_settings["use_safetensors"]:
                    pipe_params["use_safetensors"] = False

                self._do_logger(f"Loading {role} pipeline...")
                current_pipeline = FaithDiffStableDiffusionXLPipeline.from_pretrained(**pipe_params)

            else:
                self._do_logger("Loading UNet from", self.config._selected_model_path)
                unet = FaithDiffUNet.from_single_file(
                    self.config._selected_model_path,
                    torch_dtype=weight_dtype,
                    subfolder="unet",
                    disable_mmap=self.config.disable_mmap,
                )
                unet.load_additional_layers(weight_path=model_settings["model_path"], dtype=weight_dtype)
                self._do_logger(f"Loading {role} pipeline...")
                current_pipeline = FaithDiffStableDiffusionXLPipeline.from_single_file(
                    self.config._selected_model_path,
                    torch_dtype=weight_dtype,
                    cache_dir=None,
                    unet=None,
                    add_watermarker=False,
                    disable_mmap=self.config.disable_mmap,
                )
                current_pipeline.unet = unet

            current_pipeline.set_encoder_tile_settings()
            current_pipeline.mode = role
            self._do_logger(f"Loading {role} pipeline... OK!")
            return current_pipeline

        except Exception as e:
            self._do_logger(
                f"Failed to load FaithDiff model for mode '{role}'",
                e,
                level=logging.ERROR,
                exc_info=True,
            )
            return None
        finally:
            del unet
            release_memory()

    def _load_ControlNetTile_model(self, prior_model_settings, is_from_file, role):
        """
        Loads the ControlNetTile model based on the provided settings and role.
        This method initializes the ControlNet model and the corresponding pipeline
        for image processing. It can load the model either from a specified file or
        from a pre-trained model path. The method handles exceptions during the loading
        process and logs relevant information.
        Args:
            prior_model_settings (dict): A dictionary containing settings for the prior model,
                                          including variant and safetensor usage.
            is_from_file (bool): A flag indicating whether to load the model from a file.
            role (str): The role or mode for which the pipeline is being loaded (e.g., 'inference').
        Returns:
            StableDiffusionXLControlNetTileSRPipeline or None:
                Returns the initialized pipeline if successful, or None if loading fails.
        Raises:
            Exception: Logs an error if the model fails to load, including the exception details.
        """
        self._cancel_checker()
        controlnet = None
        current_pipeline = None

        try:
            weight_dtype = self.config.get_weight_dtype(self.config.weight_dtype)

            model_settings = self.model_manager.get_model_settings(ModelType.ControlNet.value, None, "controlnet-union")

            model_path = model_settings["model_path"]
            self._do_logger("Loading ControlNet Union from", model_path)
            controlnet = ControlNetUnionModel.from_pretrained(
                model_path,
                variant=model_settings["variant"],
                torch_dtype=weight_dtype,
                use_safetensors=model_settings["use_safetensors"],
                disable_mmap=self.config.disable_mmap,
            )

            if not is_from_file:
                pipe_params = {
                    "pretrained_model_name_or_path": self.config._selected_model_path,
                    "torch_dtype": weight_dtype,
                    "add_watermarker": False,
                    "controlnet": controlnet,
                    "variant": prior_model_settings["variant"],
                }

                if not prior_model_settings["use_safetensors"]:
                    pipe_params["use_safetensors"] = False

                self._do_logger(f"Loading {role} pipeline...")
                current_pipeline = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(**pipe_params)
            else:
                self._do_logger(f"Loading {role} pipeline...")
                current_pipeline = StableDiffusionXLControlNetTileSRPipeline.from_single_file(
                    self.config._selected_model_path,
                    torch_dtype=weight_dtype,
                    cache_dir=None,
                    controlnet=controlnet,
                    disable_mmap=self.config.disable_mmap,
                    add_watermarker=False,
                )

            current_pipeline.mode = role
            self._do_logger(f"Loading {role} pipeline... OK!")

            return current_pipeline

        except Exception as e:
            self._do_logger(
                f"Failed to load ControlNetTile model for mode '{role}'",
                e,
                level=logging.ERROR,
                exc_info=True,
            )
            return None

        finally:
            del controlnet
            release_memory()

    def _apply_vae_to_pipelines(self, pipelines_to_configure: List[DiffusionPipeline]):
        """
        Applies VAE (Variational AutoEncoder) configuration to the specified diffusion pipelines.
        This method handles the loading and application of VAE models to the provided pipelines.
        It supports both loading from pretrained repositories and single files, with options
        for different model types (AutoencoderKL or AutoencoderTiny).
        Args:
            pipelines_to_configure (List[DiffusionPipeline]): List of diffusion pipelines to
                configure with the specified VAE model.
        Returns:
            bool: True if the operation completes successfully.
        Notes:
            - If a non-default VAE is selected and differs from the current one, it loads and
              applies the new VAE model.
            - When switching to a custom VAE, the original (default) VAE is stored for
              potential restoration.
            - When switching back to default VAE, it restores the original VAE for all pipelines.
            - Supports both safetensors and regular model formats.
            - Configuration includes weight dtype and device settings.
        Raises:
            May raise exceptions during model loading or application (implementation specific).
        """
        self._cancel_checker()

        if self.config.selected_vae_model != "Default" and self.current_vae_model != self.config.selected_vae_model:
            loaded_vae = None
            self.vae_onnx = False

            vae_weight_dtype = self.config.get_weight_dtype(self.config.vae_weight_dtype)
            model_name, is_safetensors = self.config.get_vae_model_name()

            if not is_safetensors:
                vae_settings = self.model_manager.get_model_settings(ModelType.VAE.value, "SD-XL", model_name)
                self.config._selected_vae_model_path = vae_settings["model_path"]
                self._do_logger(
                    "Loading VAE from pretrained repo",
                    self.config._selected_vae_model_path,
                )

                vae_processor = AutoencoderKL if vae_settings["class_name"] == "AutoencoderKL" else AutoencoderTiny
                loaded_vae = vae_processor.from_pretrained(
                    vae_settings["model_path"],
                    torch_dtype=vae_weight_dtype,
                    use_safetensors=vae_settings["use_safetensors"],
                    disable_mmap=self.config.disable_mmap,
                )
            else:
                self.config._selected_vae_model_path = model_name
                self._do_logger("Loading VAE from single file", self.config._selected_vae_model_path)

                config = MODEL_DEFAULT_CONFIG["SD-XL"]["AutoencoderKL"]
                loaded_vae = AutoencoderKL.from_single_file(
                    self.config._selected_vae_model_path,
                    config=config,
                    torch_dtype=vae_weight_dtype,
                    disable_mmap=self.config.disable_mmap,
                )

            if loaded_vae:
                self._do_logger(
                    "Applying loaded VAE to pipeline(s)",
                    f"{len(pipelines_to_configure)} found",
                )
                for pipe in pipelines_to_configure:
                    if pipe is not None:
                        if self.current_vae_model == "Default" and not hasattr(pipe, "original_vae"):
                            pipe.original_vae = pipe.vae
                            pipe.original_vae.to("cpu")
                        pipe.vae = loaded_vae
                self.current_vae_model = self.config.selected_vae_model

            self._do_logger("Loading and applying VAE model... OK!")
        elif self.config.selected_vae_model == "Default" and self.current_vae_model is not None and self.current_vae_model != "Default":
            self._do_logger("Restoring default VAE model for all pipelines.")
            for pipe in pipelines_to_configure:
                if pipe is not None and hasattr(pipe, "original_vae") and pipe.original_vae is not None:
                    pipe.vae = pipe.original_vae
                    pipe.vae.to(self.config.device.value)
                    del pipe.original_vae
            self.current_vae_model = "Default"
            self.vae_onnx = False

        return True

    def _load_grounding_dino_sam_model(self):
        """
        Loads the GroundingDinoSAM2 model on demand.

        This method checks if the GroundingDinoSAM2 model is already loaded. If not, it attempts to load the model using the specified device and model path from the configuration. It logs the loading process and handles any exceptions that may occur during the loading. If the model is already loaded but the device has changed, it moves the model to the new device.

        Raises:
            Exception: If the model fails to load, an exception is raised with an error message.
        """

        if self.grounding_dino_sam2_model is None:
            self._do_logger("Loading GroundingDinoSAM2 model...")
            try:
                self.grounding_dino_sam2_model = GroundingDinoSAM2(
                    device=self.config.device.value,
                    model_path=self.image_tools_model_path,
                )
                self._do_logger("GroundingDinoSAM2 model loaded successfully.")
            except Exception as e:
                self._do_logger(
                    "Failed to load GroundingDinoSAM2 model",
                    e,
                    level=logging.ERROR,
                    exc_info=True,
                )
                raise e
        else:
            if self.grounding_dino_sam2_model.device != self.config.device.value:
                self.grounding_dino_sam2_model.to(self.config.device.value)

    def _load_llava_model(self):
        """
        Load the Llava model and processor if they are not already loaded.
        This method checks if the Llava model is currently loaded. If it is not, it attempts to load the model and its associated processor using the configuration settings provided. The loading process includes determining the appropriate device map, weight data type, and quantization settings based on the configuration.
        If the model is successfully loaded, it logs a success message and returns True. If an error occurs during the loading process, it logs the error, sets the model and processor to None, and returns False. If the model is already loaded, it logs a message indicating that and returns True.
        Returns:
            bool: True if the model is loaded successfully or already loaded, False otherwise.
        """

        if self.llava_model is None:
            try:
                device_map = self.config.get_llava_device_map()
                weight_dtype = self.config.get_weight_dtype(self.config.llava_weight_dtype)
                model_name, _ = self.config.get_model_name(self.config.llava_model)

                model_settings = self.model_manager.get_model_settings(ModelType.Caption.value, None, model_name)

                self._do_logger("Loading Llava model...")

                load_in_4bit = True if self.config.llava_quantization_mode == QuantizationMode.INT4 and self.config.enable_llava_quantization else False
                load_in_8bit = True if self.config.llava_quantization_mode == QuantizationMode.INT8 and self.config.enable_llava_quantization else False

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=weight_dtype,
                    bnb_4bit_use_double_quant=load_in_4bit,
                    llm_int8_enable_fp32_cpu_offload=True,
                )

                self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                    model_settings["model_path"],
                    dtype=weight_dtype,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    use_safetensors=model_settings["use_safetensors"],
                )

                self.llava_processor = AutoProcessor.from_pretrained(model_settings["model_path"], use_fast=True)

                self._do_logger("Llava model loaded successfully.")
                return True
            except Exception as e:
                self._do_logger("Failed to load Llava model", e, level=logging.ERROR, exc_info=True)
                self.llava_model = None
                self.llava_processor = None
                return False
        else:
            self._do_logger("Llava model already loaded.")
            return True

    def _unload_llava_model(self):
        """
        Unload llava model before inference to save VRAM
        """
        if self.llava_model is not None:
            self._do_logger("Unloading llava model...")
            self.llava_model = None
        if self.llava_processor is not None:
            self.llava_processor = None
        release_memory()

    def _check_ipa(self, role):
        """
        Check and (if needed) load IPA embeddings for a given pipeline role.
        This method determines whether IPA embeddings should be applied for the specified
        role ("Upscaler" selects the upscaler pipeline; any other value is treated as
        the restorer pipeline). It performs a series of checks and, when appropriate,
        attempts to load IPA embeddings into the pipeline.
        Parameters:
            role (str): The role to check. Use "Upscaler" to target the upscaler
                pipeline; any other value targets the restorer pipeline.
        Behavior and side effects:
            - Selects the engine corresponding to the role from self.config:
                - self.config.upscaler_engine if role == "Upscaler"
                - self.config.restorer_engine otherwise
            - If the selected engine's value equals the string "None", the function
              returns False (no IPA should be applied).
            - Determines the checkpoint to use for the role and resolves the checkpoint
              model name via self.config.get_checkpoint_model_name(checkpoint). This
              returns a tuple (model_name, is_from_file).
            - If the checkpoint is not from a file (is_from_file is False), the method
              queries prior model settings from self.model_manager.get_model_settings.
              If those settings are not found, the method returns False.
            - If self.ip_model is None and the appropriate pipeline parameters indicate
              IPA embeddings should be applied (respective apply_ipa_embeds flag is True
              for restorer or upscaler), the method:
                - obtains a pipeline manager via self._get_pipeline_by_engine(engine)
                - calls self._load_ipa(prior_model_settings, manager.pipe) to load the
                  IPA embeddings (note: prior_model_settings may be None when the
                  checkpoint was provided from a file)
              This call may initialize or update self.ip_model and the pipeline state.
        Return value:
            - Returns False if the engine is "None" or if prior model settings are
              required but not found.
            - Returns None in the normal (non-failing) path after possibly attempting
              to load IPA (i.e., no explicit True is returned).
        Notes:
            - This function relies on several instance attributes and helper methods:
              self.config, self.model_manager, self._get_pipeline_by_engine, and
              self._load_ipa. The exact meaning of "IPA" is determined by the
              implementation of _load_ipa.
            - The method intentionally distinguishes checkpoints that originate from
              files versus those registered in the model manager; the latter requires
              retrievable prior settings to proceed.
        """

        is_upscaler = role == "Upscaler"

        engine = self.config.upscaler_engine if is_upscaler else self.config.restorer_engine
        if engine.value == "None":
            return False

        checkpoint = self.config.selected_upscaler_checkpoint_model if is_upscaler else self.config.selected_restorer_checkpoint_model
        model_name, is_from_file = self.config.get_checkpoint_model_name(checkpoint)
        prior_model_settings = None
        if not is_from_file:
            prior_model_settings = self.model_manager.get_model_settings(ModelType.Diffusers.value, None, model_name)
            if prior_model_settings is None:
                return False

        if self.ip_model is None and (
            (not is_upscaler and self.config.restorer_pipeline_params.apply_ipa_embeds)
            or (is_upscaler and self.config.upscaler_pipeline_params.apply_ipa_embeds)
        ):
            manager = self._get_pipeline_by_engine(engine)
            self._load_ipa(prior_model_settings, manager.pipe)

    def _load_ipa(self, prior_model_settings, current_pipeline):
        """
        Loads the IP-Adapter model for image conditioning.
        This method initializes and loads the IP-Adapter Plus model, which is used to condition
        the diffusion process on input reference images. It uses the CLIP vision encoder
        for processing the reference images.
        Args:
            prior_model_settings (dict): Dictionary containing settings for the prior model,
                                        including the base model type.
            current_pipeline (object): The current pipeline object containing the UNet model.
        Returns:
            bool: True if loading is successful.
        Raises:
            May raise exceptions related to model loading or file access errors.
        Note:
            - The method loads IP-Adapter Plus specifically configured for Stable Diffusion XL
            - Uses CLIP ViT-H-14 as the image encoder
            - Model is loaded to CPU by default
            - Configured for 16 tokens
        """
        self._cancel_checker()

        self._do_logger("Loading IPAdapter model...")
        self.ip_model = IPLoad()
        model_settings = self.model_manager.get_model_settings("IPAdapter", "SD-XL", "ip-adapter")
        image_encoder_settings = self.model_manager.get_model_settings(ModelType.CLIP.value, prior_model_settings["model_base"], "clip-vit-h-14")
        self.ip_model.load_ip_plus(
            current_pipeline.unet,
            ip_ckpt=os.path.join(model_settings["model_path"], "ip-adapter-plus_sdxl_vit-h.safetensors"),
            device="cpu",
            num_tokens=16,
            image_encoder_path=image_encoder_settings["model_path"],
        )
        self._do_logger("Loading IPAdapter model... OK!")
        return True

    # endregion

    # region pipeline helpers
    def _prepare_pipeline_params(
        self,
        pipe_name,
        pipe_params,
        width_init,
        height_init,
        width_now,
        height_now,
        input_image,
        original_image=None,
    ):
        """
        Prepares the parameters for the specified pipeline based on the provided inputs.
        Args:
            pipe_name (str): The name of the pipeline to prepare parameters for.
            pipe_params (object): An object containing various parameters for the pipeline.
            width_init (int): The initial width of the image.
            height_init (int): The initial height of the image.
            width_now (int): The current width of the image.
            height_now (int): The current height of the image.
            input_image (any): The input image to be processed by the pipeline.
            original_image (any, optional): The original image, used in certain pipelines. Defaults to None.
        Returns:
            dict: A dictionary containing the prepared parameters for the specified pipeline.
        """

        original_callback = pipe_params.callback_on_step_end

        def cancel_check_callback(*args, **kwargs):
            self._cancel_checker()
            # If the callback is a single callable, invoke it directly.
            if callable(original_callback):
                try:
                    return original_callback(*args, **kwargs)
                except Exception:
                    return kwargs

            # If the callback is an iterable (e.g., a tuple like (None,) or a list of callbacks),
            # find the first callable and invoke it.
            try:
                for cb in original_callback:
                    if callable(cb):
                        try:
                            return cb(*args, **kwargs)
                        except Exception:
                            return kwargs
            except TypeError:
                # original_callback is neither callable nor iterable; ignore it.
                pass

            return kwargs

        if pipe_name == SUPIR_PIPELINE_NAME:
            pipeline_params = {
                "image": input_image,
                "strength": pipe_params.strength,
                "prompt": pipe_params.prompt[0],
                "negative_prompt": pipe_params.negative_prompt,
                "height": height_now,
                "width": width_now,
                "num_inference_steps": pipe_params.num_steps,
                "start_point": pipe_params.start_point.value,
                "guidance_scale": pipe_params.guidance_scale,
                "guidance_rescale": pipe_params.guidance_rescale,
                "use_linear_CFG": pipe_params.use_linear_CFG,
                "reverse_linear_CFG": pipe_params.reverse_linear_CFG,
                "guidance_scale_start": pipe_params.guidance_scale_start,
                "num_images_per_prompt": pipe_params.num_images,
                "restoration_scale": pipe_params.restoration_scale,
                "s_churn": pipe_params.s_churn,
                "s_noise": pipe_params.s_noise,
                "controlnet_conditioning_scale": pipe_params.controlnet_conditioning_scale,
                "use_linear_control_scale": pipe_params.use_linear_control_scale,
                "reverse_linear_control_scale": pipe_params.reverse_linear_control_scale,
                "control_scale_start": pipe_params.control_scale_start,
                "use_linear_PAG": pipe_params.use_linear_PAG,
                "reverse_linear_PAG": pipe_params.reverse_linear_PAG,
                "pag_scale": float(pipe_params.pag_scale),
                "pag_scale_start": float(pipe_params.pag_scale_start),
                "pag_adaptive_scale": float(pipe_params.pag_adaptive_scale),
                "use_lpw_prompt": pipe_params.use_lpw_prompt,
                "tile_size": pipe_params.tile_size,
                "tile_overlap": pipe_params.tile_overlap,
                "original_size": (height_now, width_now),
                "target_size": (height_now, width_now),
                "aesthetic_score": pipe_params.aesthetic_score,
                "negative_aesthetic_score": pipe_params.negative_aesthetic_score,
                "clip_skip": (None if pipe_params.clip_skip == 0 else pipe_params.clip_skip),
                "zero_sft_injection_flags": pipe_params.zero_sft_injection_flags,
                "zero_sft_injection_configs": pipe_params.zero_sft_injection_configs,
                "callback_on_step_end": cancel_check_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
                "execution_device": self.config.get_device(self.config.device),
                "cross_attention_kwargs": pipe_params.cross_attention_kwargs,
            }
        elif pipe_name == FAITHDIFF_PIPELINE_NAME:
            pipeline_params = {
                "image": input_image,
                "strength": pipe_params.strength,
                "prompt": pipe_params.prompt[0],
                "negative_prompt": pipe_params.negative_prompt,
                "height": height_now,
                "width": width_now,
                "num_inference_steps": pipe_params.num_steps,
                "start_point": pipe_params.start_point.value,
                "guidance_scale": pipe_params.guidance_scale,
                "guidance_rescale": pipe_params.guidance_rescale,
                "num_images_per_prompt": pipe_params.num_images,
                "image_prompt_embeds": self.image_prompt_embeds,
                "uncond_image_prompt_embeds": self.uncond_image_prompt_embeds,
                "s_churn": pipe_params.s_churn,
                "s_noise": pipe_params.s_noise,
                "conditioning_hint_scale": pipe_params.controlnet_conditioning_scale,
                "use_linear_conditioning_hint_scale": pipe_params.use_linear_control_scale,
                "reverse_linear_conditioning_hint_scale": pipe_params.reverse_linear_control_scale,
                "conditioning_hint_scale_start": pipe_params.control_scale_start,
                "use_linear_PAG": pipe_params.use_linear_PAG,
                "reverse_linear_PAG": pipe_params.reverse_linear_PAG,
                "pag_scale": float(pipe_params.pag_scale),
                "pag_scale_start": float(pipe_params.pag_scale_start),
                "pag_adaptive_scale": float(pipe_params.pag_adaptive_scale),
                "use_lpw_prompt": pipe_params.use_lpw_prompt,
                "tile_size": pipe_params.tile_size,
                "tile_overlap": pipe_params.tile_overlap,
                "original_size": (height_now, width_now),
                "target_size": (height_now, width_now),
                "aesthetic_score": pipe_params.aesthetic_score,
                "negative_aesthetic_score": pipe_params.negative_aesthetic_score,
                "clip_skip": (None if pipe_params.clip_skip == 0 else pipe_params.clip_skip),
                "callback_on_step_end": cancel_check_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
                "execution_device": self.config.get_device(self.config.device),
                "cross_attention_kwargs": pipe_params.cross_attention_kwargs,
            }
        elif pipe_name == CONTROLNET_TILE_PIPELINE_NAME:
            pipeline_params = {
                "image": input_image,
                "control_image": original_image,
                "control_mode": [6],
                "strength": pipe_params.strength,
                "prompt": pipe_params.prompt[0],
                "negative_prompt": pipe_params.negative_prompt,
                "height": height_now,
                "width": width_now,
                "num_inference_steps": pipe_params.num_steps,
                "guidance_scale": pipe_params.guidance_scale,
                "guidance_rescale": pipe_params.guidance_rescale,
                "num_images_per_prompt": pipe_params.num_images,
                "controlnet_conditioning_scale": pipe_params.controlnet_conditioning_scale,
                "tile_size": pipe_params.tile_size,
                "tile_overlap": pipe_params.tile_overlap,
                "original_size": (height_init, width_init),
                "target_size": (height_now, width_now),
                "aesthetic_score": pipe_params.aesthetic_score,
                "negative_aesthetic_score": pipe_params.negative_aesthetic_score,
                "clip_skip": (None if pipe_params.clip_skip == 0 else pipe_params.clip_skip),
                "callback_on_step_end": cancel_check_callback,
                "callback_on_step_end_tensor_inputs": ["latents"],
                "tile_weighting_method": pipe_params.tile_weighting_method.value,
                "tile_gaussian_sigma": pipe_params.tile_gaussian_sigma,
                "execution_device": self.config.get_device(self.config.device),
                "cross_attention_kwargs": pipe_params.cross_attention_kwargs,
            }
        return pipeline_params

    def _prepare_prompt(
        self,
        prompt: str | None,
        prompt_2: str | None,
        invert_prompts: bool,
        apply_prompt_2: bool,
    ) -> List[str]:
        """
        Prepares and combines the main prompt and a secondary prompt based on inversion
        and application flags.

        Args:
            prompt: The main prompt string.
            prompt_2: The secondary prompt string.
            invert_prompts: If True, `prompt_2` is placed before `prompt`.
            apply_prompt_2: If True, `prompt_2` is concatenated.

        Returns:
            A list containing the final combined prompt string.
        """
        # 1. Normalize inputs to ensure they are strings.
        prompt = prompt or ""
        prompt_2 = prompt_2 or ""

        # 2. Determine if the secondary prompt should be used at all.
        # It's only used if the flag is true AND the prompt is not empty.
        use_secondary_prompt = apply_prompt_2 and prompt_2.strip() != ""

        # 3. Handle the simple case first: no secondary prompt is used.
        if not use_secondary_prompt:
            return [prompt]

        # 4. Handle the concatenation cases, now that we know prompt_2 is valid.
        if invert_prompts:
            # If the main prompt is empty, don't add an extra space.
            final_prompt = f"{prompt_2} {prompt}".strip() if prompt else prompt_2
        else:
            # If the main prompt is empty, don't add an extra space.
            final_prompt = f"{prompt} {prompt_2}".strip() if prompt else prompt_2

        return [final_prompt]

    def _prepare_ipa(self, input_image):
        """
        Prepares the image prompt embeddings for the IPA model.
        This method checks if the IPA model is loaded and moves the model to the appropriate device if necessary.
        It resizes the input image to (512, 512) and retrieves the image embeddings. The embeddings are then
        repeated and reshaped to accommodate multiple samples. If the model was not originally on the CPU,
        it is moved back to the CPU to release memory.
        Args:
            input_image (PIL.Image): The input image to be processed.
        Returns:
            bool: True if the preparation was successful, False if the IPA model was not loaded.
        """

        self._cancel_checker()

        if self.ip_model is None:
            self._do_logger("IPA Model was not loaded!", level=logging.ERROR)
            return False

        device = self.config.get_device(self.config.device)

        if self.ip_model.image_encoder.device.type != device.type:
            self.ip_model.image_encoder.to(device)
            self.ip_model.image_proj_model.to(device)

        num_samples = 1
        self.image_prompt_embeds, self.uncond_image_prompt_embeds = self.ip_model.get_image_embeds(input_image.resize((512, 512)))
        self.image_prompt_embeds = self.image_prompt_embeds.to(self.config.get_weight_dtype(self.config.weight_dtype))
        self.uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.to(self.config.get_weight_dtype(self.config.weight_dtype))
        bs_embed, seq_len, _ = self.image_prompt_embeds.shape
        self.image_prompt_embeds = self.image_prompt_embeds.repeat(1, num_samples, 1)
        self.image_prompt_embeds = self.image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        self.uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        self.uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        if self.ip_model.image_encoder.device.type != "cpu":
            self.ip_model.image_encoder.to("cpu")
            self.ip_model.image_proj_model.to("cpu")
            release_memory()

        return True

    def _prepare_scheduler(self, role: str):
        """
        Configures the scheduler for the specified phase (Restorer or Upscaler) in the active pipeline.
        This method sets the appropriate scheduler and its configuration just before an inference phase begins.
        It checks if there is an active pipeline for the given role and retrieves the corresponding sampler name
        and configuration. If no active pipeline is found, a warning is logged, and the scheduler setup is skipped.
        Args:
            role (str): The role of the pipeline phase, either "Restorer" or "Upscaler".
        Returns:
            None: This method does not return a value.
        """

        is_upscaler = role == "Upscaler"
        pipe = self._get_active_upscaler_pipe() if is_upscaler else self._get_active_restorer_pipe()

        if pipe is None:
            self._do_logger(
                f"No active pipe found for {role}, skipping scheduler setup.",
                level=logging.WARNING,
            )
            return

        # Get the correct sampler name and config based on the role
        sampler_name = self.config.selected_upscaler_sampler.value if is_upscaler else self.config.selected_restorer_sampler.value
        sampler_config = self.config.upscaler_sampler_config.to_dict() if is_upscaler else self.config.restorer_sampler_config.to_dict()

        # Load and set the scheduler on the pipeline object
        pipe.scheduler = load_scheduler(pipe, pipe.__class__.__name__, sampler_name, **sampler_config)

    def _apply_face_restoration_mask(self, original_lq_image: Image.Image, restored_image: Image.Image) -> Image.Image:
        """
        Orchestrates the full selective face restoration process by generating masks
        and then compositing the images.

        Args:
            original_lq_image: The initial low-quality image.
            restored_image: The fully restored image from the pipeline.

        Returns:
            A composite PIL Image, or the fully restored image as a fallback.
        """
        self._cancel_checker()

        # Generate the necessary masks from the original image.
        composition_mask, background_mask = self.generate_prompt_mask(original_lq_image, self.config.mask_prompt)

        # If mask generation fails or finds nothing, return the fully restored image.
        if composition_mask is None or background_mask is None:
            self._do_logger(
                "Mask generation failed or no objects found. Returning full restoration.",
                level=logging.WARNING,
            )
            return restored_image

        # Use the generated masks to composite the final image.
        return self._composite_with_mask(
            original_lq_image=original_lq_image,
            restored_image=restored_image,
            composition_mask=composition_mask,
            background_mask=background_mask,
        )

    def _composite_with_mask(
        self,
        original_lq_image: Image.Image,
        restored_image: Image.Image,
        composition_mask: Image.Image,
        background_mask: Image.Image,
    ) -> Image.Image:
        """
        Composites a restored image onto an original image using pre-generated masks.

        Args:
            original_lq_image: The initial low-quality image.
            restored_image: The fully restored image (must be the same size as original).
            composition_mask: A feathered mask defining the area to paste from the restored image.
            background_mask: An inverted mask defining the area to keep from the original image.

        Returns:
            The final composite PIL Image.
        """
        self._cancel_checker()
        self._do_logger("--- Compositing Final Image with Mask ---")

        # Create a transparent RGBA version of the original image
        original_rgba = original_lq_image.convert("RGBA")

        # Create a transparent canvas
        transparent_background = Image.new("RGBA", original_rgba.size, (0, 0, 0, 0))

        # Use the background_mask to punch a "hole" in the original image where the face is
        background_image = Image.composite(original_rgba, transparent_background, background_mask)

        # Paste the restored image into the hole, using the feathered composition_mask for smooth blending
        background_image.paste(restored_image.convert("RGBA"), (0, 0), composition_mask)

        self._do_logger("Composition complete.")
        return background_image.convert("RGB")

    def _save_pass_image(
        self,
        initial_image,
        image_index,
        upscaled_dir,
        base_seed,
        image_sub_index,
        target_scale_factor,
        final_image,
    ):
        """
        Save an upscaled pass image to disk, optionally apply color correction, embed metadata, and return the final image.
        This helper constructs a filename for the upscaled image using the given seed and rank, logs progress, optionally applies a color-fixing step using the original (or resized) initial image as a color reference, writes the image to disk with embedded metadata, and performs a cancellation check before returning the resulting PIL Image.
        Parameters
        ----------
        initial_image : PIL.Image.Image
            The original input image for this pass. Used as a color reference when color-fix is enabled.
        image_index : int
            Zero-based index of the current image in the batch. Used to build the output image rank/name.
        upscaled_dir : str
            Directory path where the upscaled image file will be written.
        base_seed : int
            Seed used to derive the saved image filename via self.get_image_name(...).
        image_sub_index : int
            Zero-based sub-index for the image (e.g., when multiple variants per seed are generated).
        target_scale_factor : float
            Scale factor that was applied to produce the upscaled image; used in the filename (e.g., '2x', '4x').
        final_image : PIL.Image.Image
            The upscaled image to be saved and (optionally) color-corrected. This object may be replaced by a color-fixed copy.
        Returns
        -------
        PIL.Image.Image
            The final image object that was saved (possibly after color correction). This is the image that has metadata embedded by save_image_with_metadata.
        Side effects
        ------------
        - Writes an image file to disk at a path built from upscaled_dir and a name derived from base_seed, image_index and image_sub_index.
        - Calls self._do_logger(...) to report progress and actions.
        - May call wavelet_color_fix(...) or adain_color_fix(...) to modify the final image if color-fix is enabled in the configuration.
        - Embeds metadata in the saved file via save_image_with_metadata(...).
        - Calls self._cancel_checker() before returning to allow cooperative cancellation.
        Behavior notes
        --------------
        - If the configured color-fix mode is enabled, the function uses a copy of initial_image as the color reference. If the reference size differs from final_image, the reference is resized to match final_image.size before applying the color fix.
        - Only the configured color fix modes handled here are Wavelet and Adain; other modes will not alter the image in this function.
        - The saved filename will include the image rank (one-based human-readable index and sub-index) and the integer part of target_scale_factor followed by 'x'.
        - Errors when writing the file or during color-fixing (e.g., I/O errors, unexpected image modes) will propagate to the caller.
        """

        image_rank = f"{image_index + 1}_{image_sub_index + 1}"
        image_name = f"{self._get_image_name(base_seed, f'{image_rank}_{int(target_scale_factor)}x')}"
        upscaled_image_file = os.path.join(upscaled_dir, f"{image_name}{self.config.save_image_format}")
        self._do_logger("Saving final upscaled image", upscaled_image_file)

        # apply color fix if enabled
        if self.config.restorer_pipeline_params.color_fix_mode != ColorFix.Nothing:
            image_color_reference = initial_image.copy()
            if image_color_reference.width != final_image.width or image_color_reference.height != final_image.height:
                image_color_reference = image_color_reference.resize(final_image.size)

            if self.config.upscaler_pipeline_params.color_fix_mode == ColorFix.Wavelet:
                self._do_logger(
                    f"Applying {self.config.upscaler_pipeline_params.color_fix_mode.value} color fix to image {image_rank}",
                    upscaled_image_file,
                )
                final_image = wavelet_color_fix(final_image, image_color_reference)
            elif self.config.upscaler_pipeline_params.color_fix_mode == ColorFix.Adain:
                self._do_logger(
                    f"Applying {self.config.upscaler_pipeline_params.color_fix_mode.value} color fix to image {image_rank}",
                    upscaled_image_file,
                )
                final_image = adain_color_fix(final_image, image_color_reference)
        if not self.config.running_on_spaces:
            final_image = save_image_with_metadata(final_image, upscaled_image_file, self.metadata)
        self._cancel_checker()
        return final_image

    def _get_active_restorer_pipe(self) -> Optional[DiffusionPipeline]:
        """Gets the currently active restorer pipeline object."""
        if self.active_restorer_engine and self.active_restorer_engine.name in self.pipeline_managers:
            return self.pipeline_managers[self.active_restorer_engine.name].pipe
        return None

    def _get_active_upscaler_pipe(self) -> Optional[DiffusionPipeline]:
        """Gets the currently active upscaler pipeline object."""
        if self.active_upscaler_engine and self.active_upscaler_engine.name in self.pipeline_managers:
            # Handle reuse case
            if self.active_upscaler_engine == self.active_restorer_engine:
                return self._get_active_restorer_pipe()
            return self.pipeline_managers[self.active_upscaler_engine.name].pipe
        return None

    def _get_pipeline_by_engine(self, engine: Union[RestorerEngine, UpscalerEngine]):
        """Helper to get a pipeline manager from an engine enum."""
        engine_name = engine.name
        return self.pipeline_managers.get(engine_name)

    def _ensure_pipeline_is_active(self, engine: Union[RestorerEngine, UpscalerEngine]):
        """Ensures the specified engine's pipeline is loaded and on the GPU."""
        manager = self._get_pipeline_by_engine(engine)
        if manager.pipe.device.type != self.config.device.value and not self.config.enable_cpu_offload:
            self._do_logger("Activating pipeline", f"'{engine.name}'")
            manager.restore_pipe(self.config.device.value)
        elif self.config.enable_cpu_offload and not hasattr(manager.pipe, "_offload_device"):
            self.apply_optimizations()

    def _ensure_pipeline_is_inactive(self, engine: Union[RestorerEngine, UpscalerEngine]):
        """Ensures the specified engine's pipeline is on the CPU."""
        manager = self._get_pipeline_by_engine(engine)
        if manager and manager.is_active():
            self._do_logger("Putting pipeline in standby", f"'{engine.name}'")
            manager.unload_pipe()

    def _cleanup_unused_pipelines(self):
        """Removes pipeline managers that are no longer needed based on current engine selection."""
        active_engine_names = {e.name for e in [self.config.restorer_engine, self.config.upscaler_engine] if e and e.value != "None"}

        for engine_name in list(self.pipeline_managers.keys()):
            if engine_name not in active_engine_names:
                self._do_logger("Cleaning up unused pipeline manager", engine_name)
                if self.active_restorer_engine and self.active_restorer_engine.name == engine_name:
                    self.active_restorer_engine = None
                    self.current_prior_restorer_model = None
                    self.current_restorer_supir_model = None
                if self.active_upscaler_engine and self.active_upscaler_engine.name == engine_name:
                    self.active_upscaler_engine = None
                    self.current_prior_upscaler_model = None
                    self.current_upscaler_supir_model = None

                del self.pipeline_managers[engine_name]

    def _get_prompt_for_image(self, file_path_or_pil, is_dir, is_loaded_image):
        """
        Return a cleaned prompt extracted from a JSON caption file for an image.
        Parameters:
            file_path_or_pil (Path | PIL.Image.Image): image path (Path) or loaded image object.
            is_dir (bool): True if the image is in the configured image directory.
            is_loaded_image (bool): True if the image is already loaded (no prompt needed).
        Behavior:
            - If the image is already loaded or not a directory entry, returns None.
            - Looks for a JSON caption file named "<stem>.json" under self.config.image_path; logs a warning and returns None if missing.
            - Reads the "caption" field; returns an empty string if caption is empty.
            - Capitalizes the first word and ensures the prompt ends with ".", "?" or "!".
        Returns:
            str | None: formatted prompt string, empty string when caption exists but is empty, or None when no prompt should be used.
        """
        if is_loaded_image or not is_dir:
            return None

        caption_file_path = Path(self.config.image_path) / (file_path_or_pil.stem + ".json")
        if not caption_file_path.exists():
            logger.warning(f"Caption file {caption_file_path} not found, skipping file...")
            return None

        self._do_logger("Using caption file", caption_file_path)
        with open(caption_file_path, "r") as f:
            json_file = json.load(f)

        init_text: str = json_file.get("caption", "")
        if not init_text:
            return ""

        words = init_text.split()
        if words:
            words[0] = words[0].capitalize()
        prompt = " ".join(words)
        if prompt and not prompt.endswith((".", "?", "!")):
            prompt += "."
        return prompt

    def _get_image_name(self, seed, rank):
        """
        Generates a unique image name based on timestamp, seed and rank.

        Args:
            seed (int): Random seed value used
            rank (int): Rank number for the image

        Returns:
            str: Generated image name in format 'YYYY_MM_DD_HH_MM_seed_rank'
        """
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        image_name = f"{current_time}_{seed}_{rank}"
        return image_name

    def _restart_cpu_offload(self):
        """
        Restart CPU offload for all managed pipelines.
        This method attempts to reinitialize CPU offloading for every pipeline
        managed by this instance. For each pipeline it:
        - Logs that a restart is starting.
        - Cancels any active checker related to the pipeline.
        - If the pipeline exposes an `enable_model_cpu_offload` capability:
            - Calls `optionally_disable_offloading` to ensure offloading is disabled if appropriate.
            - Calls `release_memory()` to free any held resources.
            - Calls `pipe.reset_device_map()` to clear/reset the device placement map.
            - Calls `pipe.enable_model_cpu_offload()` to re-enable CPU offload.
        Side effects:
        - Mutates pipeline state (device maps, offload state) and releases process memory.
        - Emits log messages via the instance logger.
        Concurrency and performance:
        - The operation may be blocking and relatively expensive; call from a safe/single-threaded
          context or ensure callers handle concurrency appropriately.
        Exceptions:
        - Exceptions raised by any of the invoked helpers or pipeline methods are not caught
          here and will propagate to the caller.
        Returns:
        - None
        """
        self._do_logger("Restarting CPU Offload...")
        from sup_toolbox.pipeline_util import optionally_disable_offloading

        all_managed_pipes = [mgr.pipe for mgr in self.pipeline_managers.values()]
        for pipe in all_managed_pipes:
            self._cancel_checker()
            if pipe is not None and hasattr(pipe, "enable_model_cpu_offload"):
                optionally_disable_offloading(pipe)
                release_memory()
                pipe.reset_device_map()
                pipe.enable_model_cpu_offload()

    def _cancel_checker(self):
        """Checks the provided cancel event and raises an exception if set."""
        if self.cancel_event.is_set():
            raise PipelineCancelationRequested("Process was canceled by the user.")

    def _do_logger(self, message: str, value: Any = None, level=logging.INFO, exc_info=None):
        """Logs a formatted message for professional, aligned console output."""
        if value is not None:
            # Reserves 40 characters for the message description, left-aligned.
            log_message = f"{message:<40}: {value}"
        else:
            log_message = message

        logger.log(level, log_message, exc_info=exc_info)
        if self.log_callback:
            self.log_callback(advance=0)

    # endregion
