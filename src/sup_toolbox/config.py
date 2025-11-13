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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch

from sup_toolbox.modules.diffusers_local.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from sup_toolbox.modules.diffusers_local.schedulers.scheduling_euler_discrete import (
    EulerDiscreteScheduler,
)
from sup_toolbox.modules.diffusers_local.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from sup_toolbox.modules.SUPIR.pipeline_supir_stable_diffusion_xl import (
    InjectionConfigs,
    InjectionFlags,
)
from sup_toolbox.modules.SUPIR.schedulers.scheduling_dpm_solver_restore import (
    DPMSolverMultistepRestoreScheduler,
)
from sup_toolbox.modules.SUPIR.schedulers.scheduling_euler_discrete_restore import (
    EulerDiscreteRestoreScheduler,
)
from sup_toolbox.modules.SUPIR.schedulers.scheduling_unipc_restore import (
    UniPCRestoreScheduler,
)

from .enums import (
    ColorFix,
    ImageSizeFixMode,
    MemoryAttention,
    QuantizationMethod,
    QuantizationMode,
    RestorerEngine,
    RuntimeDevice,
    Sampler,
    StartPoint,
    SUPIRModel,
    UpscalerEngine,
    UpscalingMode,
    WeightDtype,
    WeightingMethod,
)


SAMPLERS_OTHERS = {
    "Euler": (
        EulerDiscreteScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace"},
    ),
    "DPM++ 1S": (
        DPMSolverMultistepScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace", "solver_order": 1},
    ),
    "DPM++ 1S Karras": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "solver_order": 1,
            "use_karras_sigmas": True,
        },
    ),
    "DPM++ 2S": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
        },
    ),
    "DPM++ 2S Karras": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": True,
        },
    ),
    "DPM++ 2M": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
        },
    ),
    "DPM++ 2M Karras": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": True,
        },
    ),
    "DPM++ 2M SDE": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 2M SDE Karras": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": True,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 3M": (
        DPMSolverMultistepScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace", "solver_order": 3},
    ),
    "DPM++ 3M SDE": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "solver_order": 3,
            "use_karras_sigmas": False,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 3M Karras": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "solver_order": 3,
            "use_karras_sigmas": True,
        },
    ),
    "UniPC": (
        UniPCMultistepScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace"},
    ),
    "DPM++ 2M Lu": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_lu_lambdas": True,
        },
    ),
    "DPM++ 2M Ef": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "euler_at_final": True,
        },
    ),
    "DPM++ 2M SDE Lu": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_lu_lambdas": True,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 2M SDE Ef": (
        DPMSolverMultistepScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "algorithm_type": "sde-dpmsolver++",
            "euler_at_final": True,
        },
    ),
}

SAMPLERS_SUPIR = {
    "Euler": (
        EulerDiscreteRestoreScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace"},
    ),
    "DPM++ 1S": (
        DPMSolverMultistepRestoreScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace", "solver_order": 1},
    ),
    "DPM++ 1S Karras": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "solver_order": 1,
            "use_karras_sigmas": True,
        },
    ),
    "DPM++ 2S": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
        },
    ),
    "DPM++ 2S Karras": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": True,
        },
    ),
    "DPM++ 2M": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
        },
    ),
    "DPM++ 2M Karras": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": True,
        },
    ),
    "DPM++ 2M SDE": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 2M SDE Karras": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": True,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 3M": (
        DPMSolverMultistepRestoreScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace", "solver_order": 3},
    ),
    "DPM++ 3M SDE": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "solver_order": 3,
            "use_karras_sigmas": False,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 3M Karras": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "solver_order": 3,
            "use_karras_sigmas": True,
        },
    ),
    "UniPC": (
        UniPCRestoreScheduler,
        {"beta_schedule": "linear", "timestep_spacing": "linspace"},
    ),
    "DPM++ 2M Lu": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_lu_lambdas": True,
        },
    ),
    "DPM++ 2M Ef": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "euler_at_final": True,
        },
    ),
    "DPM++ 2M SDE Lu": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "use_lu_lambdas": True,
            "algorithm_type": "sde-dpmsolver++",
        },
    ),
    "DPM++ 2M SDE Ef": (
        DPMSolverMultistepRestoreScheduler,
        {
            "beta_schedule": "linear",
            "timestep_spacing": "linspace",
            "algorithm_type": "sde-dpmsolver++",
            "euler_at_final": True,
        },
    ),
}

SUPIR_MODELS = {
    "Quality": {"model_name": "SUPIR_Q"},
    "Fidelity": {"model_name": "SUPIR_F"},
}

PAG_LAYERS = {
    "down.blocks.1": "down.blocks.1",
    "down.blocks.1.attn.0": "down.blocks.1.attentions.0",
    "down.blocks.1.attn.1": "down.blocks.1.attentions.1",
    "down.blocks.2": "down.blocks.2",
    "down.blocks.2.attn.0": "down.blocks.2.attentions.0",
    "down.blocks.2.attn.1": "down.blocks.2.attentions.1",
    "mid": "mid",
    "up.blocks.0": "up.blocks.0",
    "up.blocks.0.attn.0": "up.blocks.0.attentions.0",
    "up.blocks.0.attn.1": "up.blocks.0.attentions.1",
    "up.blocks.0.attn.2": "up.blocks.0.attentions.2",
    "up.blocks.1": "up.blocks.1",
    "up.blocks.1.attn.0": "up.blocks.1.attentions.0",
    "up.blocks.1.attn.1": "up.blocks.1.attentions.1",
    "up.blocks.1.attn.2": "up.blocks.1.attentions.2",  #
}

UPSCALER_SCALES = {"1x": 1, "2x": 2, "4x": 4, "8x": 8, "10x": 10}

MODEL_DEFAULT_CONFIG = {"SD-XL": {"AutoencoderKL": "src/configs/SDXL/vae_config.json"}}

SUPIR_PIPELINE_NAME = "SUPIRStableDiffusionXLPipeline"

FAITHDIFF_PIPELINE_NAME = "FaithDiffStableDiffusionXLPipeline"

CONTROLNET_TILE_PIPELINE_NAME = "StableDiffusionXLControlNetTileSRPipeline"

RESTORER = "Restorer"

UPSCALER = "Upscaler"

VALID_SUPIR_PARAMS = {
    "use_lpw_prompt",
    "apply_prompt_2",
    "supir_model",
    "restoration_scale",
    "s_churn",
    "s_noise",
    "start_point",
    "strength",
    "seed",
    "randomize_seed",
    "num_steps",
    "num_images",
    "guidance_scale",
    "guidance_rescale",
    "image_size_fix_mode",
    "tile_size",
    "upscaling_mode",
    "upscale_factor",
    "cfg_decay_rate",
    "strength_decay_rate",
    "use_linear_CFG",
    "reverse_linear_CFG",
    "guidance_scale_start",
    "controlnet_conditioning_scale",
    "use_linear_control_scale",
    "reverse_linear_control_scale",
    "control_scale_start",
    "enable_PAG",
    "pag_scale",
    "use_linear_PAG",
    "reverse_linear_PAG",
    "pag_scale_start",
    "pag_layers",
    "color_fix_mode",
    "zero_sft_injection_configs",
    "zero_sft_injection_flags",
}

VALID_FAITHDIFF_PARAMS = {
    "use_lpw_prompt",
    "apply_prompt_2",
    "invert_prompts",
    "apply_ipa_embeds",
    "s_churn",
    "s_noise",
    "start_point",
    "strength",
    "seed",
    "randomize_seed",
    "num_steps",
    "num_images",
    "guidance_scale",
    "guidance_rescale",
    "image_size_fix_mode",
    "tile_size",
    "upscaling_mode",
    "upscale_factor",
    "cfg_decay_rate",
    "strength_decay_rate",
    "controlnet_conditioning_scale",
    "use_linear_control_scale",
    "reverse_linear_control_scale",
    "control_scale_start",
    "enable_PAG",
    "pag_scale",
    "use_linear_PAG",
    "reverse_linear_PAG",
    "pag_scale_start",
    "pag_layers",
    "color_fix_mode",
}

VALID_CONTROLNETTILE_PARAMS = {
    "apply_prompt_2",
    "controlnet_conditioning_scale",
    "tile_overlap",
    "tile_weighting_method",
    "tile_gaussian_sigma",
    "strength",
    "seed",
    "randomize_seed",
    "num_steps",
    "num_images",
    "guidance_scale",
    "guidance_rescale",
    "image_size_fix_mode",
    "tile_size",
    "upscaling_mode",
    "upscale_factor",
    "cfg_decay_rate",
    "strength_decay_rate",
    "color_fix_mode",
}

# Combined dictionary of attribute names to friendly labels.
PIPELINE_PARAM_LABELS = {
    "aesthetic_score": "Aesthetic Score",
    "apply_ipa_embeds": "Apply IPA Embeds",
    "apply_prompt_2": "Apply Prompt 2",
    "cfg_decay_rate": "CFG Decay Rate",
    "color_fix_mode": "Color Fix Mode",
    "control_scale_start": "Control Scale Start",
    "controlnet_conditioning_scale": "ControlNet Scale",
    "enable_PAG": "Enable PAG",
    "seed": "Seed",
    "guidance_rescale": "Guidance Rescale",
    "guidance_scale": "Guidance Scale (CFG)",
    "guidance_scale_start": "Guidance Scale Start",
    "image_size_fix_mode": "Image Size Fix Mode",
    "invert_prompts": "Invert Prompts",
    "negative_aesthetic_score": "Negative Aesthetic Score",
    "num_images": "Number of Images",
    "num_steps": "Inference Steps",
    "pag_scale": "PAG Scale",
    "pag_scale_start": "PAG Scale Start",
    "pag_layers": "PAG Layers",
    "randomize_seed": "Randomize Seed",
    "restoration_scale": "Restoration Scale",
    "reverse_linear_CFG": "Reverse Linear CFG",
    "reverse_linear_PAG": "Reverse Linear PAG",
    "reverse_linear_control_scale": "Reverse Linear Control Scale",
    "s_churn": "S Churn",
    "s_noise": "S Noise",
    "start_point": "Start Point",
    "strength": "Denoising Strength",
    "strength_decay_rate": "Strength Decay Rate",
    "supir_model": "Model Type",
    "tile_gaussian_sigma": "Tile Guassian Sigma",
    "tile_overlap": "Tile Overlap",
    "tile_size": "Tile Size",
    "tile_weighting_method": "Tile Weighting Method",
    "upscale_factor": "Upscale Factor",
    "upscaling_mode": "Upscaling Mode",
    "use_linear_CFG": "Use Linear CFG",
    "use_linear_PAG": "Use Linear PAG",
    "use_linear_control_scale": "Use Linear Control Scale",
    "use_lpw_prompt": "Use Long prompt weighted",
    "zero_sft_injection_configs": "Zero Sft Injection Configs",
    "zero_sft_injection_flags": "Zero Sft Injection Flags",
}

SUPIR_ADVANCED_LABELS = {
    "sft_post_mid": "SFT Post Mid",
    "sft_up_block_0_stage0": "SFT Up Block 0 Stage 0",
    "sft_up_block_0_stage1": "SFT Up Block 0 Stage 1",
    "sft_up_block_0_stage2": "SFT Up Block 0 Stage 2",
    "sft_up_block_1_stage0": "SFT Up Block 1 Stage 0",
    "sft_up_block_1_stage1": "SFT Up Block 1 Stage 1",
    "sft_up_block_1_stage2": "SFT Up Block 1 Stage 2",
    "sft_up_block_2_stage0": "SFT Up Block 2 Stage 0",
    "sft_up_block_2_stage1": "SFT Up Block 2 Stage 1",
    "sft_up_block_2_stage2": "SFT Up Block 2 Stage 2",
    "cross_up_block_0_stage1": "Cross Up Block 0 Stage 1",
    "cross_up_block_0_stage2": "Cross Up Block 0 Stage 2",
    "cross_up_block_1_stage1": "Cross Up Block 1 Stage 1",
    "cross_up_block_1_stage2": "Cross Up Block 1 Stage 2",
}


@dataclass
class DefaultSettings:
    model_config_path: str = ""
    cache_dir: str = "models"  #
    checkpoints_dir: str = "models/Checkpoints"  #
    restorer_dir: str = "models/Restorer"  #
    vae_dir: str = "models/VAE"
    lora_dir: str = "loras"  #
    output_dir = "outputs"  #
    default_vae: str = "sdxl-vae-fp16-fix"  #
    llava_model: str = "llava-1.5-7b-hf"  #
    save_image_format: str = ".png"  #
    weight_dtype: WeightDtype = WeightDtype.Bfloat16  #
    vae_weight_dtype: WeightDtype = WeightDtype.Float16  #
    device: RuntimeDevice = RuntimeDevice.CUDA  #
    generator_device: RuntimeDevice = RuntimeDevice.CPU  #
    enable_cpu_offload: bool = True  #
    enable_vae_tiling: bool = True  #
    enable_vae_slicing: bool = False  #
    memory_attention: MemoryAttention = MemoryAttention.Xformers  #
    quantization_method: QuantizationMethod = QuantizationMethod.Layerwise  #
    quantization_mode: QuantizationMode = QuantizationMode.FP8  #
    enable_llava_quantization: bool = True
    llava_quantization_mode: QuantizationMode = QuantizationMode.INT4  #
    llava_offload_model: bool = False  #
    llava_weight_dtype: WeightDtype = WeightDtype.Float16  #
    llava_question_prompt: str = "Describe what you see in this image and put it in prompt format limited to 77 tokens:"  #
    allow_cuda_tf32: bool = False  #
    allow_cudnn_tf32: bool = False  #
    latest_preset: str = "Default"
    civitai_token: str = ""
    save_image_on_upscaling_passes: bool = False
    disable_mmap: bool = True
    always_offload_models: bool = True
    run_vae_on_cpu: bool = False
    running_on_spaces: bool = False


@dataclass
class SchedulerConfig:
    """Configuration class for Scheduler."""

    beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "scaled_linear"
    timestep_spacing: Literal["linspace", "leading", "trailing"] = "linspace"
    scale_linear_exponent: float = 1.93

    def to_dict(self) -> dict:
        """Converts the dataclass instance to a dictionary."""
        from dataclasses import asdict

        return asdict(self)


@dataclass
class PipelineParams:
    supir_model: SUPIRModel = SUPIRModel.Quality
    apply_prompt_2: bool = True
    strength: float = 7.0
    prompt: str = None
    negative_prompt: str = None
    num_steps: int = 30
    num_images: int = 1
    start_point: StartPoint = StartPoint.Noise
    randomize_seed: bool = True
    seed: int = -1
    guidance_scale: float = 7.0
    guidance_rescale: float = 0.5
    use_linear_CFG: bool = False
    reverse_linear_CFG: bool = False
    guidance_scale_start: float = 1.0
    cfg_decay_rate: float = 0.5
    strength_decay_rate: float = 0.5
    restoration_scale: float = 4.0
    s_churn: float = 0.3
    s_noise: float = 1.003
    controlnet_conditioning_scale: float = 1.0
    use_linear_control_scale: bool = False
    reverse_linear_control_scale: bool = False
    control_scale_start: float = 0.0
    enable_PAG: bool = False
    use_linear_PAG: bool = False
    reverse_linear_PAG: bool = False
    pag_scale_start: float = 1.0
    pag_scale: float = 3.0
    pag_adaptive_scale: float = 0
    pag_layers: Union[str, List[str]] = None
    use_lpw_prompt: bool = False
    tile_size: int = 1024
    tile_overlap: float = 0.5
    tile_gaussian_sigma: float = 0.05
    tile_weighting_method: WeightingMethod = WeightingMethod.Cosine
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    clip_skip: int = 0
    upscale_factor: str = "1x"
    image_size_fix_mode: ImageSizeFixMode = ImageSizeFixMode.Padding
    upscaling_mode: UpscalingMode = UpscalingMode.Direct
    prompt_2: str = None
    apply_ipa_embeds: bool = False
    invert_prompts: bool = False
    color_fix_mode: ColorFix = ColorFix.Nothing
    zero_sft_injection_configs: Optional[Union[Dict[str, Dict[str, Any]], InjectionConfigs]] = None
    zero_sft_injection_flags: Optional[Union[Dict[str, bool], InjectionFlags]] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = (None,)


@dataclass
class Config:
    # Load default settings
    _default_settings = DefaultSettings()

    # internals settings
    _selected_model_path: str = ""
    _selected_restore_model_path: str = ""
    _selected_vae_model_path: str = ""
    _quantization_dtype: str = None
    _ksize_for_dilate_binary_mask_fill = 31
    _ksize_for_dilate_binary_mask_fill_inverted = 25
    _gaussian_blur_ksize_for_control_mask_fill = 5

    latest_preset: str = _default_settings.latest_preset if os.path.exists(os.path.join("src", "presets", f"{_default_settings.latest_preset}.json")) else None
    civitai_token: str = _default_settings.civitai_token

    # region pipeline settings
    model_config_path: str = _default_settings.model_config_path
    cache_dir: str = _default_settings.cache_dir
    checkpoints_dir: str = _default_settings.checkpoints_dir
    restorer_dir: str = _default_settings.restorer_dir
    vae_dir: str = _default_settings.vae_dir
    default_vae: str = _default_settings.default_vae
    llava_model: str = _default_settings.llava_model
    restorer_engine: RestorerEngine = RestorerEngine.SUPIR
    restorer_pipeline_params: PipelineParams = PipelineParams()
    upscaler_engine: UpscalerEngine = UpscalerEngine.Nothing
    upscaler_pipeline_params: PipelineParams = PipelineParams()
    selected_restorer_checkpoint_model: str = ""
    selected_upscaler_checkpoint_model: str = ""
    selected_vae_model: str = ""
    selected_restorer_sampler: Sampler = Sampler.Euler
    selected_upscaler_sampler: Sampler = Sampler.Euler
    restorer_sampler_config: SchedulerConfig = SchedulerConfig()
    upscaler_sampler_config: SchedulerConfig = SchedulerConfig()
    mask_prompt: str = None
    restore_face: bool = False
    output_dir = _default_settings.output_dir
    save_image_format: str = _default_settings.save_image_format
    weight_dtype: WeightDtype = _default_settings.weight_dtype
    vae_weight_dtype: WeightDtype = _default_settings.vae_weight_dtype
    image_path: str = None
    device: RuntimeDevice = _default_settings.device
    generator_device: RuntimeDevice = _default_settings.generator_device
    save_image_on_upscaling_passes: bool = _default_settings.save_image_on_upscaling_passes
    disable_mmap: bool = _default_settings.disable_mmap
    enable_llava_quantization: bool = _default_settings.enable_llava_quantization
    llava_quantization_mode: QuantizationMode = _default_settings.llava_quantization_mode
    llava_offload_model: bool = _default_settings.llava_offload_model
    llava_weight_dtype: WeightDtype = _default_settings.llava_weight_dtype
    llava_question_prompt: str = _default_settings.llava_question_prompt
    # endregion
    # region Optimizations
    enable_cpu_offload: bool = _default_settings.enable_cpu_offload
    enable_vae_tiling: bool = _default_settings.enable_vae_tiling
    enable_vae_slicing: bool = _default_settings.enable_vae_slicing
    quantization_mode: QuantizationMode = _default_settings.quantization_mode
    quantization_method: QuantizationMethod = _default_settings.quantization_method
    allow_cudnn_tf32: bool = _default_settings.allow_cudnn_tf32
    allow_cuda_tf32: bool = _default_settings.allow_cuda_tf32
    memory_attention: MemoryAttention = _default_settings.memory_attention
    disable_mmap: bool = _default_settings.disable_mmap
    always_offload_models: bool = _default_settings.always_offload_models
    run_vae_on_cpu: bool = _default_settings.run_vae_on_cpu
    running_on_spaces: bool = _default_settings.running_on_spaces
    # endregion

    # region upscaler
    enable_SAFMN_upscaler: bool = False
    SAFMN_upscaler_scale: str = ""
    # endregion

    def __init__(self, models_root_path: str = None):
        # set models root_path
        if not bool(models_root_path):
            self.models_root_path = Path(__file__).resolve().parent
        else:
            self.models_root_path = models_root_path

    def get_bool_value(self, prop):
        return "Enabled" if prop else "Disabled"

    def get_device(self, device):
        if device.value == "cpu":
            return torch.device("cpu")
        elif device.value == "cuda":
            return torch.device("cuda")
        elif device.value == "mps":
            return torch.device("mps")

    def get_unet_device_map(self):
        device_map = {
            "conv_in": self.device.value,
            "time_proj": self.device.value,
            "time_embedding": self.device.value,
            "add_time_proj": self.device.value,
            "add_embedding": self.device.value,
            "down_blocks": self.device.value,
            "up_blocks.0.attentions.0": self.device.value,
            "up_blocks.0.attentions.1": self.device.value,
            "up_blocks.0.attentions.2": self.device.value,
            "up_blocks.0.resnets": self.device.value,
            "up_blocks.0.upsamplers": self.device.value,
            "up_blocks.1": self.device.value,
            "up_blocks.2": self.device.value,
            "mid_block": self.device.value,
            "conv_norm_out": self.device.value,
            "conv_act": self.device.value,
            "conv_out": self.device.value,
        }
        return device_map

    def get_llava_device_map(self):
        device_map = {
            "model.language_model": "cpu" if self.llava_offload_model else self.device.value,
            "model.multi_modal_projector": "cpu",
            "model.vision_tower": "cpu",
            "lm_head": "cpu",
        }
        return device_map

    def get_weight_dtype(self, dtype):
        if dtype.value == "Bfloat16":
            return torch.bfloat16
        elif dtype.value == "Float16":
            return torch.float16
        else:
            return torch.float32

    def check_if_is_safetensors(self, selected_model):
        return True if ".safetensors" in selected_model else False

    def get_checkpoint_model_name(self, selected_checkpoint_model):
        selected_model = None if not bool(selected_checkpoint_model) or selected_checkpoint_model is None else selected_checkpoint_model
        model_name = selected_model
        is_safetensors = self.check_if_is_safetensors(model_name)
        model_name = os.path.join(self.checkpoints_dir, model_name) if is_safetensors else model_name
        return (model_name, is_safetensors)

    def get_model_name(self, selected_model):
        selected_model = None if not bool(selected_model) or selected_model is None else selected_model
        model_name = selected_model
        is_safetensors = self.check_if_is_safetensors(model_name)
        model_name = os.path.join(self.restorer_dir, model_name) if is_safetensors else model_name
        return (model_name, is_safetensors)

    def get_vae_model_name(self):
        selected_model = None if not bool(self.selected_vae_model) or self.selected_vae_model is None else self.selected_vae_model
        model_name = self.default_vae if selected_model is None else selected_model
        is_safetensors = self.check_if_is_safetensors(model_name)
        model_name = os.path.join(self.vae_dir, model_name) if is_safetensors else model_name
        return (model_name, is_safetensors)

    def get_model_path(self, model_name=None):
        model_name = model_name or self.get_checkpoint_model_name()
        model_path = f"{self.cache_dir}/{model_name}" if self.cache_dir else model_name
        return model_path
