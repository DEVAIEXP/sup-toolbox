# SUPIR inference with SUP Toolbox Pipeline
# This script demonstrates how to use the SUPIR model with sup_toolbox library

from pathlib import Path

from diffusers.utils import load_image

from sup_toolbox.config import Config
from sup_toolbox.enums import (
    ColorFix,
    ImageSizeFixMode,
    MemoryAttention,
    QuantizationMethod,
    QuantizationMode,
    RestorerEngine,
    RuntimeDevice,
    Sampler,
    StartPoint,
    UpscalerEngine,
    UpscalingMode,
    WeightDtype,
)
from sup_toolbox.sup_toolbox_pipeline import SUPToolBoxPipeline
from sup_toolbox.utils.logging import logger


APP_ROOT_DIR = Path(__file__).resolve().parent.parent  # <- to run from ./scripts
config = Config(models_root_path=APP_ROOT_DIR)
config.weight_dtype = WeightDtype.Float16
config.vae_weight_dtype = WeightDtype.Float16
config.device = RuntimeDevice.CUDA
config.generator_device = RuntimeDevice.CPU
config.quantization_method = QuantizationMethod.Layerwise
config.quantization_mode = QuantizationMode.FP8
config.memory_attention = MemoryAttention.Xformers
config.save_image_on_upscaling_passes = True

# Upscaler FaithDiff
config.selected_upscaler_checkpoint_model = "RealVisXL_V5.0"
config.selected_vae_model = "sdxl-vae-fp16-fix"
config.upscaler_pipeline_params.seed = 1255098424
config.restorer_engine = RestorerEngine.Nothing
config.upscaler_engine = UpscalerEngine.FaithDiff
config.selected_upscaler_sampler = Sampler.UniPC
config.upscaler_sampler_config.scale_linear_exponent = 1.97
config.upscaler_pipeline_params.color_fix_mode = ColorFix.Wavelet
config.upscaler_pipeline_params.upscale_factor = "4x"
config.upscaler_pipeline_params.prompt = "Ultra-quality photography, flawless natural skin texture, shot on Fujifilm Superia 400, clean image, sharp focus."
config.upscaler_pipeline_params.prompt_2 = ""
config.upscaler_pipeline_params.negative_prompt = "low res, worst quality, blurry, out of focus, analog artifacts, oil painting, illustration, art, anime, cartoon, CG Style, unreal engine, render, artwork, (wrinkles:1.4), fine lines, smile lines, age spots, (blemishes:1.2), acne, scars, freckles, blotchy skin, deformed mouth, catchlights"
config.upscaler_pipeline_params.num_images = 1
config.upscaler_pipeline_params.num_steps = 30
config.upscaler_pipeline_params.use_lpw_prompt = True
config.upscaler_pipeline_params.tile_size = 1024
config.upscaler_pipeline_params.s_churn = 0.00  # 0 a 1.0
config.upscaler_pipeline_params.s_noise = 1.003
config.upscaler_pipeline_params.strength = 1.0
config.upscaler_pipeline_params.guidance_scale = 4.0
config.upscaler_pipeline_params.guidance_rescale = 0.5
config.upscaler_pipeline_params.use_linear_control_scale = False
config.upscaler_pipeline_params.reverse_linear_control_scale = False
config.upscaler_pipeline_params.controlnet_conditioning_scale = 1.0
config.upscaler_pipeline_params.control_scale_start = 1.0
config.upscaler_pipeline_params.enable_PAG = False
config.upscaler_pipeline_params.use_linear_PAG = True
config.upscaler_pipeline_params.reverse_linear_PAG = False
config.upscaler_pipeline_params.pag_scale = 1.0
config.upscaler_pipeline_params.pag_scale_start = 0.1
config.upscaler_pipeline_params.start_point = StartPoint.Noise
config.upscaler_pipeline_params.image_size_fix_mode = ImageSizeFixMode.Padding
config.upscaler_pipeline_params.upscaling_mode = UpscalingMode.Progressive
config.upscaler_pipeline_params.invert_prompts = True
config.upscaler_pipeline_params.apply_ipa_embeds = True
config.upscaler_pipeline_params.cfg_decay_rate = 0.5
config.upscaler_pipeline_params.strength_decay_rate = 0.5
config.image_path = load_image("./assets/samples/woman_restored.png")

sup_toolbox_pipeline = SUPToolBoxPipeline(config)

initialize_status = sup_toolbox_pipeline.initialize()

if initialize_status:
    result, process_status = sup_toolbox_pipeline.predict()
    if not process_status:
        status = "Process aborted!"
        logger.info(status)
        result = None
else:
    status = "Process aborted!"
    logger.info(status)
    result = None
