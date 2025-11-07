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
    SUPIRModel,
    UpscalerEngine,
    UpscalingMode,
    WeightDtype,
)
from sup_toolbox.modules.SUPIR.pipeline_supir_stable_diffusion_xl import (
    InjectionConfigs,
    InjectionFlags,
    InjectionScaleConfig,
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

# Restorer SUPIR
config.selected_restorer_checkpoint_model = "juggernautXL_juggXIByRundiffusion"
config.selected_vae_model = "sdxl-vae-fp16-fix"
config.restorer_pipeline_params.seed = 469071357
config.restorer_engine = RestorerEngine.SUPIR
config.upscaler_engine = UpscalerEngine.Nothing
config.selected_restorer_sampler = Sampler.DPM_PP_3M
config.restorer_sampler_config.scale_linear_exponent = 1.93
config.restorer_pipeline_params.color_fix_mode = ColorFix.Wavelet
config.restorer_pipeline_params.supir_model = SUPIRModel.Quality
config.restorer_pipeline_params.upscale_factor = "1x"
config.restorer_pipeline_params.prompt = "Direct flash photography. Three (30-year-old men:1.1), (all black hair:1.2). Left man: (black t-shirt:1.1) with white text 'Road Kill Cafe' and in his right forearm has distinct (dark tribal tattoo:1.2).Their hands has clearly defined fingers and distinct outlines. A (plaster interior wall: 1.1) on the left."
config.restorer_pipeline_params.prompt_2 = "Extremely detailed faces, flawless natural skin texture, skin pore detailing, 4k, 8k, clean image, no noise, shot on Fujifilm Superia 400, sharp focus, faithful colors."
config.restorer_pipeline_params.negative_prompt = "low-res, disfigured, analog artifacts, smudged, animate, (out of focus:1.2), catchlights, over-smooth, extra eyes, worst quality, unreal engine, art, aberrations, surreal, pastel drawing, (tattoo patterns on walls:1.4), tatto patterns on skin, text on walls, green wall, grainy wall texture, harsh lighting, (tribal patterns on clothing text:1.3), tattoo on chest, dead eyes, deformed fingers, undistinct fingers outlines"
config.restorer_pipeline_params.num_images = 1
config.restorer_pipeline_params.num_steps = 30
config.restorer_pipeline_params.use_lpw_prompt = True
config.restorer_pipeline_params.tile_size = 1024
config.restorer_pipeline_params.restoration_scale = 4  # 0 a 4.0
config.restorer_pipeline_params.s_churn = 0.02  # 0.02, #0 a 1.0
config.restorer_pipeline_params.s_noise = 1.003
config.restorer_pipeline_params.strength = 1.0
config.restorer_pipeline_params.use_linear_CFG = True
config.restorer_pipeline_params.guidance_scale = 9.0  # 9 : CFG, #4.0 PAG/upscale;
config.restorer_pipeline_params.reverse_linear_CFG = True
config.restorer_pipeline_params.guidance_scale_start = 5.0  # 5 : CFG;
config.restorer_pipeline_params.use_linear_control_scale = True
config.restorer_pipeline_params.reverse_linear_control_scale = True
config.restorer_pipeline_params.controlnet_conditioning_scale = 1.0
config.restorer_pipeline_params.control_scale_start = 1.0
config.restorer_pipeline_params.enable_PAG = False
config.restorer_pipeline_params.use_linear_PAG = True
config.restorer_pipeline_params.reverse_linear_PAG = True
config.restorer_pipeline_params.pag_scale = 2.0
config.restorer_pipeline_params.pag_scale_start = 0.1
config.restorer_pipeline_params.start_point = StartPoint.Noise
config.restorer_pipeline_params.image_size_fix_mode = ImageSizeFixMode.Padding
config.restorer_pipeline_params.upscaling_mode = UpscalingMode.Direct
injection_flags_to_use = InjectionFlags(cross_up_block_0_stage1_active=False, cross_up_block_1_stage1_active=False)
injection_configs_to_use = InjectionConfigs(
    sft_post_mid=InjectionScaleConfig(scale_end=1.0, linear=True, scale_start=0.7, reverse=True),
    sft_up_block_0_stage0=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=True),
    sft_up_block_0_stage1=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.8, reverse=True),
    sft_up_block_0_stage2=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=True),
    sft_up_block_1_stage0=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=True),
    sft_up_block_1_stage1=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=False),
    sft_up_block_1_stage2=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=False),
    sft_up_block_2_stage0=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.9, reverse=False),
    sft_up_block_2_stage1=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.7, reverse=True),
    sft_up_block_2_stage2=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.7, reverse=True),
    cross_up_block_0_stage1=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=False),
    cross_up_block_0_stage2=InjectionScaleConfig(scale_end=1.0, linear=True, scale_start=0.7, reverse=True),
    cross_up_block_1_stage1=InjectionScaleConfig(scale_end=1.0, linear=False, scale_start=0.1, reverse=False),
    cross_up_block_1_stage2=InjectionScaleConfig(scale_end=1.0, linear=True, scale_start=0.7, reverse=True),
)
config.restorer_pipeline_params.zero_sft_injection_configs = injection_configs_to_use
config.restorer_pipeline_params.zero_sft_injection_flags = injection_flags_to_use
config.image_path = load_image("./assets/samples/band.png")

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
