# SUPIR Diffusers Pipeline Script
# This script demonstrates how to use the SUPIR model with the Diffusers library
# Note: Some utility functions are being reused from the sup_toolbox, but you can implement them yourself if you wish.

import os

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
from PIL.Image import Resampling

from sup_toolbox.config import PAG_LAYERS, SUPIR_PIPELINE_NAME
from sup_toolbox.modules.SUPIR.models.autoencoderKL_SUPIR import AutoencoderKLSUPIR
from sup_toolbox.modules.SUPIR.models.controlnet_SUPIR import SUPIRControlNetModel
from sup_toolbox.modules.SUPIR.models.unet_2d_condition_SUPIR import UNet2DConditionModel
from sup_toolbox.modules.SUPIR.pipeline_supir_stable_diffusion_xl import (
    InjectionConfigs,
    InjectionFlags,
    InjectionScaleConfig,
    SUPIRStableDiffusionXLPipeline,
)
from sup_toolbox.pipeline_util import load_scheduler, quantize_FP8
from sup_toolbox.utils.colorfix import wavelet_color_fix
from sup_toolbox.utils.image import padding_image_size


MAX_SEED = np.iinfo(np.int32).max

device = "cuda"
dtype = torch.float16
dtype_2 = torch.float32  # SUPIR autoencoder dosen't have fix for fp16

model_id = "./models/Diffusers/juggernautXL_juggXIByRundiffusion"
supir_parts_path = "./models/Restorer/SUPIR_Q"

# Initialize the models and pipeline
supir_controlnet = SUPIRControlNetModel.from_pretrained(
    f"{supir_parts_path}/controlnet",
    torch_dtype=dtype,
)

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path=model_id,
    subfolder="unet",
    supir_model_path=supir_parts_path,
    torch_dtype=dtype,
)

vae = AutoencoderKL.from_pretrained("./models/VAE/sdxl-vae-fp16-fix", torch_dtype=dtype_2, use_safetensors=False)
denoise_encoder = AutoencoderKLSUPIR.from_pretrained(supir_parts_path, subfolder="denoise_encoder", torch_dtype=dtype_2)

pipe = SUPIRStableDiffusionXLPipeline.from_pretrained(
    model_id,
    vae=vae,
    denoise_encoder=denoise_encoder,
    unet=None,
    controlnet=supir_controlnet,
    torch_dtype=dtype,
)

# This is optional to improve memory efficiency
quantize_FP8(unet)
pipe.unet = unet
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()

# Recommended for lower VRAM usage
pipe.enable_model_cpu_offload()

# Set selected scheduler (Using our adapted scheduler is not mandatory, but it will not be possible to adjust the `scale_linear_exponent` parameter, which showed better results with values ​​between 1.93 and 1.97.)
scheduler_config = {
    "restore_cfg_s_tmin": 0.05,
    "sigma_max_for_guidance": 14.6146,
    "beta_schedule": "scaled_linear",
    "scale_linear_exponent": 1.93,
    "timestep_spacing": "linspace",
}
scheduler_name = "DPM++ 3M"  # "Euler"
pipe.scheduler = load_scheduler(pipe, SUPIR_PIPELINE_NAME, scheduler_name, **scheduler_config)
# ---- End of scheduler selection ----

# Input params
prompt = "Direct flash photography. Three (30-year-old men:1.1), (all black hair:1.2). Left man: (black t-shirt:1.1) with white text 'Road Kill Cafe' and in his right forearm has distinct (dark tribal tattoo:1.2).Their hands has clearly defined fingers and distinct outlines. A (plaster interior wall: 1.1) on the left."
prompt_2 = "Extremely detailed faces, flawless natural skin texture, skin pore detailing, 4k, 8k, clean image, no noise, shot on Fujifilm Superia 400, sharp focus, faithful colors."
negative_prompt = "low-res, disfigured, analog artifacts, smudged, animate, (out of focus:1.2), catchlights, over-smooth, extra eyes, worst quality, unreal engine, art, aberrations, surreal, pastel drawing, (tattoo patterns on walls:1.4), tatto patterns on skin, text on walls, green wall, grainy wall texture, harsh lighting, (tribal patterns on clothing text:1.3), tattoo on chest, dead eyes, deformed fingers, undistinct fingers outlines"

prompt = prompt + " " + prompt_2
print(prompt)
print(negative_prompt)

select_pag_layers = [
    "mid",
    "up.blocks.0.attn.2",
    "up.blocks.1.attn.2",
]  # the same layers used in SUPIR CrossAttn Injections

pag_layers = [PAG_LAYERS[l] for l in list(select_pag_layers)]
# pipe.enable_pag(pag_layers) # to PAG works you need to uncomment this line

upscale = 1  # Do first restore in 1x and then upscale progressively with 2x to desired size
start_point = "noise"  # ("lr" or "noise") #Default SUPIR is noise

# Load image
lq_image = load_image("./assets/samples/band.png")

original_height = lq_image.height
original_width = lq_image.width

print(f"Current resolution: H:{original_height} x W:{original_width}")
width = original_width * int(upscale)
height = original_height * int(upscale)

print(f"Final resolution: H:{height} x W:{width}")

lq_image = lq_image.resize((width, height), Resampling.LANCZOS)
image = lq_image.resize((width, height), Image.LANCZOS)
input_image, width_init, height_init, width_now, height_now = padding_image_size(lq_image)

# SUPIR Injection Configs
injection_flags_to_use = InjectionFlags(
    cross_up_block_0_stage1_active=False,  # Default SUPIR
    cross_up_block_1_stage1_active=False,  # Default SUPIR
)

# Here, `scale_end` is like `controlnet_conditioning_scale`;
# it represents the final value, and `scale_start` the initial value when `linear=True` and `reverse=False`.
# If `reverse=True`, then `scale_end` will be the initial value and `scale_start` will be the final value.
# If `linear=False`, then the value used will be from `controlnet_conditioning_scale`.
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

# Restoration
# seed = random.randint(0, MAX_SEED)
seed = 469071357
print(seed)
generator = torch.Generator(device="cpu").manual_seed(seed)

gen_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=input_image,
    num_inference_steps=30,
    num_images_per_prompt=1,
    restoration_scale=4,  # 0 a 4.0
    s_churn=0.02,  # 0.2, #0 a 1.0
    s_noise=1.003,
    strength=1.0,  # this param is experimental and works only when start_point="lr", 0.9 best matches the final result to the original image.
    output_type="pil",
    generator=generator,
    height=height_now,
    width=width_now,
    start_point=start_point,
    original_size=(height_now, width_now),
    target_size=(height_now, width_now),
    use_linear_CFG=True,
    guidance_scale=9.0,  # 9 : CFG, #4.0 PAG/upscale;
    reverse_linear_CFG=True,
    guidance_scale_start=5.0,  # 5 : CFG;
    use_linear_control_scale=True,
    reverse_linear_control_scale=True,  # this param is experimental
    controlnet_conditioning_scale=1.0,
    control_scale_start=1.0,
    use_linear_PAG=True,
    reverse_linear_PAG=True,  # this param is experimental
    pag_scale=2.0,  # 2.0
    pag_scale_start=0.1,
    use_lpw_prompt=True,
    tile_size=1024,
    zero_sft_injection_flags=injection_flags_to_use,  # if it is not None will override the default flags in the model
    zero_sft_injection_configs=injection_configs_to_use,  # if it is not None the config map will override these params: controlnet_conditioning_scale, use_linear_control_scale, reverse_linear_control_scale, these map param is experimental.
)[0]

cropped_image = gen_image[0].crop((0, 0, width_init, height_init))
os.makedirs("outputs/restored", exist_ok=True)
final_image = wavelet_color_fix(cropped_image, image.resize((width_init, height_init)))
final_image.save("outputs/restored/result_SUPIR.png")
