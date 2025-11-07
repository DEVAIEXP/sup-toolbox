# MoDControlnetTile Diffusers Pipeline Script
# This script demonstrates how to use the MoDControlnetTile model with the Diffusers library
# Note: Some utility functions are being reused from the sup_toolbox, but you can implement them yourself if you wish.

import os

import numpy as np
import torch
from diffusers import AutoencoderKL, ControlNetUnionModel
from diffusers.utils import load_image
from PIL import Image

from sup_toolbox.modules.MoDControlTile.mod_controlnet_tile_sr_sdxl import (
    StableDiffusionXLControlNetTileSRPipeline,
)
from sup_toolbox.pipeline_util import load_scheduler, quantize_FP8
from sup_toolbox.utils.colorfix import wavelet_color_fix
from sup_toolbox.utils.image import padding_image_size


device = "cuda"
dtype = torch.float16
MAX_SEED = np.iinfo(np.int32).max

# Initialize the models and pipeline
controlnet = ControlNetUnionModel.from_pretrained("models/Controlnet/union-promax-sdxl-1.0", torch_dtype=torch.float16, variant="fp16").to(device=device)
vae = AutoencoderKL.from_pretrained("models/VAE/sdxl-vae-fp16-fix", torch_dtype=dtype, use_safetensors=False)

model_id = "./models/Diffusers/RealVisXL_V5.0"
pipe = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)

# This is optional to improve memory efficiency
pipe.enable_xformers_memory_efficient_attention()
quantize_FP8(pipe.unet)
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

# Recommended for lower VRAM usage
pipe.enable_model_cpu_offload()

# Set selected scheduler (Using our adapted scheduler is not mandatory, but it will not be possible to adjust the `scale_linear_exponent` parameter, which showed better results with values ​​between 1.93 and 1.97.)
scheduler_config = {
    "beta_schedule": "scaled_linear",
    "scale_linear_exponent": 1.97,
    "timestep_spacing": "linspace",
}

pipe.scheduler = load_scheduler(pipe, pipe.__class__.__name__, "UniPC", **scheduler_config)

# Input params
scale_factor = 2
guidance_scale = 4
num_inference_steps = 35
denoising_strenght = 0.65
controlnet_strength = 1.0
tile_weighting_method = pipe.TileWeightingMethod.COSINE.value
tile_gaussian_sigma = 0.3
tile_overlap = 256
max_tile_size = 1024  # or 1280
prompt = "high-quality, noise-free edges, high quality, 4k, hd, 8k"
negative_prompt = "blurry, pixelated, noisy, low resolution, artifacts, poor details"

# Load image
control_image = load_image("./assets/samples/1.jpg")

original_height = control_image.height
original_width = control_image.width

print(f"Current resolution: H:{original_height} x W:{original_width}")
control_image_adjusted, width_init, height_init, width_now, height_now = padding_image_size(control_image, 32)

# Pre-upscale image for tiling
new_size = (
    int(control_image_adjusted.width * scale_factor),
    int(control_image_adjusted.height * scale_factor),
)
input_image = control_image_adjusted.resize(new_size, Image.LANCZOS)

print(f"Final resolution: H:{input_image.height} x W:{input_image.width}")
input_image_adjusted, width_init, height_init, width_now, height_now = padding_image_size(input_image, 32)

print(f"Target resolution: H:{height_now} x W:{width_now}")

# seed = random.randint(0, MAX_SEED)
seed = 42
print(seed)
generator = torch.Generator(device="cpu").manual_seed(seed)

# Image generation
gen_image = pipe(
    image=input_image_adjusted,
    control_image=control_image_adjusted,
    control_mode=[6],
    controlnet_conditioning_scale=float(controlnet_strength),
    prompt=prompt,
    negative_prompt=negative_prompt,
    tile_overlap=tile_overlap,
    height=height_now,
    width=width_now,
    original_size=(control_image_adjusted.height, control_image_adjusted.width),
    target_size=(height_now, width_now),
    guidance_scale=guidance_scale,
    strength=float(denoising_strenght),
    tile_weighting_method=tile_weighting_method,
    tile_size=max_tile_size,
    tile_gaussian_sigma=float(tile_gaussian_sigma),
    num_inference_steps=num_inference_steps,
    generator=generator,
)["images"][0]

os.makedirs("outputs/upscaled", exist_ok=True)
final_image = wavelet_color_fix(gen_image, control_image.resize((width_init, height_init)))
final_image.save("outputs/upscaled/result_ControlTile.png")
