# FaithDiff Diffusers Pipeline Script
# This script demonstrates how to use the FaithDiff model with the Diffusers library
# Note: Some utility functions are being reused from the sup_toolbox, but you can implement them yourself if you wish.

import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from PIL import Image

from sup_toolbox.config import PAG_LAYERS
from sup_toolbox.modules.FaithDiff.models.faithdiff_unet import UNet2DConditionModel
from sup_toolbox.modules.FaithDiff.pipeline_faithdiff_stable_diffusion_xl import (
    FaithDiffStableDiffusionXLPipeline,
)
from sup_toolbox.modules.IPAdapter.ip_load import IPLoad
from sup_toolbox.pipeline_util import load_scheduler, quantize_FP8
from sup_toolbox.utils.colorfix import wavelet_color_fix
from sup_toolbox.utils.image import padding_image_size


device = "cuda"
dtype = torch.float16
MAX_SEED = np.iinfo(np.int32).max

# Initialize the models and pipeline
vae = AutoencoderKL.from_pretrained("models/VAE/sdxl-vae-fp16-fix", torch_dtype=dtype, use_safetensors=False)
model_id = "./models/Diffusers/RealVisXL_V5.0"
pipe = FaithDiffStableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    vae=vae,
    unet=None,
    use_safetensors=True,
    variant="fp16",
).to(device)

# Here we need to use pipeline internal unet model
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", variant="fp16", use_safetensors=True, torch_dtype=dtype)

# Load aditional layers to the model
unet.load_additional_layers(weight_path="./models/Restorer/Faithdiff/diffusion_pytorch_model.safetensors", dtype=dtype)

# The use of ip-adapter-plus is optional but recommended to remove artifacts
ip_model = IPLoad()  # ip-adapter-plus_sdxl_vit-h
ip_model.load_ip_plus(
    pipe.unet,
    ip_ckpt="./models/IPAdapter/H94-adapters/ip-adapter-plus_sdxl_vit-h.safetensors",
    device=device,
    num_tokens=16,
    image_encoder_path="models/CLIP/CLIP-ViT-H-14",
)

# This is optional to improve memory efficiency
quantize_FP8(unet)
pipe.unet = unet
pipe.enable_xformers_memory_efficient_attention()

# Enable vae tiling
pipe.set_encoder_tile_settings()
pipe.enable_vae_tiling()

# Recommended for lower VRAM usage
pipe.enable_model_cpu_offload()

# Set selected scheduler (Using our adapted scheduler is not mandatory, but it will not be possible to adjust the `scale_linear_exponent` parameter, which showed better results with values ​​between 1.93 and 1.97.)
scheduler_config = {
    "beta_schedule": "scaled_linear",
    "scale_linear_exponent": 1.97,
    "timestep_spacing": "linspace",
}

pipe.scheduler = load_scheduler(pipe, pipe.__class__.__name__, "UniPC", **scheduler_config)
# ---- End of scheduler selection ----

# Input params
prompt = "The image features a woman in her 55s with blonde hair and a white shirt, smiling at the camera. She appears to be in a good mood and is wearing a white scarf around her neck. "
prompt_2 = "Ultra-quality photography, flawless natural skin texture, shot on Fujifilm Superia 400, clean image, sharp focus."

# FaithDiff tends to age the person in the image, so in the negative prompt you can add keywords to avoid it like wrinkles, fine lines, smile lines, age spots, blemishes, acne, scars, freckles, blotchy skin, etc.
negative_prompt = "low res, worst quality, blurry, out of focus, analog artifacts, oil painting, illustration, art, anime, cartoon, CG Style, unreal engine, render, artwork, (wrinkles:1.4), fine lines, smile lines, age spots, (blemishes:1.2), acne, scars, freckles, blotchy skin, deformed mouth, catchlights"

upscale = 2  # For upscaling 4x or greater, perform a 2x progressive upscale on the resulting image.
start_point = "noise"  # ("lr" or "noise")

# Load image
lq_image = load_image("./assets/samples/woman.png")

original_height = lq_image.height
original_width = lq_image.width

print(f"Current resolution: H:{original_height} x W:{original_width}")
width = original_width * int(upscale)
height = original_height * int(upscale)

print(f"Final resolution: H:{height} x W:{width}")

image = lq_image.resize((width, height), Image.LANCZOS)
input_image, width_init, height_init, width_now, height_now = padding_image_size(image)

prompt = prompt_2 + " " + prompt  # This inversion works better for some prompts
print(prompt)
print(negative_prompt)

# Enable PAG layers optionally
select_pag_layers = ["mid", "down.blocks.2.attn.0", "down.blocks.2.attn.1", "up.blocks.0.attn.0"]
pag_layers = [PAG_LAYERS[l] for l in list(select_pag_layers)]
# pipe.enable_pag(pag_layers) # to PAG works you need to uncomment this line

# prepare IP-adapter-plus embeddings
num_samples = 1
image_prompt_embeds, uncond_image_prompt_embeds = ip_model.get_image_embeds(input_image.resize((512, 512)))
bs_embed, seq_len, _ = image_prompt_embeds.shape
image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

# Restoration
# seed = random.randint(0, MAX_SEED)
seed = 1255098424
print(seed)
generator = torch.Generator(device="cpu").manual_seed(seed)

gen_image = pipe(
    image=input_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,  # 20 steps is enough for some images, up to 30 can bring a slight smoothing effect.
    guidance_scale=6,  # >5 can introduce artifacts if you dont use negative prompt
    strength=1.0,  # this param is experimental and works only when start_point="lr", 0.9 best matches the final result to the original image.
    s_churn=0.0,  # 0 to 1.0
    s_noise=1.003,
    image_prompt_embeds=image_prompt_embeds,
    uncond_image_prompt_embeds=uncond_image_prompt_embeds,
    generator=generator,
    start_point=start_point,
    height=height_now,
    width=width_now,
    original_size=(height_now, width_now),
    target_size=(height_now, width_now),
    use_linear_conditioning_hint_scale=False,  # this param is experimental
    reverse_linear_conditioning_hint_scale=False,  # this param is experimental
    conditioning_hint_scale=1.0,  # if use_linear enabled, try 1.2 then reduce to 1.0 on reverse linear.
    conditioning_hint_scale_start=1.0,  # values ​​below 1.0 cause degradation.
    use_linear_PAG=True,  # this param is experimental
    reverse_linear_PAG=False,  # this param is experimental
    use_lpw_prompt=True,
    pag_scale=1.0,
    pag_scale_start=0.1,
).images[0]

cropped_image = gen_image.crop((0, 0, width_init, height_init))
os.makedirs("outputs/restored", exist_ok=True)
final_image = wavelet_color_fix(cropped_image, image.resize((width_init, height_init)))
final_image.save("outputs/restored/result_FaithDiff.png")
