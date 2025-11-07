import os

import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from sup_toolbox.modules.IPAdapter.resampler import Resampler


class IPLoad:
    def load_ip_plus(self, unet, image_encoder_path, ip_ckpt, device, num_tokens):
        # load image encoder
        self.ip_ckpt = ip_ckpt
        self.unet = unet
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=2048,
            ff_mult=4,
        ).to(device, dtype=torch.float16)

        self.load_ip_adapter()

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])

    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        with torch.inference_mode():
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.image_encoder.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
