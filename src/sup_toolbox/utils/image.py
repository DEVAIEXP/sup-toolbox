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

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, PngImagePlugin, UnidentifiedImageError

from sup_toolbox.utils.logging import logger
from sup_toolbox.utils.system import release_memory


def check_image(current_image, max_step_scale=1.5, scale_factor=1):
    divisor = 64
    orig_w, orig_h = current_image.size

    min_img_dim = min(orig_w, orig_h)
    if orig_w == orig_h and min_img_dim <= 768:  # use 1024 for square images
        min_target_dim = 1024
    elif min_img_dim <= 768:
        min_target_dim = 768
    else:
        min_target_dim = 1024

    if orig_w < orig_h:
        target_w_calc = min_target_dim
        target_h_calc = int(target_w_calc * (orig_h / orig_w))
    else:
        target_h_calc = min_target_dim
        target_w_calc = int(target_h_calc * (orig_w / orig_h))

    final_w = int(round(target_w_calc / divisor) * divisor)
    final_h = int(round(target_h_calc / divisor) * divisor)
    final_w = max(final_w, int(round(orig_w / divisor) * divisor) if orig_w >= divisor else divisor)
    final_h = max(final_h, int(round(orig_h / divisor) * divisor) if orig_h >= divisor else divisor)

    target_w = int(final_w * scale_factor)
    target_h = int(final_h * scale_factor)
    resized_image = progressive_resize_pil(
        current_image,
        target_final_width=target_w,
        target_final_height=target_h,
        max_step_scale=max_step_scale,
        divisor_safe=divisor,
    )

    return resized_image


def progressive_resize_pil(
    current_image_pil: Image.Image,
    target_final_width: int,
    target_final_height: int,
    max_step_scale: float = 2.0,
    divisor_safe: int = 8,
    resample_method=Image.LANCZOS,
):
    """
    Progressively resizes a PIL Image to target dimensions in multiple steps.

    Args:
        current_image_pil (PIL.Image.Image): The input PIL image.
        target_final_width (int): The final target width in pixels.
        target_final_height (int): The final target height in pixels.
        max_step_scale (float, optional): Maximum scaling factor per step. Defaults to 2.0.
        divisor_safe (int, optional): Ensures dimensions are multiples of this. Defaults to 8.
        resample_method (Image.Resampling, optional): PIL resampling filter. Defaults to Image.LANCZOS.

    Returns:
        PIL.Image.Image: The upscaled (resized) PIL image.
    """
    current_w, current_h = current_image_pil.size
    logger.info(f"Progressive Resize: Start {current_w}x{current_h} -> Target {target_final_width}x{target_final_height}")

    if current_w >= target_final_width and current_h >= target_final_height:
        logger.info("Initial image already meets or exceeds target dimensions. No resize needed.")
        if current_w != target_final_width or current_h != target_final_height:
            logger.info(f"Resizing/cropping to exact target: {target_final_width}x{target_final_height}")
            current_image_pil = current_image_pil.resize((target_final_width, target_final_height), resample=resample_method)
        return current_image_pil

    previous_iter_w, previous_iter_h = -1, -1
    loop_count = 0
    max_loops = 10

    while (current_w < target_final_width or current_h < target_final_height) and loop_count < max_loops:
        loop_count += 1
        if current_w == previous_iter_w and current_h == previous_iter_h:
            logger.warning("Stagnation detected in progressive resize (dimensions did not change). Stopping.")
            break

        previous_iter_w, previous_iter_h = current_w, current_h
        scale_needed_w = target_final_width / current_w if current_w > 0 else float("inf")
        scale_needed_h = target_final_height / current_h if current_h > 0 else float("inf")

        if current_w < target_final_width and current_h < target_final_height:
            step_scale_candidate = min(scale_needed_w, scale_needed_h)
            step_scale = min(max_step_scale, step_scale_candidate)
            if step_scale <= 1.0:
                step_scale = min(max_step_scale, max(scale_needed_w, scale_needed_h))
                if step_scale <= 1.0:
                    logger.info("Very close to target, small step scale. Will do final adjustment.")
                    break
        elif current_w < target_final_width:
            step_scale = min(max_step_scale, scale_needed_w)
        elif current_h < target_final_height:
            step_scale = min(max_step_scale, scale_needed_h)
        else:
            break

        if step_scale < 1.001 and (current_w < target_final_width or current_h < target_final_height):
            logger.info(f"Calculated step_scale ({step_scale:.3f}) is too small. Attempting a slightly larger step or breaking.")
            if (
                max(
                    target_final_width / current_w if current_w > 0 else 1,
                    target_final_height / current_h if current_h > 0 else 1,
                )
                < 1.05
            ):
                break
            step_scale = 1.05

        target_step_w = current_w * step_scale
        target_step_h = current_h * step_scale
        new_w = max(current_w, int(round(target_step_w / divisor_safe) * divisor_safe))
        new_h = max(current_h, int(round(target_step_h / divisor_safe) * divisor_safe))
        new_w = min(new_w, target_final_width)
        new_h = min(new_h, target_final_height)
        if new_w == current_w and new_h == current_h and step_scale > 1.001:
            logger.info(
                f"  Dimensions after rounding ({new_w}x{new_h}) did not change from {current_w}x{current_h} with step_scale {step_scale:.2f}. Forcing minimum increase."
            )
            if current_w < target_final_width:
                new_w = min(target_final_width, current_w + divisor_safe)
            if current_h < target_final_height:
                new_h = min(target_final_height, current_h + divisor_safe)

        if new_w == current_w and new_h == current_h:
            logger.info(f"  No change in dimensions ({new_w}x{new_h}). Likely at target or stuck. Breaking loop.")
            break

        logger.info(f"  Step {loop_count}: Current {current_w}x{current_h} -> Resizing to {new_w}x{new_h}")
        current_image_pil = current_image_pil.resize((new_w, new_h), resample=resample_method)
        current_w, current_h = current_image_pil.size

    if current_w != target_final_width or current_h != target_final_height:
        logger.info(f"Final adjustment: Current {current_w}x{current_h} -> Target {target_final_width}x{target_final_height}")

        if current_w != target_final_width or current_h != target_final_height:
            logger.info(f"Performing final resize to {target_final_width}x{target_final_height}")
            current_image_pil = current_image_pil.resize((target_final_width, target_final_height), resample=resample_method)
            current_w, current_h = current_image_pil.size

    logger.info(f"Progressive Resize finished. Final resolution: {current_w}x{current_h}")
    return current_image_pil


# copied and adapted from https://huggingface.co/spaces/gokaygokay/TileUpscalerV2, licensed under Apache 2.0.
def create_hdr_effect(original_image, hdr):
    """
    Applies an HDR (High Dynamic Range) effect to an image based on the specified intensity.

    Args:
        original_image (PIL.Image.Image): The original image to which the HDR effect will be applied.
        hdr (float): The intensity of the HDR effect, ranging from 0 (no effect) to 1 (maximum effect).

    Returns:
        PIL.Image.Image: The image with the HDR effect applied.
    """
    if hdr == 0:
        return original_image  # No effect applied if hdr is 0

    # Convert the PIL image to a NumPy array in BGR format (OpenCV format)
    cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    # Define scaling factors for creating multiple exposures
    factors = [
        1.0 - 0.9 * hdr,
        1.0 - 0.7 * hdr,
        1.0 - 0.45 * hdr,
        1.0 - 0.25 * hdr,
        1.0,
        1.0 + 0.2 * hdr,
        1.0 + 0.4 * hdr,
        1.0 + 0.6 * hdr,
        1.0 + 0.8 * hdr,
    ]

    # Generate multiple exposure images by scaling the original image
    images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]

    # Merge the images using the Mertens algorithm to create an HDR effect
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)

    # Convert the HDR image to 8-bit format (0-255 range)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype("uint8")

    release_memory()

    # Convert the image back to RGB format and return as a PIL image
    return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))


def normalize_0_255(img):
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min) * 255
    return normalized_img.astype(np.uint8)


def dilate_mask(mask, ksize):
    try:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(float)
    except Exception:
        pass
    mask = normalize_0_255(mask)
    kernel = np.ones((ksize, ksize), np.uint8)
    kernel_size = kernel.shape[0] // 2
    dilated_mask = np.zeros_like(mask).astype(np.uint8)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0:  # white pixel
                y_start = max(0, y - kernel_size)
                y_end = min(mask.shape[0], y + kernel_size + 1)
                x_start = max(0, x - kernel_size)
                x_end = min(mask.shape[1], x + kernel_size + 1)

                dilated_mask[y_start:y_end, x_start:x_end] = 255  # set white

    return dilated_mask


def apply_gaussian_blur(image_mask, mask_blur_x=35, mask_blur_y=35):
    if mask_blur_x > 0:
        np_mask = np.array(image_mask)
        kernel_size = 2 * int(2.5 * mask_blur_x + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur_x)
        image_mask = Image.fromarray(np_mask)

    if mask_blur_y > 0:
        np_mask = np.array(image_mask)
        kernel_size = 2 * int(2.5 * mask_blur_y + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur_y)
        image_mask = Image.fromarray(np_mask)

    image_mask = np.array(image_mask)
    return image_mask


def padding_image_size(image: Image.Image, divisor: int = 32) -> Tuple[Image.Image, int, int, int, int]:
    """
    Pads an image so its dimensions (height and width) are multiples of a given divisor.
    Handles RGB/BGR conversion for OpenCV.

    Args:
        image (PIL.Image.Image): The input image.
        divisor (int): The divisor that the new height and width should be multiples of.

    Returns:
        Tuple[PIL.Image.Image, int, int, int, int]:
            - Padded PIL Image.
            - Original width.
            - Original height.
            - Padded width.
            - Padded height.
    """
    original_width, original_height = image.size

    pad_h = (divisor - original_height % divisor) % divisor
    pad_w = (divisor - original_width % divisor) % divisor

    min_img_dim = min(original_width, original_height)

    if pad_h == 0 and pad_w == 0 and min_img_dim <= 768 and original_width == original_height:
        target_w = 1024
        target_h = 1024
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        logger.info(f"Min Resizing: Start {original_width}x{original_height} -> Target {target_w}x{target_h}")
        return image, target_w, target_h, target_w, target_h  # square images to 1024
    elif pad_h == 0 and pad_w == 0:
        return image, original_width, original_height, original_width, original_height

    padded_width = original_width + pad_w
    padded_height = original_height + pad_h

    logger.info(f"Padding Resize: Start {original_width}x{original_height} -> Target {padded_width}x{padded_height}")
    image_np_rgb = np.array(image)

    if image_np_rgb.shape[-1] == 4:  # RGBA
        if image.mode == "RGBA":
            image_rgb = image.convert("RGB")
            image_np_rgb = np.array(image_rgb)
            print("Warning: RGBA image converted to RGB for padding with OpenCV.")

    image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
    padded_image_np_bgr = cv2.copyMakeBorder(image_np_bgr, top=0, bottom=pad_h, left=0, right=pad_w, borderType=cv2.BORDER_REPLICATE)

    padded_image_np_rgb = cv2.cvtColor(padded_image_np_bgr, cv2.COLOR_BGR2RGB)
    padded_image_pil = Image.fromarray(padded_image_np_rgb)

    return padded_image_pil, original_width, original_height, padded_width, padded_height


def image2tensor(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def tensor2image(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def save_image_with_metadata(image_data: str | Path | Image.Image | np.ndarray, save_path: str, metadata: Dict[str, Any]) -> Image.Image | None:
    """
    Adds custom metadata to an image and saves it to the specified path.

    For PNG, it uses tEXt chunks.
    For JPEG and other EXIF-supporting formats, it serializes the metadata
    as a JSON string and stores it in the 'UserComment' EXIF tag.

    Args:
        image_data: Image data as a filepath, Path, PIL Image, or NumPy array.
        save_path: Filepath where the modified image will be saved.
        metadata: Dictionary of custom metadata to add to the image.

    Returns:
        The saved image as a PIL Image object, or None if saving failed.
    """
    USER_COMMENT_TAG_ID = 0x9286

    if not save_path or not metadata:
        if isinstance(image_data, Image.Image):
            image_data.save(save_path)
            return image_data
        try:
            image = Image.open(image_data)
            image.save(save_path)
            return image

        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            print(f"Warning: Could not open or save image. Reason: {e}")
            return None

    try:
        if isinstance(image_data, (str, Path)):
            image = Image.open(image_data)
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data.copy()
        else:
            return None
    except Exception as e:
        print(f"Error loading image data: {e}")
        return None

    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGB")

    _, ext = os.path.splitext(save_path)
    file_format = image.format or ext.replace(".", "").upper()

    if file_format == "PNG":
        png_info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            png_info.add_text(str(key), str(value))

        image.save(save_path, "PNG", pnginfo=png_info, quality=100)

    elif file_format in ["JPEG", "JPG", "TIFF"]:
        try:
            metadata_json_string = json.dumps(metadata)
            exif_data = image.info.get("exif")

            if exif_data:
                exif = Image.Exif()
                exif.frombytes(exif_data)
            else:
                exif = Image.Exif()

            encoding_prefix = b"\x00\x00\x00\x00\x00\x00\x00\x00"
            comment_bytes = metadata_json_string.encode("utf-8")
            exif[USER_COMMENT_TAG_ID] = encoding_prefix + comment_bytes

            image.save(save_path, exif=exif.tobytes())

        except Exception:
            print(f"Metadata is not supported for `{file_format}` format, saving without it.")
            image.save(save_path)

    else:
        print(f"Warning: Metadata not supported for format '{file_format}'. Saving image without metadata.")
        image.save(save_path)

    try:
        saved_image = Image.open(save_path)
        return saved_image.convert("RGB")
    except Exception as e:
        print(f"Failed to reopen saved image at {save_path}. Error: {e}")
        return None
