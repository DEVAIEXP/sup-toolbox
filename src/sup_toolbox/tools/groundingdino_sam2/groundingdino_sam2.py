# Copyright 2025 The DEVAIEXP Team. All rights reserved.
#
# This file is a derivative work based on the original notebook
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
#
# Original work is Copyright (c) 2021 NielsRogge
# and was licensed under the MIT License.
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
from typing import List, Optional

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from tqdm import tqdm


# Data Classes
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    ymax: int
    xmax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None


class GroundingDinoSAM2:
    def __init__(
        self,
        device="cpu",
        sam2_model_config="sam2.1_hiera_l.yaml",
        grounding_dino_config="GroundingDINO_SwinT_OGC.py",
        sam2_checkpoint="sam2.1_hiera_large.pt",
        grounding_dino_checkpoint="groundingdino_swint_ogc.pth",
        model_path="./models/ImageTools",
    ):
        self.device = device
        self.local_sam2_model = build_sam2(
            os.path.join(model_path, "DINOSAM2", sam2_model_config),
            os.path.join(model_path, "DINOSAM2", sam2_checkpoint),
            device=device,
        )
        self.local_sam2_predictor = SAM2ImagePredictor(self.local_sam2_model)
        self.local_grounding_model = load_model(
            os.path.join(model_path, "DINOSAM2", grounding_dino_config),
            os.path.join(model_path, "DINOSAM2", grounding_dino_checkpoint),
            device=device,
        )

    # Main Processing Functions
    def annotate(self, image: Image.Image, detection_results: List[DetectionResult]) -> np.ndarray:
        image_cv2 = np.array(image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        for detection in detection_results:
            label = detection.label
            score = detection.score
            box = detection.box
            mask = detection.mask

            color = np.random.randint(0, 256, size=3)

            if mask is not None:
                overlay = image_cv2.copy()
                mask_display = (mask * 255).astype(np.uint8)
                overlay[mask == 1] = color
                image_cv2 = cv2.addWeighted(image_cv2, 0.7, overlay, 0.3, 0)

                contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

            cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
            cv2.putText(
                image_cv2,
                f"{label}: {score:.2f}",
                (box.xmin, box.ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                2,
            )

        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    def segment(
        self,
        image: Image.Image,
        detection_results: List[DetectionResult],
        local_sam2_predictor=None,
    ) -> List[DetectionResult]:
        image_source = np.array(image)  # Removed load_image, use directly PIL Image
        local_sam2_predictor.set_image(image_source)

        # Get boxes in correct format
        boxes = np.array([det.box.xyxy for det in detection_results])

        # Use SAM2 predictor
        with torch.no_grad():
            masks, _, _ = local_sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

        # Handle mask dimensionality
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Assign masks to detection results
        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask > 0  # Convert to boolean mask

        return detection_results

    def detect(
        self,
        image: Image.Image,
        labels: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        local_grounding_model=None,
    ) -> List[DetectionResult]:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        w, h = image.size

        text_prompt = ". ".join(labels) + "."
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                boxes, confidences, phrases = predict(
                    model=local_grounding_model,
                    image=image_transformed.to(self.device),
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

        # Scale boxes by image dimensions
        boxes = boxes.to(self.device) * torch.Tensor([w, h, w, h]).to(self.device)
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        detections = []
        for box, score, label in zip(boxes, confidences, phrases):
            detections.append(
                DetectionResult(
                    score=score,
                    label=label,
                    box=BoundingBox(
                        xmin=int(box[0]),
                        ymin=int(box[1]),
                        xmax=int(box[2]),
                        ymax=int(box[3]),
                    ),
                )
            )
        return detections

    def process_image(
        self,
        input_image,
        text_prompt,
        box_threshold,
        text_threshold,
        local_sam2_predictor,
        local_grounding_model,
        apply_blur=True,
        blur_ksize=3,
        invert_mask=False,
        invert_mask_sequence=False,
        output_type="PIL",
    ):
        try:
            with tqdm(desc="Parsing with 'GroundingDinoSAM2' model...", iterable=[input_image]) as progress_bar:
                for _, image in enumerate([input_image]):
                    image = image.convert("RGB")
                    # Detect and segment
                    labels = [text_prompt]
                    detections = self.detect(
                        image,
                        labels,
                        box_threshold,
                        text_threshold,
                        local_grounding_model=local_grounding_model,
                    )

                    if not detections:
                        print(f"\r\nNo detections found for image with prompt '{text_prompt}'")
                        progress_bar.update()
                        return np.array(image), None, []

                    detections = self.segment(image, detections, local_sam2_predictor=local_sam2_predictor)

                    # Create visualization
                    segmented_image = self.annotate(image, detections)
                    if output_type == "PIL":
                        segmented_image = Image.fromarray(segmented_image)

                    # Handle mask generation and coordinates
                    combined_mask = None
                    detection_masks = []

                    if detections:
                        # Create binary mask
                        combined_mask = np.zeros(np.array(image).shape[:2], dtype=np.uint8)

                        for detection in detections:
                            if detection.mask is not None:
                                detection_mask = detection.mask.astype(np.uint8)
                                detection_mask = detection_mask * 255
                                combined_mask |= detection_mask

                                if invert_mask:
                                    inverted_mask = cv2.bitwise_not(detection_mask)
                                    detection_mask = inverted_mask

                                if apply_blur:
                                    detection_mask_blur = cv2.GaussianBlur(detection_mask, (blur_ksize, blur_ksize), 0)
                                    mask_normalized = detection_mask_blur / 255.0
                                    mask_normalized = np.power(mask_normalized, 0.5)
                                    detection_mask_blur = mask_normalized * 255
                                    detection_mask_blur = detection_mask_blur.astype(np.uint8)
                                    detection_mask = detection_mask_blur

                                if output_type == "PIL":
                                    detection_mask = Image.fromarray(detection_mask).convert("L")
                                detection_masks.append(detection_mask)

                        if len(detection_masks) > 1:
                            if invert_mask_sequence:
                                detection_masks = reversed(detection_masks)
                                detection_masks = list(detection_masks)

                        # Convert to image
                        if invert_mask:
                            combined_mask = cv2.bitwise_not(combined_mask)

                        if apply_blur:
                            detection_mask_blur = cv2.GaussianBlur(combined_mask, (blur_ksize, blur_ksize), 0)
                            mask_normalized = detection_mask_blur / 255.0
                            mask_normalized = np.power(mask_normalized, 0.5)
                            combined_mask = mask_normalized * 255
                            combined_mask = combined_mask.astype(np.uint8)

                        if output_type == "PIL":
                            combined_mask = Image.fromarray(combined_mask).convert("L")

                    progress_bar.update()
                return segmented_image, combined_mask, detection_masks
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.array(input_image), None, []  # Return None and empty list on error

    def to(self, device):
        for m in self.local_grounding_model.modules():
            m.to(device)
        self.local_sam2_model.to(device)
        self.device = device

    def __call__(
        self,
        input_image: Image.Image,
        prompt,
        box_threshold=0.38,
        text_threshold=0.25,
        apply_blur=True,
        blur_ksize=3,
        invert_mask=False,
        invert_mask_sequence=False,
        output_type="PIL",
    ):
        segmented_image, combined_mask, detected_masks = self.process_image(
            input_image,
            prompt,
            box_threshold,
            text_threshold,
            self.local_sam2_predictor,
            self.local_grounding_model,
            apply_blur=apply_blur,
            blur_ksize=blur_ksize,
            invert_mask=invert_mask,
            invert_mask_sequence=invert_mask_sequence,
            output_type=output_type,
        )

        detected_masks = None if len(detected_masks) == 0 else detected_masks
        return segmented_image, combined_mask, detected_masks
