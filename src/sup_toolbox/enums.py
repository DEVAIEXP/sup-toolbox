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

from enum import Enum


class UpscalingMode(Enum):
    """
    Defines the upscaling method.
    - Progressive: Upscales iteratively (e.g., 2x, then 2x again).
    - Direct: Upscales directly to the target scale in a single step.
    """

    Progressive = "Progressive"
    Direct = "Direct"

    @staticmethod
    def from_str(value):
        if value == "Progressive":
            return UpscalingMode.Progressive
        if value == "Direct":
            return UpscalingMode.Direct
        else:
            raise NotImplementedError


class ColorFix(Enum):
    """
    Postprocessing ColorFix for generated image
    """

    Nothing = "None"
    Wavelet = "Wavelet"
    Adain = "Adain"

    @staticmethod
    def from_str(value):
        if value == "None":
            return ColorFix.Nothing
        if value == "Wavelet":
            return ColorFix.Wavelet
        if value == "Adain":
            return ColorFix.Adain
        else:
            raise NotImplementedError


class Sampler(Enum):
    Euler = "Euler"
    DPM_PP_1S = "DPM++ 1S"
    DPM_PP_1S_KARRAS = "DPM++ 1S Karras"
    DPM_PP_2S = "DPM++ 2S"
    DPM_PP_2S_KARRAS = "DPM++ 2S Karras"
    DPM_PP_2M = "DPM++ 2M"
    DPM_PP_2M_KARRAS = "DPM++ 2M Karras"
    DPM_PP_2M_SDE = "DPM++ 2M SDE"
    DPM_PP_2M_SDE_KARRAS = "DPM++ 2M SDE Karras"
    DPM_PP_2M_LU = "DPM++ 2M Lu"
    DPM_PP_2M_SDE_LU = "DPM++ 2M SDE Lu"
    DPM_PP_2M_EF = "DPM++ 2M Ef"
    DPM_PP_2M_SDE_EF = "DPM++ 2M SDE Ef"
    DPM_PP_3M = "DPM++ 3M"
    DPM_PP_3M_KARRAS = "DPM++ 3M Karras"
    DPM_PP_3M_SDE = "DPM++ 3M SDE"
    UniPC = "UniPC"

    @staticmethod
    def from_str(value):
        if value == "Euler":
            return Sampler.Euler
        if value == "DPM++ 1S":
            return Sampler.DPM_PP_1S
        if value == "DPM++ 1S Karras":
            return Sampler.DPM_PP_1S_KARRAS
        if value == "DPM++ 2S":
            return Sampler.DPM_PP_2S
        if value == "DPM++ 2S Karras":
            return Sampler.DPM_PP_2S_KARRAS
        if value == "DPM++ 2M":
            return Sampler.DPM_PP_2M
        if value == "DPM++ 2M Karras":
            return Sampler.DPM_PP_2M_KARRAS
        if value == "DPM++ 2M SDE":
            return Sampler.DPM_PP_2M_SDE
        if value == "DPM++ 2M SDE Karras":
            return Sampler.DPM_PP_2M_SDE_KARRAS
        if value == "DPM++ 2M Lu":
            return Sampler.DPM_PP_2M_LU
        if value == "DPM++ 2M SDE Lu":
            return Sampler.DPM_PP_2M_SDE_LU
        if value == "DPM++ 2M Ef":
            return Sampler.DPM_PP_2M_EF
        if value == "DPM++ 2M SDE Ef":
            return Sampler.DPM_PP_2M_SDE_EF
        if value == "DPM++ 3M":
            return Sampler.DPM_PP_3M
        if value == "DPM++ 3M Karras":
            return Sampler.DPM_PP_3M_KARRAS
        if value == "DPM++ 3M SDE":
            return Sampler.DPM_PP_3M_SDE
        if value == "UniPC":
            return Sampler.UniPC
        else:
            raise NotImplementedError


class ImageSizeFixMode(Enum):
    """
    Image size Fix modes for input images.
    """

    Nothing = "None"
    Padding = "Padding"
    ProgressiveResize = "Progressive Resize"

    @staticmethod
    def from_str(value):
        if value == "None":
            return ImageSizeFixMode.Nothing
        if value == "Padding":
            return ImageSizeFixMode.Padding
        if value == "Progressive Resize":
            return ImageSizeFixMode.ProgressiveResize
        else:
            raise NotImplementedError


class RestorerEngine(Enum):
    """
    Selected restorer engine.
    """

    Nothing = "None"
    SUPIR = "SUPIR"
    FaithDiff = "FaithDiff"

    @staticmethod
    def from_str(value):
        if value == "None":
            return RestorerEngine.Nothing
        if value == "SUPIR":
            return RestorerEngine.SUPIR
        if value == "FaithDiff":
            return RestorerEngine.FaithDiff
        else:
            raise NotImplementedError


class UpscalerEngine(Enum):
    """
    Selected upscaler engine.
    """

    Nothing = "None"
    SUPIR = "SUPIR"
    FaithDiff = "FaithDiff"
    ControlNetTile = "ControlNetTile"

    @staticmethod
    def from_str(value):
        if value == "None":
            return UpscalerEngine.Nothing
        if value == "SUPIR":
            return UpscalerEngine.SUPIR
        if value == "FaithDiff":
            return UpscalerEngine.FaithDiff
        if value == "ControlNetTile":
            return UpscalerEngine.ControlNetTile
        else:
            raise NotImplementedError


class SUPIRModel(Enum):
    """
    SUPIRModel
    """

    Quality = "Quality"
    Fidelity = "Fidelity"

    @staticmethod
    def from_str(value):
        if value == "Quality":
            return SUPIRModel.Quality
        if value == "Fidelity":
            return SUPIRModel.Fidelity
        else:
            raise NotImplementedError


class QuantizationMethod(Enum):
    """
    Quantization Method
    """

    Nothing = "None"
    Quanto = "Quanto Library"
    Layerwise = "Layerwise & Bnb"

    @staticmethod
    def from_str(value):
        if value == "None":
            return QuantizationMethod.Nothing
        if value == "Quanto Library":
            return QuantizationMethod.Quanto
        if value == "Layerwise & Bnb":
            return QuantizationMethod.Layerwise
        else:
            raise NotImplementedError


class QuantizationMode(Enum):
    """
    Quantization Mode
    """

    FP8 = "FP8"
    NF4 = "NF4"
    INT8 = "INT8"
    INT4 = "INT4"

    @staticmethod
    def from_str(value):
        if value == "FP8":
            return QuantizationMode.FP8
        if value == "NF4":
            return QuantizationMode.NF4
        if value == "INT8":
            return QuantizationMode.INT8
        if value == "INT4":
            return QuantizationMode.INT4
        else:
            raise NotImplementedError


class MemoryAttention(Enum):
    """
    Memory Attention
    """

    Nothing = "None"
    Xformers = "xformers"
    SDP = "sdp"

    @staticmethod
    def from_str(value):
        if value == "None":
            return MemoryAttention.Nothing
        if value == "xformers":
            return MemoryAttention.Xformers
        if value == "sdp":
            return MemoryAttention.SDP
        else:
            raise NotImplementedError


class RuntimeDevice(Enum):
    """
    Selected runtime device.
    """

    Balanced = "balanced"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    @staticmethod
    def from_str(value):
        if value == "balanced":
            return RuntimeDevice.Balanced
        if value == "cpu":
            return RuntimeDevice.CPU
        if value == "cuda":
            return RuntimeDevice.CUDA
        if value == "mps":
            return RuntimeDevice.MPS
        else:
            raise NotImplementedError


class WeightDtype(Enum):
    """
    Selected weight dtype.
    """

    Float32 = "Float32"
    Float16 = "Float16"
    Bfloat16 = "Bfloat16"

    @staticmethod
    def from_str(value):
        if value == "Float32":
            return WeightDtype.Float32
        if value == "Float16":
            return WeightDtype.Float16
        if value == "Bfloat16":
            return WeightDtype.Bfloat16
        else:
            raise NotImplementedError


class PromptMethod(Enum):
    """
    Selected prompt method.
    """

    Normal = "Normal"
    Weighted = "Long prompt weighted"

    @staticmethod
    def from_str(value):
        if value == "Normal":
            return PromptMethod.Normal
        if value in ("Weighted", "Long prompt weighted"):
            return PromptMethod.Weighted


class StartPoint(Enum):
    """
    Latent start point
    """

    LR = "lr"
    Noise = "noise"

    @staticmethod
    def from_str(value):
        if value == "lr":
            return StartPoint.LR
        if value == "noise":
            return StartPoint.Noise
        else:
            raise NotImplementedError


class WeightingMethod(Enum):
    """
    Weighting Method for Tile
    """

    Cosine = "Cosine"
    Gaussian = "Gaussian"

    @staticmethod
    def from_str(value):
        if value == "Cosine":
            return WeightingMethod.Cosine
        if value == "Gaussian":
            return WeightingMethod.Gaussian
        else:
            raise NotImplementedError


class ModelType(Enum):
    """
    Model Type
    """

    ImageTools = "ImageTools"
    CLIP = "CLIP"
    ControlNet = "ControlNet"
    Diffusers = "Diffusers"
    Caption = "Caption"
    LoRA = "LoRA"
    Refiner = "Refiner"
    Restorer = "Restorer"
    VAE = "VAE"
