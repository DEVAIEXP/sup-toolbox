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

import gc
import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict

import bitsandbytes as bnb
import numpy as np
import torch
from accelerate import dispatch_model
from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
from diffusers import DiffusionPipeline
from diffusers.models.embeddings import TextTimeEmbedding, TimestepEmbedding
from optimum.quanto import freeze, qfloat8_e4m3fn, qint4, qint8, quantize
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image
from torch import nn

from sup_toolbox.modules.FaithDiff.models.faithdiff_unet import Encoder
from sup_toolbox.utils.system import release_memory

from .config import SAMPLERS_OTHERS, SAMPLERS_SUPIR, SUPIR_PIPELINE_NAME


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
MAX_SEED = np.iinfo(np.int32).max


class PipelineDeviceManager:
    """
    A manager to save, reset, and restore the device placement strategy
    of a DiffusionPipeline by inspecting the actual device of each submodule.
    This correctly handles granular, user-defined device_maps.
    """

    def __init__(self, pipe: DiffusionPipeline, is_active: bool = True):
        self.pipe = pipe
        self.saved_state = None
        self._is_managed = False
        self._is_active = is_active

    def is_active(self):
        return self._is_active

    def inspect_device_map(
        self,
        return_granular: bool = False,
        show_map_in_console: bool = False,
        log_to_file: bool = False,
        log_dir: str = "logs",
    ) -> Dict[str, Any]:
        """
        Inspects the pipeline's components and reconstructs a granular
        device_map for each one by checking the actual device of its submodules.

        This is a read-only operation that provides a snapshot of the current
        device placement strategy.

        Args:
            return_granular (bool, optional):
                If True, always returns the full granular map, even if all
                submodules are on the same device. Defaults to False.
            show_map_in_console (bool, optional):
                If True, prints a summary or the full map to the console.
                May be truncated for very large models. Defaults to False.
            log_to_file (bool, optional):
                If True, saves the detailed granular maps to a timestamped
                log file for full inspection. Defaults to False.
            log_dir (str, optional):
                The directory where the log file will be saved. Defaults to "logs".

        Returns:
            A dictionary representing the saved state, where keys are component
            names (e.g., "unet", "vae") and values contain their reconstructed
            device_map.
        """
        if show_map_in_console or log_to_file:
            print("Inspecting pipeline's current device placement state...")

        log_content = []
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"device_map_inspection_{timestamp}.log"
            os.makedirs(log_dir, exist_ok=True)
            log_filepath = os.path.join(log_dir, log_filename)
            log_content.append(f"Device Map Inspection Report - {timestamp}\n")

        state = {}
        for name, component in self.pipe.components.items():
            if not isinstance(component, torch.nn.Module):
                continue

            reconstructed_map = OrderedDict()
            for module_name, module in component.named_modules():
                try:
                    module_device = str(next(module.parameters()).device)
                    reconstructed_map[module_name] = module_device
                except StopIteration:
                    continue

            if reconstructed_map:
                simplified_map = OrderedDict()
                unique_devices = set(reconstructed_map.values())

                if len(unique_devices) == 1 and not return_granular:
                    simplified_map[""] = unique_devices.pop()
                    message = f"  - Component '{name}' is entirely on device '{simplified_map['']}'."
                    if show_map_in_console:
                        print(message)
                    if log_to_file:
                        log_content.append(message)
                else:
                    simplified_map = reconstructed_map
                    formatted_map_str = json.dumps(simplified_map, indent=2)
                    message = f"  - Found granular device_map for '{name}':\n{formatted_map_str}"
                    if show_map_in_console:
                        # Print to console (may be truncated)
                        print(message)
                    if log_to_file:
                        # Add the full, untruncated string to our log content
                        log_content.append(message)

                state[name] = {"map": simplified_map}

        # Write the collected log content to the file at the end
        if log_to_file:
            try:
                with open(log_filepath, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(log_content))
                print(f"Full device map inspection has been saved to: {log_filepath}")
            except Exception as e:
                print(f"Error writing to log file: {e}")

        return state

    def save_state(self):
        self.saved_state = self.inspect_device_map()

    def reset_device_map(self):
        r"""
        Resets the device maps (if any) to None.
        """
        self.pipe.remove_all_hooks()
        for name, component in self.pipe.components.items():
            if isinstance(component, torch.nn.Module):
                component.to("cpu")
        self.pipe.hf_device_map = None

    def unload_pipe(self):
        """
        Saves the current device placement state by calling `inspect_device_map`,
        removes accelerate hooks, and moves all components to the CPU to free VRAM.
        """
        if not self._is_active:
            print("Pipeline is already inactive/unloaded.")
            return

        # 2.Remove all accelerate hooks from the actual nn.Module components. and move components to CPU
        self.reset_device_map()
        if hasattr(self.pipe, "_offload_device"):
            optionally_disable_offloading(self.pipe)
            del self.pipe._offload_device
        # self.pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self._is_managed = True
        self._is_active = False  # Mark as inactive
        # print("Pipeline unloaded successfully.")

    def restore_pipe(self, device):
        if device:
            self.pipe.to(device)
            self._is_active = True

    def restore_components_device(self):
        for name, state_info in self.saved_state.items():
            if name not in self.pipe.components:
                continue

            component = getattr(self.pipe, name)
            device_map = state_info["map"]
            device = device_map.get("", component.device.type).rpartition(":")
            device = device[0] if device[1] else device[2]
            if component.device.type != device:
                dispatch_model(component, device_map=device_map)
                component.hf_device_map = device_map

    def restore_state(self):
        """
        Restores the pipeline to its original device placement using the
        reconstructed device_map.
        """
        if not self._is_managed or self.saved_state is None:
            print("No state to restore. Please call save_state_and_unload() first.")
            return

        for name, state_info in self.saved_state.items():
            if name not in self.pipe.components:
                continue

            component = getattr(self.pipe, name)
            device_map = state_info["map"]

            dispatch_model(component, device_map=device_map)
            component.hf_device_map = device_map

        self.saved_state = None
        self._is_managed = False
        self._is_active = True


def load_scheduler(pipe, pipe_name, selected_sampler, **kwargs):
    if pipe_name == SUPIR_PIPELINE_NAME:
        scheduler, add_kwargs = SAMPLERS_SUPIR[selected_sampler]
    else:
        scheduler, add_kwargs = SAMPLERS_OTHERS[selected_sampler]

    config = pipe.scheduler.config
    if kwargs is not None and len(kwargs) > 0:
        managed_kwargs = {**kwargs}
    else:
        managed_kwargs = {**add_kwargs}

    if selected_sampler in ("LCM", "LCM trailing"):
        config = {x: config[x] for x in config if x not in ("skip_prk_steps", "interpolation_type", "use_karras_sigmas")}
    elif selected_sampler in ("TCD", "TCD trailing"):
        config = {x: config[x] for x in config if x not in ("skip_prk_steps")}

    return scheduler.from_config(config, **managed_kwargs)


# region optimization
def optionally_disable_offloading(_pipeline):
    """
    Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

    Args:
        _pipeline (`DiffusionPipeline`):
            The pipeline to disable offloading for.

    Returns:
        tuple:
            A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
    """
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    if _pipeline is not None:
        for _, component in _pipeline.components.items():
            if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)

                remove_hook_from_module(component, recurse=True)

    return (is_model_cpu_offload, is_sequential_cpu_offload)


def quantize_quanto(unet, quantization_mode):
    if unet is None:
        return

    UNET_DTYPES = {"FP8": qfloat8_e4m3fn, "INT8": qint8, "INT4": qint4}
    quantize(unet, weights=UNET_DTYPES[quantization_mode], exclude=["denoise_encoder.*"])
    freeze(unet)
    release_memory()


def quantize_FP8(unet):
    if unet is None:
        return
    dtype = unet.dtype
    unet.to(torch.float8_e4m3fn)
    for module in unet.modules():  # revert lora modules to prevent errors with fp8
        if isinstance(module, (BaseTunerLayer, TextTimeEmbedding, TimestepEmbedding, Encoder)):
            module.to(dtype)

    if hasattr(unet, "encoder_hid_proj"):  # revert ip adapter modules to prevent errors with fp8
        if unet.encoder_hid_proj is not None:
            for module in unet.encoder_hid_proj.modules():
                module.to(dtype)
    release_memory()


def quantize_NF4(module):
    if module is None:
        return

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            device = child.weight.data.device

            # Create and configure the Linear layer
            has_bias = True if child.bias is not None else False

            # TODO: Make that configurable
            # fp16 for compute dtype leads to faster inference
            # and one should almost always use nf4 as a rule of thumb
            bnb_4bit_compute_dtype = torch.float16
            quant_type = "nf4"

            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=has_bias,
                compute_dtype=bnb_4bit_compute_dtype,
                quant_type=quant_type,
            )

            new_layer.load_state_dict(child.state_dict())
            new_layer = new_layer.to(device)

            # Set the attribute
            setattr(module, name, new_layer)
        else:
            # Recursively apply to child modules
            quantize_NF4(child)


# endregion
