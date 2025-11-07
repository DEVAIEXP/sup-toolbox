# Copyright 2025 The DEVAIEXP Team. All rights reserved.
#
# This file is a derivative work based on the original repository
# https://github.com/arenasys/stable-diffusion-webui-model-toolkit
#
# Original work is Copyright (c) 2023 arenatemp
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

import torch

from .toolkit import EMA_PREFIX, compute_metric, fix_model, get_allowed_keys, inspect_model, load, load_components, resolve_arch, tensor_shape, tensor_size


class ToolkitModel:
    def __init__(self):
        self.filename = ""
        self.model = {}
        self.metadata = {}

        self.partial = False

        self.fix_clip = False
        self.broken = []
        self.renamed = []

        self.a_found = {}
        self.a_rejected = {}
        self.a_resolved = {}
        self.a_potential = []
        self.a_type = ""
        self.a_classes = {}
        self.a_components = []

        self.m_str = "----/----/----"
        self.m_unet = None
        self.m_vae = None
        self.m_clip = None

        self.z_total = 0
        self.z_waste = 0
        self.z_junk = 0
        self.z_ema = 0

        self.k_junk = []
        self.k_ema = []


def do_analysis(model):
    tm = ToolkitModel()
    tm.model = model

    tm.a_found, tm.a_rejected = inspect_model(model, all=True)
    tm.a_resolved = resolve_arch(tm.a_found)

    if not tm.a_resolved:
        tm.m_str = "----/----/----"
        return tm

    tm.a_potential = list(tm.a_found.keys())
    tm.a_type = next(iter(tm.a_resolved))
    tm.a_classes = tm.a_resolved[tm.a_type]
    tm.a_components = [tm.a_classes[c][0] for c in tm.a_classes]

    tm.m_str, m = compute_metric(model, tm.a_resolved)
    tm.m_unet, tm.m_vae, tm.m_clip = m

    allowed = get_allowed_keys(tm.a_resolved)

    for k in model.keys():
        kk = (k, tensor_shape(k, model[k]))
        z = tensor_size(model[k])
        tm.z_total += z

        if kk in allowed:
            if z and model[k].dtype == torch.float32:
                tm.z_waste += z / 2
            if z and model[k].dtype == torch.float64:
                tm.z_waste += z - (z / 4)
        else:
            if k.startswith(EMA_PREFIX):
                tm.z_ema += z
                tm.k_ema += [k]
            else:
                tm.z_junk += z
                tm.k_junk += [k]

    return tm


source_list = []
file_list = []
loaded = None


def find_source(source):
    if not source:
        return None
    index = source_list.index(source)
    if index >= len(file_list):
        return None
    if not os.path.exists(file_list[index]):
        return None
    return file_list[index]


def do_load(source):
    global loaded

    loaded = None

    if not os.path.exists(source):
        raise RuntimeError(f"Cannot find {source}!")
    else:
        model, _ = load(source)
        renamed, broken = fix_model(model, fix_clip=True)
        loaded = do_analysis(model)
        loaded.renamed = renamed
        loaded.broken = broken
        loaded.fix_clip = True
        loaded.filename = source

    return loaded


load_components("sup_toolbox.tools.model_toolkit.components")
