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
import traceback
from importlib.resources import files
from pathlib import Path

from huggingface_hub import snapshot_download

from sup_toolbox.config import Config
from sup_toolbox.utils.logging import logger


class ModelManager:
    MODEL_CONFIG = None

    def __init__(self, config: Config):
        self.config = config
        self.load_model_config()

    def load_model_config(self):
        try:
            model_path = (
                files("sup_toolbox.configs").joinpath("model_config.json")
                if not bool(self.config.model_config_path) or self.config.model_config_path is None
                else self.config.model_config_path
            )
            if not os.path.exists(model_path):
                raise Exception(f"Model configuration file {model_path} does not exist!")

            with open(model_path, "r", encoding="utf8") as file:
                models = file.read()

            models_dict = json.loads(models)
            self.MODEL_CONFIG = models_dict
        except Exception as e:
            logger.error(f"Error in load_model_config: {str(e)}")
            print(traceback.format_exc())
            self.MODEL_LIST = []

    def filter_models_by_model_type(self, model_type, root_key="SetupModels"):
        filtered_models = list(self.MODEL_CONFIG[root_key][model_type])
        return filtered_models

    def get_predefined_models(self):
        return self.MODEL_CONFIG["PredefinedModels"]

    def filter_models_by_model_name(self, model_type, model_base, model_name, model_category=None, root_key="SetupModels"):
        filtered_models = [
            model
            for model in self.MODEL_CONFIG[root_key][model_type]
            if model["model_base"] == (model_base if (bool(model_base) and model_base is not None) else model["model_base"])
            and model["model_name"] == model_name
            and model["model_category"] == (model_category if (bool(model_category) and model_category is not None) else model["model_category"])
        ]
        filtered_models = filtered_models[0] if len(filtered_models) > 0 else []
        return filtered_models

    def filter_models_by_model_base(self, model_type, model_base, root_key="SetupModels"):
        filtered_models = [model for model in self.MODEL_CONFIG[root_key][model_type] if model["model_base"] == model_base]
        return filtered_models

    def get_model_settings(
        self,
        model_type,
        model_base,
        model_name,
        model_dir=None,
        model_category=None,
        is_download=False,
        download_in_root=False,
    ):
        try:
            model = self.filter_models_by_model_name(model_type, model_base, model_name, model_category=model_category)
            if len(model) == 0:
                raise Exception(f"Model with type '{model_type}', base '{model_base}', name '{model_name}' and category '{model_category}' not found.")

            model_path = model["model_id"]
            has_cache_dir = bool(self.config.cache_dir) and self.config.cache_dir is not None
            has_model_dir = bool(model_dir) and model_dir is not None
            if has_model_dir:
                model_path = os.path.join(model_dir, model_type) if download_in_root else os.path.join(model_dir, model_type, model["local_id"])
                return {**model, "model_path": model_path}
            elif has_cache_dir:
                cache_dir = Path(self.config.cache_dir)
                if cache_dir.is_absolute():
                    model_cache_dir = cache_dir
                else:
                    model_cache_dir = os.path.join(self.config.models_root_path, cache_dir)
                _model_path = os.path.join(model_cache_dir, model_type) if download_in_root else os.path.join(model_cache_dir, model_type, model["local_id"])
                os.makedirs(_model_path, exist_ok=True)
                model_path = _model_path
            return {
                **model,
                "model_path": (model_path if is_download and has_cache_dir else None if is_download else model_path),
            }
        except Exception as e:
            logger.error(f"Error in load_model_config: {str(e)}")
            print(traceback.format_exc())
            self.MODEL_LIST = []

    def prepare_models(self, always_download_models: bool = False):
        setup_models = dict(self.MODEL_CONFIG["SetupModels"]).keys()
        print("Checking model cache...")
        for model_type in setup_models:
            models = self.filter_models_by_model_type(model_type)
            for model in models:
                model_settings = self.get_model_settings(
                    model_type,
                    model["model_base"],
                    model["model_name"],
                    model["model_dir"],
                    model["model_category"],
                    is_download=True,
                    download_in_root=model["download_in_root"],
                )
                model_path = Path(model_settings["model_path"])
                exists_and_not_empty = model_path.is_dir() and bool(os.listdir(model_path))
                if not exists_and_not_empty or always_download_models:
                    print(f"Preparing {model_type} models...")
                    snapshot_download(
                        repo_id=model_settings["model_id"],
                        local_dir=model_path,
                        revision=model_settings["revision"],
                        use_auth_token=False,
                        local_dir_use_symlinks=False,
                        allow_patterns=model_settings["allow_patterns"],
                        ignore_patterns=model_settings["ignore_patterns"],
                    )
                    print(f"{model_type}...OK!")
