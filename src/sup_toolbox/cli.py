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

import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Import configuration classes and enums from the sup_toolbox package
from sup_toolbox.config import Config
from sup_toolbox.enums import (
    ImageSizeFixMode,
    PromptMethod,
    RestorerEngine,
    UpscalerEngine,
    UpscalingMode,
)
from sup_toolbox.modules.model_manager import ModelManager
from sup_toolbox.modules.SUPIR.pipeline_supir_stable_diffusion_xl import (
    InjectionConfigs,
    InjectionFlags,
    InjectionScaleConfig,
)
from sup_toolbox.sup_toolbox_pipeline import SUPToolBoxPipeline
from sup_toolbox.utils.logging import logger


# Load environment variables from a .env file if present
load_dotenv()


def deep_merge_dicts(base_dict: dict, merge_dict: dict) -> dict:
    """
    Recursively merges merge_dict into base_dict.
    merge_dict values overwrite base_dict values.
    """
    for key, value in merge_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_merge_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _apply_config_value(config_obj: Any, key: str, value: Any):
    """
    Helper function to safely set an attribute on a config object,
    handling enums and type conversions.
    """
    if not hasattr(config_obj, key):
        # This warning is suppressed for specific grouping keys that are handled by recursion.
        if key not in [
            "general",
            "cfg_settings",
            "controlnet_settings",
            "pag_settings",
            "post_processsing_settings",
        ]:
            logger.warning(f"Config object of type {type(config_obj).__name__} has no parameter '{key}'. Skipping.")
        return

    field_type = type(getattr(config_obj, key, None))
    if hasattr(field_type, "from_str") and isinstance(value, str):
        setattr(config_obj, key, field_type.from_str(value))
    elif hasattr(field_type, "__members__") and isinstance(value, str):
        try:
            setattr(config_obj, key, field_type(value))
        except ValueError:
            for member in field_type:
                if member.name.lower() == value.lower():
                    setattr(config_obj, key, member)
                    break
            else:
                logger.warning(f"Invalid value '{value}' for enum '{key}'.")
    else:
        setattr(config_obj, key, value)


def create_config_from_layers(cli_args: argparse.Namespace, base_config: Config) -> Config:
    """
    Builds the final Config object by layering settings on top of a pre-loaded base_config.
    Priority order: base_config < settings.json < preset.json < command-line overrides.
    """
    config = base_config

    # Apply the base settings.json file
    settings_path = Path(cli_args.settings_file) if cli_args.settings_file else Path(__file__).parent / "configs" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings_data = json.load(f)
            for key, value in settings_data.items():
                _apply_config_value(config, key, value)
        except Exception as e:
            logger.error(f"Error loading settings file {settings_path}: {e}")
    elif cli_args.settings_file:
        logger.error(f"Custom settings file not found at {settings_path}.")
    else:
        logger.warning(f"Default settings file not found at {settings_path}. Using class defaults only.")

    # Apply the specified preset.json file
    preset_path = Path(cli_args.preset_or_path)

    # If the path is not an existing file, it assumes it's a default preset name.
    if not preset_path.is_file():
        preset_filename = f"{cli_args.preset_or_path}.json"
        preset_path = Path(__file__).parent / "presets" / preset_filename

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found. Neither as a direct path '{cli_args.preset_or_path}' nor as a default preset name.")

    with open(preset_path, "r", encoding="utf-8") as f:
        preset_data = json.load(f)

    # Apply top-level keys from preset
    top_level_map = {
        "restorer_engine": "restorer_engine",
        "upscaler_engine": "upscaler_engine",
        "restorer_model": "selected_restorer_checkpoint_model",
        "upscaler_model": "selected_upscaler_checkpoint_model",
        "vae_model": "selected_vae_model",
        "restorer_sampler": "selected_restorer_sampler",
        "upscaler_sampler": "selected_upscaler_sampler",
    }
    for json_key, config_attr in top_level_map.items():
        if json_key in preset_data and preset_data[json_key] is not None:
            _apply_config_value(config, config_attr, preset_data[json_key])

    # Helper for special prompt fields
    def map_extra_fields(role_prefix: str, preset_data: dict, config: Config):
        params_obj = getattr(config, f"{role_prefix}_pipeline_params")
        if preset_data.get(f"{role_prefix}_prompt_1"):
            params_obj.prompt = preset_data[f"{role_prefix}_prompt_1"]
        if preset_data.get(f"{role_prefix}_prompt_2"):
            params_obj.prompt_2 = preset_data[f"{role_prefix}_prompt_2"]
        if preset_data.get(f"{role_prefix}_negative_prompt"):
            params_obj.negative_prompt = preset_data[f"{role_prefix}_negative_prompt"]
        if preset_data.get(f"{role_prefix}_prompt_method"):
            params_obj.use_lpw_prompt = preset_data[f"{role_prefix}_prompt_method"] == PromptMethod.Weighted.value

    map_extra_fields("restorer", preset_data, config)
    map_extra_fields("upscaler", preset_data, config)

    # Helper for recursively mapping settings dictionaries
    def map_settings_recursively(source_dict: dict, target_obj: Any):
        if not source_dict:
            return
        for key, value in source_dict.items():
            if isinstance(value, dict):
                if hasattr(target_obj, key):
                    map_settings_recursively(value, getattr(target_obj, key))
                else:
                    map_settings_recursively(value, target_obj)
            else:
                _apply_config_value(target_obj, key, value)

    # Apply nested settings from preset
    if "restorer_settings" in preset_data:
        map_settings_recursively(preset_data["restorer_settings"], config.restorer_pipeline_params)
    if "upscaler_settings" in preset_data:
        map_settings_recursively(preset_data["upscaler_settings"], config.upscaler_pipeline_params)
    if "restorer_sampler_settings" in preset_data:
        map_settings_recursively(preset_data["restorer_sampler_settings"], config.restorer_sampler_config)
    if "upscaler_sampler_settings" in preset_data:
        map_settings_recursively(preset_data["upscaler_sampler_settings"], config.upscaler_sampler_config)

    # Manually map SUPIR advanced settings
    if "restorer_supir_advanced_settings" in preset_data and config.restorer_engine == RestorerEngine.SUPIR:
        json_advanced = preset_data["restorer_supir_advanced_settings"]
        if json_advanced:
            flags, configs = InjectionFlags(), InjectionConfigs()
            for key, values_dict in json_advanced.items():
                if hasattr(flags, f"{key}_active"):
                    setattr(flags, f"{key}_active", values_dict.get("sft_active", True))
                if hasattr(configs, key):
                    valid_keys = {f.name for f in dataclasses.fields(InjectionScaleConfig)}
                    filtered_values = {k: v for k, v in values_dict.items() if k in valid_keys}
                    setattr(configs, key, InjectionScaleConfig(**filtered_values))
            config.restorer_pipeline_params.zero_sft_injection_flags = flags
            config.restorer_pipeline_params.zero_sft_injection_configs = configs

    if "upscaler_supir_advanced_settings" in preset_data and config.upscaler_engine == UpscalerEngine.SUPIR:
        json_advanced = preset_data["upscaler_supir_advanced_settings"]
        if json_advanced:
            flags, configs = InjectionFlags(), InjectionConfigs()
            for key, values_dict in json_advanced.items():
                if hasattr(flags, f"{key}_active"):
                    setattr(flags, f"{key}_active", values_dict.get("sft_active", True))
                if hasattr(configs, key):
                    valid_keys = {f.name for f in dataclasses.fields(InjectionScaleConfig)}
                    filtered_values = {k: v for k, v in values_dict.items() if k in valid_keys}
                    setattr(configs, key, InjectionScaleConfig(**filtered_values))
            config.upscaler_pipeline_params.zero_sft_injection_flags = flags
            config.upscaler_pipeline_params.zero_sft_injection_configs = configs

    # Apply command-line overrides
    config.image_path = cli_args.image_path

    if cli_args.restorer_engine:
        config.restorer_engine = RestorerEngine.from_str(cli_args.restorer_engine)
    if cli_args.upscaler_engine:
        config.upscaler_engine = UpscalerEngine.from_str(cli_args.upscaler_engine)
    if cli_args.restorer_model:
        config.selected_restorer_checkpoint_model = cli_args.restorer_model
    if cli_args.upscaler_model:
        config.selected_upscaler_checkpoint_model = cli_args.upscaler_model
    if cli_args.vae_model:
        config.selected_vae_model = cli_args.vae_model
    if cli_args.restore_face is not None:
        config.restore_face = cli_args.restore_face
    if cli_args.mask_prompt:
        config.mask_prompt = cli_args.mask_prompt

    override_map = {
        "_supir_model": "supir_model",
        "_prompt": "prompt",
        "_prompt_2": "prompt_2",
        "_negative_prompt": "negative_prompt",
        "_seed": "seed",
        "_steps": "num_steps",
        "_guidance_scale": "guidance_scale",
        "_guidance_rescale": "guidance_rescale",
        "_image_size_fix_mode": "image_size_fix_mode",
        "_tile_size": "tile_size",
        "_upscaling_mode": "upscaling_mode",
        "_upscale_factor": "upscale_factor",
        "_cfg_decay_rate": "cfg_decay_rate",
        "_strength_decay_rate": "strength_decay_rate",
        "_apply_prompt_2": "apply_prompt_2",
        "_restoration_scale": "restoration_scale",
        "_s_churn": "s_churn",
        "_s_noise": "s_noise",
        "_start_point": "start_point",
        "_strength": "strength",
        "_use_linear_cfg": "use_linear_CFG",
        "_reverse_linear_cfg": "reverse_linear_CFG",
        "_guidance_scale_start": "guidance_scale_start",
        "_controlnet_conditioning_scale": "controlnet_conditioning_scale",
        "_use_linear_control_scale": "use_linear_control_scale",
        "_reverse_linear_control_scale": "reverse_linear_control_scale",
        "_control_scale_start": "control_scale_start",
        "_enable_pag": "enable_PAG",
        "_pag_scale": "pag_scale",
        "_use_linear_pag": "use_linear_PAG",
        "_reverse_linear_pag": "reverse_linear_PAG",
        "_pag_scale_start": "pag_scale_start",
        "_color_fix_mode": "color_fix_mode",
        "_invert_prompts": "invert_prompts",
        "_apply_ipa_embeds": "apply_ipa_embeds",
        "_tile_overlap": "tile_overlap",
        "_tile_weighting_method": "tile_weighting_method",
        "_tile_gaussian_sigma": "tile_gaussian_sigma",
    }

    for role_prefix in ["restorer", "upscaler"]:
        params_obj = getattr(config, f"{role_prefix}_pipeline_params")
        for arg_suffix, config_key in override_map.items():
            arg_name = f"{role_prefix}{arg_suffix}"
            cli_value = getattr(cli_args, arg_name, None)
            if cli_value is not None:
                _apply_config_value(params_obj, config_key, cli_value)

        prompt_method_arg_name = f"{role_prefix}_prompt_method"
        prompt_method_value = getattr(cli_args, prompt_method_arg_name, None)
        if prompt_method_value is not None:
            params_obj.use_lpw_prompt = prompt_method_value == PromptMethod.Weighted.value

    return config


def handle_run_command(args: argparse.Namespace, initial_config: Config):
    """
    Contains the logic to execute the pipeline using a fully prepared config.
    """
    try:
        final_config = create_config_from_layers(args, initial_config)

        logger.info("Initializing Model Manager and checking for required models...")
        model_manager = ModelManager(final_config)

        model_manager.prepare_models(always_download_models=args.always_download_models)
        logger.info("Model check complete.")

        sup_toolbox_pipeline = SUPToolBoxPipeline(final_config, models_root_path=Path(__file__).resolve().parent.parent)

        if sup_toolbox_pipeline.initialize():
            result, process_status = sup_toolbox_pipeline.predict()
            if process_status and result:
                output_path = args.output
                if output_path:
                    result.save(output_path)
                    logger.info(f"Process finished. Image saved to {final_config.output_dir} and copy of image saved to {output_path}")
                else:
                    logger.info(f"Process finished. Image saved to {final_config.output_dir}")

            else:
                logger.error("Process failed.")
        else:
            logger.error("Pipeline initialization failed.")
    except Exception as e:
        logger.error(f"An error occurred during the 'run' command: {e}", exc_info=True)


def handle_export_preset_command(args: argparse.Namespace):
    """
    Finds a default preset and copies it to a user-specified location.
    """
    try:
        preset_name = args.preset_name
        output_file = Path(args.output_file)
        if output_file.suffix.lower() != ".json":
            output_file = output_file.with_suffix(".json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        default_preset_path = Path(__file__).parent / "presets" / f"{preset_name}.json"
        if not default_preset_path.exists():
            logger.error(f"Default preset '{preset_name}' not found.")
            return
        shutil.copy(default_preset_path, output_file)
        logger.info(f"Successfully exported preset '{preset_name}' to '{output_file}'")
    except Exception as e:
        logger.error(f"An error occurred during the 'export-preset' command: {e}", exc_info=True)


def main():
    # Pre-parse to find the settings file path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--settings-file", help="Path to a custom settings.json file.")
    pre_args, remaining_args = pre_parser.parse_known_args()

    # Load base config to find directories
    temp_config = Config()
    settings_path = Path(pre_args.settings_file) if pre_args.settings_file else Path(__file__).parent / "configs" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings_data = json.load(f)
            for key, value in settings_data.items():
                _apply_config_value(temp_config, key, value)
        except Exception as e:
            logger.error(f"Could not pre-load settings file for help message: {e}")

    # Build the dynamic list of available models
    model_choices_help = ""
    try:
        model_manager = ModelManager(temp_config)
        pretrained_models = model_manager.filter_models_by_model_type("Diffusers")
        pretrained_names = [m["model_name"] for m in pretrained_models]
        local_files = []
        checkpoints_dir = temp_config.checkpoints_dir
        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            local_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith((".safetensors", ".ckpt"))])
        all_model_choices = pretrained_names + local_files
        if all_model_choices:
            model_choices_help = "\nAvailable models:\n" + "\n".join(f"  - {name}" for name in all_model_choices)
    except Exception as e:
        logger.warning(f"Could not dynamically list available models in help message: {e}")

    # Build the main parser with dynamic help text
    parser = argparse.ArgumentParser(
        description="SUP-Toolbox: A command-line interface for the image processing pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Command: run
    parser_run = subparsers.add_parser(
        "run",
        help="Run the main processing pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    presets_dir = Path(__file__).parent / "presets"
    available_presets = [f.stem for f in presets_dir.glob("*.json")]

    core_group = parser_run.add_argument_group("Core Arguments")
    core_group.add_argument("--settings-file", help="Path to a custom settings.json file (already parsed if provided).")
    core_group.add_argument(
        "preset_or_path",
        help="Name of a built-in preset OR a file path to a custom .json preset.\nAvailable built-in presets:\n"
        + "\n".join(f"  - {p}" for p in available_presets),
    )
    core_group.add_argument("image_path", help="Path to the input image or a directory of images.")
    core_group.add_argument("--output", "-o", help="Path to save the output image.")
    core_group.add_argument(
        "--always-download-models",
        action="store_true",
        help="If specified, forces a full scan and download of all required models.",
    )

    config_group = parser_run.add_argument_group("Top-Level Config Overrides")
    config_group.add_argument(
        "--restorer-engine",
        dest="restorer_engine",
        choices=[e.value for e in RestorerEngine if e.value != "None"],
        help="Override the restorer engine.",
    )
    config_group.add_argument(
        "--upscaler-engine",
        dest="upscaler_engine",
        choices=[e.value for e in UpscalerEngine if e.value != "None"],
        help="Override the upscaler engine.",
    )
    config_group.add_argument(
        "--restorer-model",
        dest="restorer_model",
        help=f"Override the restorer's main checkpoint model name.{model_choices_help}",
    )
    config_group.add_argument(
        "--upscaler-model",
        dest="upscaler_model",
        help=f"Override the upscaler's main checkpoint model name.{model_choices_help}",
    )
    config_group.add_argument("--vae-model", dest="vae_model", help="Override the VAE model name.")
    config_group.add_argument(
        "--restorer-supir-model",
        dest="restorer_supir_model",
        choices=["Quality", "Fidelity"],
        help="[SUPIR Only] Override the restorer's SUPIR model type.",
    )
    config_group.add_argument(
        "--upscaler-supir-model",
        dest="upscaler_supir_model",
        choices=["Quality", "Fidelity"],
        help="[SUPIR Only] Override the upscaler's SUPIR model type.",
    )
    config_group.add_argument(
        "--restore-face",
        dest="restore_face",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable selective face restoration (applies to restorer).",
    )
    config_group.add_argument(
        "--mask-prompt",
        dest="mask_prompt",
        help="Prompt for face/object detection mask (e.g., 'face') (applies to restorer).",
    )

    # Helper to add granular, role-specific arguments
    def add_role_arguments(group, role_prefix):
        # --- Prompting ---
        group.add_argument(
            f"--{role_prefix}-prompt-method",
            dest=f"{role_prefix}_prompt_method",
            choices=[e.value for e in PromptMethod],
            help=f"Override {role_prefix}'s prompt method.",
        )
        group.add_argument(
            f"--{role_prefix}-prompt",
            dest=f"{role_prefix}_prompt_1",
            help=f"Override {role_prefix}'s positive prompt.",
        )
        group.add_argument(
            f"--{role_prefix}-prompt-2",
            dest=f"{role_prefix}_prompt_2",
            help=f"Override {role_prefix}'s secondary positive prompt.",
        )
        group.add_argument(
            f"--{role_prefix}-negative-prompt",
            dest=f"{role_prefix}_negative_prompt",
            help=f"Override {role_prefix}'s negative prompt.",
        )
        group.add_argument(
            f"--{role_prefix}-apply-prompt-2",
            dest=f"{role_prefix}_apply_prompt_2",
            action=argparse.BooleanOptionalAction,
            help=f"Override whether to apply prompt 2 for the {role_prefix}.",
        )

        # --- General Generation ---
        group.add_argument(
            f"--{role_prefix}-seed",
            dest=f"{role_prefix}_seed",
            type=int,
            help=f"Override {role_prefix}'s generation seed.",
        )
        group.add_argument(
            f"--{role_prefix}-steps",
            dest=f"{role_prefix}_steps",
            type=int,
            help=f"Override {role_prefix}'s number of inference steps.",
        )
        group.add_argument(
            f"--{role_prefix}-strength",
            dest=f"{role_prefix}_strength",
            type=float,
            help=f"Override {role_prefix}'s denoising strength.",
        )
        group.add_argument(
            f"--{role_prefix}-cfg-scale",
            dest=f"{role_prefix}_guidance_scale",
            type=float,
            help=f"Override {role_prefix}'s CFG scale.",
        )
        group.add_argument(
            f"--{role_prefix}-guidance-rescale",
            dest=f"{role_prefix}_guidance_rescale",
            type=float,
            help=f"Override {role_prefix}'s guidance rescale.",
        )
        group.add_argument(
            f"--{role_prefix}-scale-factor",
            dest=f"{role_prefix}_upscale_factor",
            help=f"Override {role_prefix}'s scale factor (e.g., '2x').",
        )
        group.add_argument(
            f"--{role_prefix}-tile-size",
            dest=f"{role_prefix}_tile_size",
            type=int,
            choices=[1024, 1280],
            help=f"Override {role_prefix}'s tile size.",
        )
        group.add_argument(
            f"--{role_prefix}-image-size-fix-mode",
            dest=f"{role_prefix}_image_size_fix_mode",
            choices=[e.value for e in ImageSizeFixMode],
            help=f"Override {role_prefix}'s image size fix mode.",
        )
        group.add_argument(
            f"--{role_prefix}-upscaling-mode",
            dest=f"{role_prefix}_upscaling_mode",
            choices=[e.value for e in UpscalingMode],
            help=f"Override {role_prefix}'s upscaling mode.",
        )
        group.add_argument(
            f"--{role_prefix}-cfg-decay-rate",
            dest=f"{role_prefix}_cfg_decay_rate",
            type=float,
            help="[Progressive Upscale] Override the CFG decay rate.",
        )
        group.add_argument(
            f"--{role_prefix}-strength-decay-rate",
            dest=f"{role_prefix}_strength_decay_rate",
            type=float,
            help="[Progressive Upscale] Override the strength decay rate.",
        )

        # --- Sampler Settings (s_churn, s_noise) ---
        group.add_argument(
            f"--{role_prefix}-s-churn",
            dest=f"{role_prefix}_s_churn",
            type=float,
            help=f"[SUPIR/FaithDiff] Override {role_prefix}'s s_churn.",
        )
        group.add_argument(
            f"--{role_prefix}-s-noise",
            dest=f"{role_prefix}_s_noise",
            type=float,
            help=f"[SUPIR/FaithDiff] Override {role_prefix}'s s_noise.",
        )
        group.add_argument(
            f"--{role_prefix}-start-point",
            dest=f"{role_prefix}_start_point",
            choices=["lr", "noise"],
            help="[SUPIR/FaithDiff] Override the start point.",
        )

        # --- Advanced CFG ---
        group.add_argument(
            f"--{role_prefix}-use-linear-cfg",
            dest=f"{role_prefix}_use_linear_cfg",
            action=argparse.BooleanOptionalAction,
            help=f"Override linear CFG usage for the {role_prefix}.",
        )
        group.add_argument(
            f"--{role_prefix}-reverse-linear-cfg",
            dest=f"{role_prefix}_reverse_linear_cfg",
            action=argparse.BooleanOptionalAction,
            help=f"Override reverse linear CFG for the {role_prefix}.",
        )
        group.add_argument(
            f"--{role_prefix}-guidance-scale-start",
            dest=f"{role_prefix}_guidance_scale_start",
            type=float,
            help="Override the starting guidance scale for linear CFG.",
        )

        # --- Advanced ControlNet ---
        group.add_argument(
            f"--{role_prefix}-controlnet-scale",
            dest=f"{role_prefix}_controlnet_conditioning_scale",
            type=float,
            help="Override the ControlNet conditioning scale.",
        )
        group.add_argument(
            f"--{role_prefix}-use-linear-control-scale",
            dest=f"{role_prefix}_use_linear_control_scale",
            action=argparse.BooleanOptionalAction,
            help="Override linear ControlNet scale usage.",
        )
        group.add_argument(
            f"--{role_prefix}-reverse-linear-control-scale",
            dest=f"{role_prefix}_reverse_linear_control_scale",
            action=argparse.BooleanOptionalAction,
            help="Override reverse linear ControlNet scale.",
        )
        group.add_argument(
            f"--{role_prefix}-control-scale-start",
            dest=f"{role_prefix}_control_scale_start",
            type=float,
            help="Override the starting value for linear ControlNet scaling.",
        )

        # --- Advanced PAG ---
        group.add_argument(
            f"--{role_prefix}-enable-pag",
            dest=f"{role_prefix}_enable_pag",
            action=argparse.BooleanOptionalAction,
            help="Enable or disable PAG.",
        )
        group.add_argument(
            f"--{role_prefix}-pag-scale",
            dest=f"{role_prefix}_pag_scale",
            type=float,
            help="Override the PAG scale.",
        )
        group.add_argument(
            f"--{role_prefix}-use-linear-pag",
            dest=f"{role_prefix}_use_linear_pag",
            action=argparse.BooleanOptionalAction,
            help="Override linear PAG usage.",
        )
        group.add_argument(
            f"--{role_prefix}-reverse-linear-pag",
            dest=f"{role_prefix}_reverse_linear_pag",
            action=argparse.BooleanOptionalAction,
            help="Override reverse linear PAG.",
        )
        group.add_argument(
            f"--{role_prefix}-pag-scale-start",
            dest=f"{role_prefix}_pag_scale_start",
            type=float,
            help="Override the starting value for linear PAG scaling.",
        )

        # --- Engine-Specific ---
        group.add_argument(
            f"--{role_prefix}-restoration-scale",
            dest=f"{role_prefix}_restoration_scale",
            type=float,
            help=f"[SUPIR Only] Override the {role_prefix}'s restoration scale.",
        )
        group.add_argument(
            f"--{role_prefix}-invert-prompts",
            dest=f"{role_prefix}_invert_prompts",
            action=argparse.BooleanOptionalAction,
            help="[FaithDiff Only] Override prompt inversion.",
        )
        group.add_argument(
            f"--{role_prefix}-apply-ipa-embeds",
            dest=f"{role_prefix}_apply_ipa_embeds",
            action=argparse.BooleanOptionalAction,
            help="[FaithDiff Only] Override IP-Adapter embedding usage.",
        )
        group.add_argument(
            f"--{role_prefix}-tile-overlap",
            dest=f"{role_prefix}_tile_overlap",
            type=int,
            help="[ControlNetTile Only] Override the tile overlap.",
        )
        group.add_argument(
            f"--{role_prefix}-tile-weighting-method",
            dest=f"{role_prefix}_tile_weighting_method",
            choices=["Cosine", "Gaussian"],
            help="[ControlNetTile Only] Override the tile weighting method.",
        )
        group.add_argument(
            f"--{role_prefix}-tile-gaussian-sigma",
            dest=f"{role_prefix}_tile_gaussian_sigma",
            type=float,
            help="[ControlNetTile Only] Override the tile Gaussian sigma.",
        )

        # --- Post-processing ---
        group.add_argument(
            f"--{role_prefix}-color-fix-mode",
            dest=f"{role_prefix}_color_fix_mode",
            choices=["None", "Adain", "Wavelet"],
            help=f"Override {role_prefix}'s color fix mode.",
        )

    # Create argument groups for each role
    restorer_group = parser_run.add_argument_group("Restorer-Specific Parameter Overrides")
    add_role_arguments(restorer_group, "restorer")

    upscaler_group = parser_run.add_argument_group("Upscaler-Specific Parameter Overrides")
    add_role_arguments(upscaler_group, "upscaler")

    # Command: export-preset
    parser_export = subparsers.add_parser("export-preset", help="Export a default preset to a local file for customization.")
    parser_export.add_argument("preset_name", choices=available_presets, help="The name of the default preset to export.")
    parser_export.add_argument(
        "output_file",
        help="The path where the new preset file will be saved (e.g., 'my_custom_preset.json').",
    )

    # Final Parse
    args = parser.parse_args()

    # Execute Command
    if args.command == "run":
        handle_run_command(args, initial_config=temp_config)
    elif args.command == "export-preset":
        handle_export_preset_command(args)


if __name__ == "__main__":
    main()
