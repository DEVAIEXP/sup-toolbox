# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-10

### Added

-   **Command-Line Interface (CLI):**
    -   Introduced a powerful CLI for running the pipeline and managing configurations.
    -   Added `run` command to process images using presets.
    -   Added `export-preset` command to export built-in presets for customization.
-   **Layered Configuration System:**
    -   Implemented a robust, layered configuration system with a clear priority: `Defaults < settings.json < preset.json < CLI overrides`.
    -   Added support for a global `settings.json` file for base configurations (e.g., model paths).
    -   Added a `--settings-file` argument to allow users to provide a custom base configuration.
-   **Preset Management:**
    -   Added a `presets/` directory with pre-configured JSON files for common workflows (e.g., `RES_SUPIR`, `RES_FaithDiff_UPS_SUPIR`).
    -   The CLI now intelligently loads presets either by name (from the package) or by direct file path.
-   **Dynamic Help Messages:**
    -   The CLI's `--help` message now dynamically scans the `checkpoints_dir` and lists available `.safetensors` models, improving usability.
-   **Environment Variable Support:**
    -   Integrated `python-dotenv` to automatically load environment variables from a `.env` file at startup.

### Changed

-   **Project Structure:**
    -   Refactored the project into two distinct parts: the `sup-toolbox` library and a separate Gradio UI application.
    -   Command-line logic was moved from miscellaneous scripts into a structured `sup_toolbox/cli.py` module with a registered entry point.
-   **Model Loading Logic:**
    -   Refactored `SUPToolBoxPipeline.load_models` to be more modular and intelligent.
    -   The pipeline now correctly handles scenarios where `Restorer` and `Upscaler` share the same engine but require different sub-models (e.g., different SUPIR models).
    -   Optimizations (`apply_optimizations`) are now correctly reapplied only when a pipeline is reloaded.
-   **Configuration Handling:**
    -   The CLI's configuration parser was completely rewritten to manually and robustly translate the UI-generated JSON preset structure into the internal `Config` object, removing any dependency on UI-specific dataclasses.
-   **Progressive Upscaling Logic:**
    -   The decay algorithm for `cfg_decay_rate` and `strength_decay_rate` was changed from a compounding percentage to a distributed linear decay, providing more predictable and stable results.

### Fixed

-   **Model Name Alignment:** Corrected internal inconsistencies in configuration classes (e.g., `generation_seed` is now consistently `seed`), aligning the library's data model with the structure of the JSON presets.
-   **CLI Override Logic:** Fixed a bug where command-line overrides for nested parameters (e.g., `--restorer-steps`) were not being applied correctly.
-   **SUPIR Model Override:** Fixed a bug where `--restorer-supir-model` was incorrectly treated as a global parameter instead of a role-specific one.

---