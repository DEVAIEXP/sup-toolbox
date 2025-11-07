<h1 align="center">SUP - Toolbox (Scaling UP Toolbox) for image restoration and upscaling 🚀</h1>
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow:hidden;">    
</div>

[![PyPI version](https://badge.fury.io/py/sup_toolbox.svg)](https://badge.fury.io/py/sup_toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/sup_toolbox.svg)](https://pypi.org/project/sup_toolbox/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.gnu.org/licenses/apache-2.0)
[![Downloads](https://static.pepy.tech/badge/sup_toolbox)](https://pepy.tech/project/sup_toolbox)
[![Downloads/Month](https://static.pepy.tech/badge/sup_toolbox/month)](https://pepy.tech/project/sup_toolbox)
If you liked this project, please give me a star! ⭐

## About

**SUP-Toolbox** (Scaling UP Toolbox) is a powerful and flexible command-line application for advanced image restoration and upscaling. It integrates state-of-the-art diffusion models like **SUPIR**, **FaithDiff**, and **ControlNetTile** to provide superior image enhancement, detail recovery, and high-resolution synthesis.

Whether you're restoring old photos or upscaling digital art, SUP-Toolbox offers granular control over every step of the process through a robust pipeline and an intuitive configuration system.

## Features

-   **Multiple Engines:** Seamlessly switch between and combine different restoration and upscaling engines.
-   **Powerful CLI:** A comprehensive command-line interface designed for single-image processing, batch operations, and automation.
-   **Preset System:** Use built-in presets for common tasks or create, save, and load your own custom configurations in JSON format.
-   **Layered Configuration:** A flexible system that layers settings from defaults, a main settings file, presets, and command-line arguments for maximum control.
-   **Advanced Controls:** Fine-tune dozens of parameters, including schedulers, samplers, guidance scales, ControlNet settings, and model-specific features like SUPIR's SFT injection.
-   **Automatic Model Management:** Automatically downloads and caches all required models on the first run.

## Requirements

### Software
-   Python 3.10 or newer.
-   Git.
-   An appropriate version of PyTorch with CUDA support.

### Hardware & Storage

-   **GPU:** An NVIDIA GPU with CUDA support and Compute Capability 7.0+ is required for full feature support (including FP8 quantization).
-   **System RAM:** A minimum of 16GB of system RAM is recommended.
-   **VRAM:** GPU memory usage varies significantly. **12GB+ of VRAM** is recommended for a comfortable experience. See the detailed [VRAM Usage Matrix](#vram-usage-matrix) for specific scenarios.
-   **Disk Space:** A minimum of **83 GB** of free disk space is required to download and store all the models used by this application.
-   **Disk Type:** For optimal performance, especially faster model loading times, it is highly recommended to store the models on a **Solid-State Drive (SSD)**, preferably an **NVMe M.2 drive**.


## Installation

### 1. Clone the Repository
First, clone the project to your local machine:
```bash
git clone https://github.com/your-username/sup-toolbox.git
cd sup-toolbox
```

### 2. Configure Settings (Important for WSL2 Users)

Before running the setup script, you may need to adjust the base configuration, especially if you are using WSL2.

1.  Locate the base settings file at `sup_toolbox/configs/settings.json`.
2.  Open it in a text editor.
3.  **For WSL2 users:** It is highly recommended to disable memory mapping to prevent slow model loading issues when accessing files from the Windows filesystem. Change this line:
    ```json
    "disable_mmap": false,
    ```
    to:
    ```json
    "disable_mmap": true,
    ```
4.  You can also update the `checkpoints_dir`, `vae_dir`, etc., to point to your existing model directories.

### 3. Run the Setup Script

We provide automated setup scripts that create a Python virtual environment, activate it, and install all necessary dependencies.

*   **On Windows:**
    Simply double-click the `setup.bat` file, or run it from a terminal:
    ```batch
    .\setup.bat
    ```

*   **On Linux or macOS:**
    First, make the script executable, then run it:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

After the script finishes, the virtual environment will be active, and the `sup-toolbox` command will be available in your terminal.

*For manual installation steps, please refer to the [Manual Installation Guide](./docs/manual_install.md).*

## Usage

There are two primary ways to use **SUP-Toolbox**: as a powerful **Command-Line Interface (CLI)** for quick results and batch processing, or as a **Python Library** for integration into your own custom applications and workflows.

---

### 1. As a Command-Line Interface (CLI)

The CLI is the quickest way to get started. It uses a flexible preset system to manage all settings.

#### Quick Start Workflow

1.  **Export a Built-in Preset:**
    First, export a default configuration file to use as a template. This makes it easy to see all available options.
    ```bash
    # This command finds the 'RES_SUPIR' preset and saves it as 'my_settings.json'
    sup-toolbox export-preset RES_SUPIR my_settings.json
    ```

2.  **Customize Your Preset (Optional):**
    Open the newly created `my_settings.json` file in a text editor and adjust the parameters as needed (e.g., change prompts, steps, or models).

3.  **Run the Pipeline:**
    Execute the `run` command, pointing to your image and your chosen preset (either a built-in one by name or your custom file by path).
    
    *   **Using a built-in preset:**
        ```bash
        sup-toolbox run RES_FaithDiff ./assets/samples/woman.png
        ```

    *   **Using your custom preset file:**
        ```bash
        sup-toolbox run my_settings.json ./assets/samples/woman.png
        ```
    
    *   **With command-line overrides:**
        For quick experiments, you can override any setting from the preset.
        ```bash
        sup-toolbox run RES_SUPIR ./assets/samples/band.png --restorer-steps 30 --restorer-seed 12345
        ```

*(For a full list of commands and parameters, see the [CLI Reference](#cli-reference-all-parameters)).*

---

### 2. As a Python Library

For advanced use cases and integration into other applications, you can import and use the `SUPToolBoxPipeline` directly in your Python code. This gives you maximum flexibility.

We provide a set of example scripts in the `/scripts` directory to demonstrate how to use the library.

#### Example Scripts

The `/scripts` folder contains practical examples for each engine. They are divided into two categories:

*   **`*_pipeline_diffusers.py`:** These scripts show how to use the underlying models with the standard `diffusers` library, without the `SUP-Toolbox` pipeline wrapper. They are useful for understanding the core mechanics.
*   **`*_pipeline_toolbox.py`:** **(Recommended)** These scripts demonstrate the power and simplicity of using the `SUPToolBoxPipeline`. They show how to configure and run the entire workflow with just a few lines of code.

**Available Scripts:**
-   `SUPIR_restorer_pipeline_toolbox.py`
-   `SUPIR_upscaler_pipeline_toolbox.py`
-   `FaithDiff_restorer_pipeline_toolbox.py`
-   `FaithDiff_upscaler_pipeline_toolbox.py`
-   `ControlnetTile_pipeline_toolbox.py`
-   *(and their `_diffusers.py` counterparts)*

#### Basic Library Usage Example

Here is a minimal example of how to use `SUP-Toolbox` as a library, inspired by the scripts:

```python
from pathlib import Path
from sup_toolbox.config import Config
from sup_toolbox.enums import ColorFix, RestorerEngine
from sup_toolbox.sup_toolbox_pipeline import SUPToolBoxPipeline
from sup_toolbox.utils.logging import logger

# 1. Create a configuration object
config = Config(models_root_path=Path(__file__).resolve().parent.parent)

# 2. Set the desired parameters
config.restorer_engine = RestorerEngine.SUPIR
config.selected_restorer_checkpoint_model = "juggernautXL_juggXIByRundiffusion"
config.image_path = "./assets/samples/band.png"

# You can configure any parameter, including nested ones
config.restorer_pipeline_params.color_fix_mode = ColorFix.Wavelet
config.restorer_pipeline_params.num_steps = 30
config.restorer_pipeline_params.use_lpw_prompt = True
config.restorer_pipeline_params.prompt = "Direct flash photography. Three (30-year-old men:1.1), (all black hair:1.2). Left man: (black t-shirt:1.1) with white text 'Road Kill Cafe' and in his right forearm has distinct (dark tribal tattoo:1.2).Their hands has clearly defined fingers and distinct outlines. A (plaster interior wall: 1.1) on the left."
config.restorer_pipeline_params.prompt_2 =  "Extremely detailed faces, flawless natural skin texture, skin pore detailing, 4k, 8k, clean image, no noise, shot on Fujifilm Superia 400, sharp focus, faithful colors."
config.restorer_pipeline_params.negative_prompt = "low-res, disfigured, analog artifacts, smudged, animate, (out of focus:1.2), catchlights, over-smooth, extra eyes, worst quality, unreal engine, art, aberrations, surreal, pastel drawing, (tattoo patterns on walls:1.4), tatto patterns on skin, text on walls, green wall, grainy wall texture, harsh lighting, (tribal patterns on clothing text:1.3), tattoo on chest, dead eyes, deformed fingers, undistinct fingers outlines"

# 3. Initialize the pipeline
# The models_root_path should point to the root of the sup-toolbox project
pipeline = SUPToolBoxPipeline(config)

# 4. Run the process
initialize_status = pipeline.initialize()
if initialize_status:
    result, process_status = pipeline.predict()
    if not process_status:
        status = "Process aborted!"
        logger.info(status)
        result = None
else:
    status = "Process aborted!"
    logger.info(status)
    result = None
```
#### Configuration Objects Reference

The pipeline is controlled by a hierarchy of dataclasses. For a complete list of all available attributes and their descriptions, please refer to our detailed API documentation:

➡️ **[Library Usage: Configuration Objects Guide](./docs/sup_toolbox_usage.md)**

This approach gives you programmatic control over the entire pipeline, making it easy to integrate into your own projects.

---
## Configuration System

SUP-Toolbox uses a layered approach for configuration, providing maximum flexibility. Settings are applied in the following order of priority (later steps override earlier ones):

1.  **Default Class Values:** The hardcoded defaults in the source code.
2.  **`settings.json` File:** Your base configuration, containing global paths and hardware settings. You can specify a custom file with `--settings-file`.
3.  **Preset File (`*.json`):** The main configuration for a specific task, loaded by name or by path.
4.  **CLI Overrides:** Any arguments provided directly on the command line (e.g., `--restorer-steps 50`).

### Built-in Presets

The preset names follow a `ROLE_ENGINE` convention (`RES` for Restorer, `UPS` for Upscaler).

| Preset Name | Description |
| :--- | :--- |
| **`Default`** | A minimal configuration, ideal as a blank slate for creating a new preset from scratch. |
| **`RES_SUPIR`** | **Restoration Only:** Uses the `SUPIR` engine for 1x image restoration. |
| **`RES_FaithDiff`** | **Restoration Only:** Uses the `FaithDiff` engine for 1x image restoration. |
| **`UPS_ControlnetTile`** | **Upscaling Only:** Uses the `ControlNetTile` engine to upscale an image without an initial restoration pass. |
| **`RES_SUPIR_UPS_SUPIR`** | **Full Pipeline (SUPIR):** A complete workflow using `SUPIR` for both restoration and upscaling. |
| **`RES_FaithDiff_UPS_FaithDiff`**| **Full Pipeline (FaithDiff):** A complete workflow using `FaithDiff` for both phases. |
| **`RES_SUPIR_UPS_FaithDiff`** | **Hybrid Pipeline:** Uses `SUPIR` for restoration and `FaithDiff` for upscaling. |
| **`RES_FaithDiff_UPS_SUPIR`** | **Hybrid Pipeline:** Uses `FaithDiff` for restoration and `SUPIR` for upscaling. |

## CLI Reference: All Parameters

For full control, you can override any parameter. Most are role-specific and require a `--restorer-` or `--upscaler-` prefix.

<details>
<summary><strong>Click to expand the full list of CLI arguments</strong></summary>

### Top-Level Config Overrides
| Argument | Description |
| :--- | :--- |
| `--restorer-engine <ENGINE>`| Overrides the restorer engine. |
| `--upscaler-engine <ENGINE>`| Overrides the upscaler engine. |
| `--restorer-model <NAME>`| Overrides the restorer's main checkpoint model name. |
| `--upscaler-model <NAME>` | Overrides the upscaler's main checkpoint model name. |
| `--vae-model <NAME>` | Overrides the VAE model name. |
| `--restorer-supir-model <TYPE>` | [SUPIR Only] Overrides the restorer's SUPIR model type (`Quality` or `Fidelity`). |
| `--upscaler-supir-model <TYPE>` | [SUPIR Only] Overrides the upscaler's SUPIR model type (`Quality` or `Fidelity`). |
| `--restore-face` / `--no-restore-face` | Enables or disables selective face restoration for the restorer. |
| `--mask-prompt <PROMPT>` | Sets the prompt for face/object detection mask (e.g., 'face'). |

### Role-Specific Overrides (`--restorer-*` or `--upscaler-*`)

#### **Prompting**
| Argument Suffix | Description (from UI) |
| :--- | :--- |
| `-prompt <TEXT>` | Overrides the positive prompt. |
| `-prompt-2 <TEXT>` | Overrides the secondary positive prompt. |
| `-negative-prompt <TEXT>` | Overrides the negative prompt. |
| `-prompt-method <METHOD>` | Overrides the prompt method (e.g., `Long prompt weighted`). |

#### **General Generation**
| Argument Suffix | Description (from UI) |
| :--- | :--- |
| `-seed <INT>` | The random seed for generation. -1 means a random seed. |
| `-steps <INT>` | The number of denoising steps. |
| `-strength <FLOAT>` | How much to denoise the original image (0.0 to 1.0). |
| `-cfg-scale <FLOAT>` | Classifier-Free Guidance Scale. How strongly the prompt is adhered to. |
| `-guidance-rescale <FLOAT>`| Guidance rescale factor (phi). |
| `-scale-factor <STR>` | Resolution upscale factor (e.g., '2x', '4x'). |
| `-tile-size <INT>` | The size of latent tiles (1024 or 1280). |
| `-image-size-fix-mode <MODE>` | Method to handle aspect ratio mismatches (`Progressive Resize`, `Padding`, etc.). |
| `-upscaling-mode <MODE>` | Upscaling method (`Progressive` or `Direct`). |
| `-cfg-decay-rate <FLOAT>` | [Progressive Upscale] The percentage to reduce CFG at each pass. |
| `-strength-decay-rate <FLOAT>` | [Progressive Upscale] The percentage to reduce Denoising Strength at each pass. |

#### **Sampler Settings (for SUPIR/FaithDiff)**
| Argument Suffix | Description (from UI) |
| :--- | :--- |
| `-s-churn <FLOAT>` | Stochasticity churn factor. Adds extra noise at each step. |
| `-s-noise <FLOAT>` | Stochasticity noise factor. |
| `-start-point <MODE>` | Start from low-res latents ('lr') or pure noise ('noise'). |

#### **Advanced Settings (CFG, ControlNet, PAG)**
*(These arguments are for fine-tuning and often require a deep understanding of the models)*
| Argument Suffix | Description |
| :--- | :--- |
| `-use-linear-cfg` / `-no-use-linear-cfg` | Enable/disable linear CFG scaling. |
| `-controlnet-scale <FLOAT>` | The weight of the ControlNet guidance. |
| `-enable-pag` / `-no-enable-pag` | Enable or disable Perturbed Attention Guidance. |
| *(... and many more for linear scaling, start values, etc. Use `--help` for a full list)* |

#### **Engine-Specific Settings**
| Argument Suffix | Description |
| :--- | :--- |
| `-restoration-scale <FLOAT>`| **[SUPIR Only]** Strength of the SUPIR restoration guidance. |
| `-invert-prompts` / `-no-invert-prompts` | **[FaithDiff Only]** Invert the order of prompt 1 and prompt 2. |
| `-apply-ipa-embeds` / `-no-apply-ipa-embeds` | **[FaithDiff Only]** Apply IP-Adapter embeddings during diffusion. |
| `-tile-overlap <INT>` | **[ControlNetTile Only]** Overlap in pixels between tiles. |
| `-tile-weighting-method <METHOD>` | **[ControlNetTile Only]** Method for blending tile edges (`Cosine` or `Gaussian`). |

#### **Post-Processing**
| Argument Suffix | Description |
| :--- | :--- |
| `-color-fix-mode <MODE>` | Applies a color correction method (`None`, `Adain`, `Wavelet`). |

</details>

## VRAM Usage Matrix

The following tables provide approximate peak VRAM usage based on different configurations. These tests were performed with an **NVIDIA Driver version 581.42**. Your results may vary.

<details>
<summary><strong>Click to expand VRAM usage tables</strong></summary>

-   **Default VAE:** Uses the standard VAE that comes with the checkpoint model.
-   **FP16 VAE:** Uses a dedicated, memory-efficient VAE (`madebyollin/sdxl-vae-fp16-fix`).
-   **CPU Offload:** Refers to the `enable_cpu_offload` setting.

#### 🚀 SUPIR Engine

| Precision | CPU Offload | VAE Type | Peak VRAM Usage | System RAM |
| :--- | :--- | :--- | :--- | :--- |
| **BFloat16** | ❌ No | Default | ~14.5 GB | ~13 GB |
| | ✅ Yes | Default / FP16 | ~9.5 GB | ~13 GB |
| **Float16** | ❌ No | Default | ~16.0 GB | ~13 GB |
| | ❌ No | FP16 Fix | ~13.5 GB | ~13 GB |
| **FP8 / INT8** | ❌ No | FP16 Fix | **~11.0 GB** | ~13 GB |
| | ✅ Yes | Default / FP16 | **~7.0 GB** | ~13 GB |

#### 🎨 FaithDiff Engine

| Precision | CPU Offload | VAE Type | Peak VRAM Usage | System RAM |
| :--- | :--- | :--- | :--- | :--- |
| **BFloat16** | ❌ No | Default / FP16 Fix | ~9.5 GB | ~14 GB |
| | ✅ Yes | Default / FP16 Fix | ~7.0 GB | ~14 GB |
| **Float16** | ❌ No | Default | ~10.5 GB | ~14 GB |
| | ❌ No | FP16 Fix | ~9.5 GB | ~14 GB |
| | ✅ Yes | Default / FP16 Fix | 7.0 - 9.0 GB | ~12 GB |
| **FP8 / INT8**| ❌ No | Default | ~10.5 GB | ~14 GB |
| | ❌ No | FP16 Fix | **~6.7 GB** | ~14 GB |
| | ✅ Yes | Default | 9.0 - 13.0 GB | ~12 GB |
| | ✅ Yes | FP16 Fix | **~4.5 GB** | ~12 GB |

#### 🧩 ControlNetTile Engine

| Precision | CPU Offload | VAE Type | Peak VRAM Usage | System RAM |
| :--- | :--- | :--- | :--- | :--- |
| **BFloat16 / Float16** | ❌ No | FP16 Fix | ~11.0 GB | ~11 GB |
| | ✅ Yes | FP16 Fix | ~9.0 GB | ~11 GB |
| **FP8 / INT8** | ✅ Yes | FP16 Fix | **~6.5 GB** | ~10-12 GB |

### Key Takeaways
-   Using the **FP16 Fix VAE** provides a significant and consistent reduction in VRAM usage.
-   **CPU Offloading** is the most effective way to reduce VRAM, allowing powerful models to run on cards with less memory (e.g., 8-12 GB).
-   **INT8/FP8 Quantization** with CPU Offloading offers the lowest VRAM footprint.

</details>

## Engine Deep Dives

For a more detailed explanation of each processing engine, including visual comparisons and specific parameter tuning advice, please refer to the documents below:

-   **Engine Deep Dive: SUPIR** (TODO)
-   **Engine Deep Dive: FaithDiff** (TODO)
-   **[Engine Deep Dive: ControlNetTile](./docs/controlnet_tile.md)**

## Releases and Changelog

For a detailed list of changes, new features, and bug fixes for each version, please see the **[CHANGELOG.md](CHANGELOG.md)** file.

## Licensing

This project is released under a hybrid licensing model. It is crucial to understand the terms governing each component before using this software.

### Main Project License

The original code for this project, including the Gradio interface, the main controller, and the overall structure, is licensed under the **Apache License, Version 2.0**. You can find the full license text in the `LICENSE` file at the root of this repository.

### Third-Party Components & Special Conditions

This toolbox integrates several powerful third-party modules, each governed by its own license. For a complete and detailed list of all components and their respective copyrights, please see the `NOTICE` file.

The most important component to be aware of is **SUPIR**.

> **⚠️ Important Notice Regarding SUPIR (Non-Commercial Use Only)**
>
> The SUPIR module in this toolbox is an adaptation of the official implementation for the research paper: *"Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild"* by Yu et al.
>
> In accordance with the permission granted by the authors and their "Non-Commercial Use Only Declaration," the use of this component is strictly governed by its original license. This license **PROHIBITS ANY USE** that is "primarily intended for or directed towards commercial advantage or monetary compensation."
>
> By using the SUPIR functionality within this toolbox, you agree to adhere to these non-commercial terms.
>
> For any inquiries or to obtain permission for commercial use, you must contact Dr. Jinjin Gu at `jinjin.gu@suppixel.ai`.
>
> We are required to provide a clear link back to the original project, which you can find here: **[Original SUPIR Project](https://github.com/Fanghua-Yu/SUPIR)**. The full license text provided by the authors is included in the SUPIR module directory.

Other key components include:
*   **[FaithDiff](https://github.com/JyChen9811/FaithDiff):** Licensed under the MIT License.
*   **[Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers):** Licensed under the MIT License.
*   **IPAdapter & ControlNet pipelines:** Based on code licensed under the Apache License, Version 2.0.

**By cloning, installing, or using this software in any way, you acknowledge that you have read and agree to comply with the licensing terms of all its constituent components.**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEVAIEXP/sup-toolbox&type=Date)](https://star-history.com/#DEVAIEXP/sup-toolbox&Date)

## Contact
If you have any questions, please contact: contact@devaiexp.com