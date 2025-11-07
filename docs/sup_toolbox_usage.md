
## Library Usage: Configuration Objects

When using `sup-toolbox` as a Python library, you interact with a main `Config` object. This object and its nested dataclasses hold all the parameters for the pipeline. This section documents the key configuration classes.

---

### 1. The Main `Config` Object

This is the top-level object that you pass to the `SUPToolBoxPipeline`. It controls which engines are active, which models are used, and holds global settings for paths, hardware, and optimizations.

#### **Core Pipeline Attributes**
| Attribute | Type | Description |
| :--- | :--- | :--- |
| `restorer_engine` | `RestorerEngine` | Selects the engine for the restoration phase. |
| `upscaler_engine` | `UpscalerEngine` | Selects the engine for the upscaling phase. |
| `selected_restorer_checkpoint_model`| `str` | The name of the main checkpoint model for the restorer. |
| `selected_upscaler_checkpoint_model` | `str` | The name of the main checkpoint model for the upscaler. |
| `selected_vae_model` | `str` | The name of the VAE model to use. |
| `selected_restorer_sampler` | `Sampler` | The sampler algorithm to use for the restorer. |
| `selected_upscaler_sampler` | `Sampler` | The sampler algorithm to use for the upscaler. |
| `restorer_supir_model` | `SUPIRModel` | [SUPIR Only] The specific SUPIR model variant (`Quality` or `Fidelity`) for the restorer. |
| `upscaler_supir_model` | `SUPIRModel` | [SUPIR Only] The specific SUPIR model variant for the upscaler. |
| `restore_face` | `bool` | Enables selective face restoration during the restoration phase. |
| `mask_prompt` | `str` | The prompt used to generate a mask for face restoration (e.g., 'face'). |
| `image_path` | `str` | Path to the input image or directory of images. |

#### **File Paths and Saving**
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `output_dir` | `str` | Directory where generated images will be saved. |
| `cache_dir` | `str` | Path to a directory to download and cache pre-trained model weights. |
| `checkpoints_dir` | `str` | Directory path for local `.safetensors` model weights. |
| `vae_dir` | `str` | Directory path for local VAE `.safetensors` model weights. |
| `save_image_format` | `str` | Default format for saving generated images (e.g., `.png`, `.jpg`). |
| `save_image_on_upscaling_passes` | `bool` | Saves intermediate images between progressive upscaling passes. |

#### **Hardware & Performance Optimizations**
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `device` | `RuntimeDevice` | The primary device (`cuda`, `cpu`) for running the pipelines. |
| `generator_device` | `RuntimeDevice` | The primary device (`cuda`, `cpu`) for generator seeds. |
| `weight_dtype` | `WeightDtype` | Precision for loading model weights (`Float16`, `Bfloat16`, `Float32`). |
| `vae_weight_dtype` | `WeightDtype` | Precision for loading the VAE weights. `Float16` is often faster. |
| `enable_cpu_offload` | `bool` | Saves VRAM by keeping models on the CPU and moving them to the GPU only when needed. |
| `always_offload_models` | `bool` | Offloads models to CPU immediately after finishing inference to free up VRAM. |
| `run_vae_on_cpu` | `bool` | Runs the VAE on the CPU to save GPU memory. |
| `enable_vae_tiling` | `bool` | Processes the VAE in tiles to decode large images with less VRAM. |
| `enable_vae_slicing` | `bool` | Processes image batches one slice at a time through the VAE to save VRAM. |
| `memory_attention` | `MemoryAttention` | Optimized attention mechanism to save VRAM and increase speed (`xformers` recommended). |
| `quantization_method`| `QuantizationMethod`| Quantization mechanism to save VRAM and increase speed (`None`, `Quanto Library`, `Layerwise & Bnb`). |
| `quantization_mode` | `QuantizationMode`| Quantization mode (`FP8`, `NF4`). |
| `allow_cuda_tf32` | `bool` | Enable TensorFloat-32 for matrix multiplications on Ampere+ GPUs. |
| `allow_cudnn_tf32` | `bool` | Enable TensorFloat-32 for cuDNN convolutions on Ampere+ GPUs. |
| `disable_mmap` | `bool` | Disable memory-mapping for loading model files. Enable this for shared/network drives. |

#### **LLaVA Captioning Settings**
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `enable_llava_quantization` | `bool` | Enable quantization for LLaVA models to save VRAM. |
| `llava_quantization_mode` | `QuantizationMode` | Quantization mode used for the LLaVA model (`INT4`, `INT8`). |
| `llava_offload_model` | `bool` | Offload the LLaVA model to the CPU to save GPU memory. |
| `llava_weight_dtype` | `WeightDtype` | Precision for loading LLaVA model weights. |
| `llava_question_prompt` | `str` | The prompt used to query LLaVA for image descriptions. |

---

### 2. `PipelineParams` Object

The `config.restorer_pipeline_params` and `config.upscaler_pipeline_params` attributes are instances of this class. They hold the detailed parameters for the inference process of each engine.

**Key Attributes:**

#### General & Generation
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `seed` | `int` | The random seed for generation. -1 means a random seed. |
| `num_steps` | `int` | Number of denoising steps. More steps can improve quality but take longer. |
| `strength` | `float` | How much to denoise the original image. 1.0 is full denoising. |
| `guidance_scale` | `float` | How strongly the prompt is adhered to (CFG Scale). |
| `guidance_rescale` | `float` | Guidance rescale factor (phi). |
| `num_images` | `int` | Number of output images to generate. |
| `image_size_fix_mode` | `ImageSizeFixMode` | Method to handle aspect ratio mismatches during processing. |
| `tile_size` | `int` | The size (in pixels) of the square tiles to use when latent tiling is enabled. |
| `upscale_factor` | `str` | Resolution upscale factor (e.g., '2x', '4x'). |
| `upscaling_mode` | `UpscalingMode` | `Progressive` for quality or `Direct` for speed. |
| `cfg_decay_rate` | `float` | [Progressive Upscale] Percentage to reduce CFG at each pass. |
| `strength_decay_rate` | `float` | [Progressive Upscale] Percentage to reduce Denoising Strength at each pass. |

#### Prompting
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `prompt` | `str` | The main positive prompt. |
| `prompt_2` | `str` | A secondary positive prompt, often for style and quality keywords. |
| `negative_prompt` | `str` | The negative prompt, specifying what to avoid. |
| `apply_prompt_2` | `bool` | If enabled, concatenates prompt 2 with prompt 1. |
| `use_lpw_prompt` | `bool` | Set based on the `prompt_method`. Enables long prompt weighting. |
| `invert_prompts` | `bool` | [FaithDiff Only] Starts the prompt with prompt 2 and ends with prompt 1. |
| `apply_ipa_embeds`| `bool` | [FaithDiff Only] Apply IP-Adapter embeddings during diffusion. |

#### Engine-Specific
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `restoration_scale` | `float` | [SUPIR Only] Strength of the SUPIR restoration guidance. |
| `s_churn` | `float` | [SUPIR/FaithDiff] Stochasticity churn factor. Adds extra noise at each step. |
| `s_noise` | `float` | [SUPIR/FaithDiff] Stochasticity noise factor. |
| `start_point` | `str` | [SUPIR/FaithDiff] Start from low-res latents ('lr') or pure noise ('noise'). |
| `tile_overlap` | `int` | [ControlNetTile Only] Overlap in pixels between tiles to reduce seams. |
| `tile_weighting_method`| `str` | [ControlNetTile Only] Method for blending tile edges (`Cosine` or `Gaussian`). |
| `tile_gaussian_sigma`| `float` | [ControlNetTile Only] Sigma parameter for Gaussian weighting. |

#### Advanced Guidance (CFG, ControlNet, PAG)
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `use_linear_CFG` | `bool` | Linearly increase CFG scale during sampling. |
| `guidance_scale_start`| `float` | The starting value for linear CFG scaling. |
| `controlnet_conditioning_scale`| `float` | The weight of the ControlNet guidance. |
| `use_linear_control_scale`| `bool` | Linearly increase ControlNet scale during sampling. |
| `enable_PAG` | `bool` | Enable Perturbed Attention Guidance. |
| `pag_scale` | `float` | The scale factor for the perturbed attention guidance. |
| `pag_layers` | `List[str]` | The UNet layers where PAG should be applied. |

#### Post-Processing
| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `color_fix_mode` | `ColorFix` | Applies the color profile of the input image to the output (`None`, `Adain`, `Wavelet`). |

---

### 3. `SchedulerConfig` Object

The `config.restorer_sampler_config` and `config.upscaler_sampler_config` attributes are instances of this class.

| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `scale_linear_exponent` | `float` | Exponent for the scale linear beta schedule. |
| `beta_schedule` | `str` | The beta schedule type (`linear`, `scaled_linear`, etc.). |
| `timestep_spacing`| `str` | The timestep spacing method (`linspace`, `leading`, `trailing`). |

---

### 4. SUPIR Advanced Injection Objects

For the `SUPIR` engine, you can provide advanced SFT injection settings. These are set on the `PipelineParams` object.

-   **`zero_sft_injection_flags`**: An instance of `InjectionFlags`. This dataclass contains boolean flags (`sft_post_mid_active`, etc.) to enable or disable injection at each specific stage.
-   **`zero_sft_injection_configs`**: An instance of `InjectionConfigs`. This dataclass contains nested `InjectionScaleConfig` objects (`sft_post_mid`, etc.) that control the behavior of the injection at each stage.

**`InjectionScaleConfig` Attributes:**

| Attribute | Type | Description (from UI) |
| :--- | :--- | :--- |
| `scale_end` | `float` | The final weight of the ControlNet guidance for this stage. |
| `linear` | `bool` | Linearly increase Control scale during this stage's sampling. |
| `scale_start`| `float` | The starting value for linear ControlNet guidance. |
| `reverse` | `bool` | Linearly decrease ControlNet scale during this stage's sampling. |

---