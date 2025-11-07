# Engine Deep Dive: ControlNetTile Upscaler

The `ControlNetTile` engine within **SUP-Toolbox** is an advanced upscaling pipeline that leverages the **ControlNet Tile** model and **Mixture-of-Diffusers (MoD)** techniques. This method integrates tile-based diffusion directly into the latent space denoising process, overcoming the limitations of conventional pixel-space tiling.

By processing tiles in the latent space, this engine ensures smoother transitions, eliminates visible seams, and optimizes resource usage, making it an excellent choice for high-resolution upscaling tasks.

### Key Features

-   **Latent Space Tiling:** Tiles are processed directly in the latent space, improving efficiency and final image quality.
-   **Seamless Transitions:** Advanced weighting methods (e.g., Gaussian or Cosine) ensure smooth blending between tiles, preventing visible grid artifacts.
-   **High Scalability:** Designed to handle large-scale image upscaling with consistent quality.
-   **Detail Preservation:** The ControlNet guides the diffusion process to faithfully preserve and enhance details from the low-resolution input image.

### Why Use the ControlNetTile Engine?

-   **High-Quality Upscaling:** Latent space processing avoids common artifacts like seams and inconsistencies between tiles.
-   **Detail Coherence:** Ideal for upscaling images where preserving the original structure and detail is critical.
-   **Efficient for Large Images:** Provides a robust solution for generating high-resolution outputs that might otherwise exceed VRAM limits.

---

### Method Comparison

The conventional approach to tiled upscaling often processes tiles in pixel space, which can lead to visible seams and inconsistencies. The `ControlNetTile` engine in SUP-Toolbox avoids these issues by working in the latent space.

The images below highlight the difference. On the left, common artifacts from older methods; on the right, the clean result from our latent-space approach.

| Conventional Method: Visible Seams & Inconsistency | **Our Method: Seamless & Consistent** |
| :---: | :---: |
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_control_tile_borders.png" width="400"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled_tile_borders.png" width="400"> |

---

### Key Parameters in SUP-Toolbox

When using the `ControlNetTile` engine in the UI or CLI, these are the most important parameters to configure:

| Parameter | CLI Override Suffix | Description (from UI) |
| :--- | :--- | :--- |
| **Denoising Strength** | `-strength <FLOAT>` | How much to denoise the original image. Lower values preserve more of the original image structure, while higher values allow for more creative details. 0.65 is a good starting point. |
| **ControlNet Scale** | `-controlnet-scale <FLOAT>`| The weight of the ControlNet Tile guidance. This controls how strictly the upscaled output adheres to the details of the input image. |
| **Tile Overlap** | `-tile-overlap <INT>` | Overlap in pixels between tiles to reduce seams. A value between 64 and 128 is typical. |
| **Tile Weighting Method**| `-tile-weighting-method <METHOD>`| The method used for blending tile edges. `Cosine` is generally a good default. `Gaussian` is an alternative. |
| **Tile Gaussian Sigma** | `-tile-gaussian-sigma <FLOAT>`| [If using Gaussian] The sigma parameter for the Gaussian weighting of tiles. Only active when the weighting method is set to `Gaussian`. |

### Example Results

Here are some examples of images upscaled using the `ControlNetTile` engine within the SUP-Toolbox pipeline.

| Example 1: 1024px -> 2x |
| :---: |
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_sample_1x_to_2x.PNG"> |
| <a href="https://imgsli.com/NDI2NDU4">View Comparison</a> | 

| Example 2: 1024 -> 8x |
| :---: |
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/2_sample_1x_to_8x.PNG"> |
| <a href="https://imgsli.com/NDI2NDYz">View Comparison</a> | 

| Example 3: 512px -> 4x |
| :---: |
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/3_sample_1x_to_4x.PNG"> |
| <a href="https://imgsli.com/NDI2NDYw">View Comparison</a> | 

---
