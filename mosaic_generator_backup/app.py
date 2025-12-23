"""Standalone Gradio app that mirrors Mosaic_Generator/src/gradio_interface.py."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
from PIL import Image

try:
    from .config import Config, Implementation, MatchSpace
    from .mosaic_builder import MosaicPipeline
except ImportError:  # pragma: no cover - script execution fallback
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from config import Config, Implementation, MatchSpace  # type: ignore  # noqa: E402
    from mosaic_builder import MosaicPipeline  # type: ignore  # noqa: E402


def create_default_config(
    grid_size: int = 32,
    tile_size: int = 32,
    output_width: int = 768,
    output_height: int = 768,
    color_matching: str = "Lab (perceptual)",
    use_uniform_quantization: bool = False,
    quantization_levels: int = 8,
    use_kmeans_quantization: bool = False,
    kmeans_colors: int = 8,
    normalize_tile_brightness: bool = False,
) -> Config:
    """Build a Config dataclass from UI inputs."""
    match_space = MatchSpace.LAB if color_matching == "Lab (perceptual)" else MatchSpace.RGB
    return Config(
        grid=grid_size,
        tile_size=tile_size,
        out_w=output_width,
        out_h=output_height,
        impl=Implementation.VECT,
        match_space=match_space,
        use_uniform_q=use_uniform_quantization,
        q_levels=quantization_levels,
        use_kmeans_q=use_kmeans_quantization,
        k_colors=kmeans_colors,
        tile_norm_brightness=normalize_tile_brightness,
    )


def generate_mosaic(
    image: Image.Image,
    grid_size: int,
    tile_size: int,
    output_width: int,
    output_height: int,
    color_matching: str,
    use_uniform_quantization: bool,
    quantization_levels: int,
    use_kmeans_quantization: bool,
    kmeans_colors: int,
    normalize_tile_brightness: bool,
    progress=gr.Progress(),
) -> Tuple[Image.Image | None, Image.Image | None, str, str]:
    """Gradio callback that runs the pipeline and formats UI-friendly text."""
    if image is None:
        return None, None, "Please upload an image.", ""

    try:
        config = create_default_config(
            grid_size,
            tile_size,
            output_width,
            output_height,
            color_matching,
            use_uniform_quantization,
            quantization_levels,
            use_kmeans_quantization,
            kmeans_colors,
            normalize_tile_brightness,
        )
        pipeline = MosaicPipeline(config)

        progress(0.15, desc="Loading tiles (first run may take longer)...")
        results = pipeline.run_full_pipeline(image)
        progress(0.8, desc="Calculating metrics...")

        mosaic_img = results["outputs"]["mosaic"]
        processed_img = results["outputs"]["processed_image"]
        metrics = results["metrics"]
        interpretations = results["metrics_interpretation"]

        metrics_text = f"""
**Quality Metrics:**
- **MSE:** {metrics['mse']:.6f} — {interpretations['mse']}
- **PSNR:** {metrics['psnr']:.2f} dB — {interpretations['psnr']}
- **SSIM:** {metrics['ssim']:.4f} — {interpretations['ssim']}
- **RMSE:** {metrics['rmse']:.6f}
- **MAE:** {metrics['mae']:.6f}

**Color Analysis:**
- **Color MSE:** {metrics['color_mse']:.6f}
- **Histogram Correlation:** {metrics['histogram_correlation']:.4f}
"""

        timing = results["timing"]
        timing_text = f"""
**Processing Times:**
- **Preprocessing:** {timing['preprocessing']:.3f}s
- **Grid Analysis:** {timing['grid_analysis']:.3f}s
- **Tile Mapping:** {timing['tile_mapping']:.3f}s
- **Total:** {timing['total']:.3f}s

**Configuration:**
- **Grid Size:** {config.grid}×{config.grid} ({config.grid**2} tiles)
- **Tile Size:** {config.tile_size}px
- **Output Resolution:** {mosaic_img.width}×{mosaic_img.height}
- **Color Matching:** {config.match_space.value}
"""

        progress(1.0, desc="Complete!")
        return mosaic_img, processed_img, metrics_text, timing_text

    except Exception as exc:  # pragma: no cover - surfaced to UI
        error_msg = f"Error generating mosaic: {exc}"
        print(error_msg)
        return None, None, error_msg, ""


def benchmark_grid_sizes(image: Image.Image, grid_sizes: str, progress=gr.Progress()) -> str:
    """Benchmark the pipeline for several comma-separated grid sizes."""
    if image is None:
        return "Please upload an image for benchmarking."

    try:
        parsed_sizes = [int(value.strip()) for value in grid_sizes.split(",") if value.strip()]
        if not parsed_sizes:
            return "Provide at least one grid size (comma separated)."

        report_lines: List[str] = ["**Grid Size Performance Analysis:**", ""]
        total = len(parsed_sizes)
        run_details: List[Dict[str, float]] = []

        for idx, size in enumerate(parsed_sizes):
            progress((idx + 1) / total, desc=f"Benchmarking grid {size}×{size}...")
            custom_config = create_default_config(grid_size=size)
            pipeline = MosaicPipeline(custom_config)
            start = time.time()
            run_results = pipeline.run_full_pipeline(image)
            elapsed = time.time() - start
            report_lines.append(f"**Grid {size}×{size}:**")
            report_lines.append(f"- Processing Time: {elapsed:.3f}s")
            report_lines.append(f"- Total Tiles: {size * size}")
            report_lines.append(f"- Tiles per Second: {(size * size) / elapsed:.1f}")
            report_lines.append(f"- MSE: {run_results['metrics']['mse']:.6f}")
            report_lines.append(f"- SSIM: {run_results['metrics']['ssim']:.4f}")
            report_lines.append("")
            run_details.append(
                {
                    "grid_size": size,
                    "processing_time": elapsed,
                    "total_tiles": size * size,
                }
            )

        if len(run_details) >= 2:
            first = run_details[0]
            last = run_details[-1]
            tile_ratio = last["total_tiles"] / first["total_tiles"] if first["total_tiles"] else 0.0
            time_ratio = last["processing_time"] / first["processing_time"] if first["processing_time"] else 0.0
            report_lines.append("**Scaling Analysis:**")
            report_lines.append(f"- Tile increase ratio: {tile_ratio:.2f}×")
            report_lines.append(f"- Time increase ratio: {time_ratio:.2f}×")
            efficiency = tile_ratio / time_ratio if time_ratio else 0.0
            report_lines.append(f"- Scaling efficiency: {efficiency:.2f}")
            is_linear = abs(time_ratio - tile_ratio) / tile_ratio < 0.1 if tile_ratio else False
            report_lines.append(f"- Linear scaling: {'Yes' if is_linear else 'No'}")

        return "\n".join(report_lines)

    except Exception as exc:  # pragma: no cover - surfaced to UI
        return f"Error during grid size benchmarking: {exc}"


def create_interface() -> gr.Blocks:
    """Construct the Gradio Blocks layout shared by both projects."""
    with gr.Blocks(title="Mosaic Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Mosaic Generator")
        gr.Markdown("Generate photomosaics backed by the reorganized Mosaic_Gnerator_Improved pipeline.")

        with gr.Tab("Generate Mosaic"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Upload & Configure")
                    input_image = gr.Image(type="pil", label="Upload Image", height=300)

                    with gr.Accordion("Basic Settings", open=True):
                        grid_size = gr.Slider(minimum=8, maximum=128, step=8, value=32, label="Grid Size (N×N)")
                        tile_size = gr.Slider(minimum=4, maximum=64, step=4, value=32, label="Tile Size (px)")
                        output_width = gr.Slider(minimum=256, maximum=1024, step=64, value=768, label="Output Width")
                        output_height = gr.Slider(minimum=256, maximum=1024, step=64, value=768, label="Output Height")

                    with gr.Accordion("Advanced Settings", open=False):
                        color_matching = gr.Radio(
                            choices=["Lab (perceptual)", "RGB (euclidean)"],
                            value="Lab (perceptual)",
                            label="Color Matching Space",
                        )
                        use_uniform_quantization = gr.Checkbox(label="Use Uniform Quantization", value=False)
                        quantization_levels = gr.Slider(minimum=4, maximum=16, step=2, value=8, label="Quantization Levels")
                        use_kmeans_quantization = gr.Checkbox(label="Use K-means Quantization", value=False)
                        kmeans_colors = gr.Slider(minimum=4, maximum=32, step=2, value=8, label="K-means Colors")
                        normalize_tile_brightness = gr.Checkbox(label="Normalize Tile Brightness", value=False)

                    generate_btn = gr.Button("Generate Mosaic", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("## Results")
                    with gr.Row():
                        mosaic_output = gr.Image(label="Generated Mosaic", height=400)
                        processed_output = gr.Image(label="Processed Input", height=400)
                    with gr.Row():
                        metrics_output = gr.Markdown(label="Quality Metrics")
                        timing_output = gr.Markdown(label="Processing Information")

        with gr.Tab("Performance Analysis"):
            gr.Markdown("## Performance Benchmarking")
            with gr.Row():
                with gr.Column():
                    benchmark_image = gr.Image(type="pil", label="Image for Benchmarking", height=200)
                    grid_sizes_input = gr.Textbox(value="16,32,48,64", label="Grid Sizes (comma-separated)")
                    benchmark_grid_btn = gr.Button("Benchmark Grid Sizes", variant="secondary")
                with gr.Column():
                    benchmark_output = gr.Markdown(label="Benchmark Results")

        with gr.Tab("About"):
            gr.Markdown(
                """
            ## About
            This UI exposes the same functionality as `Mosaic_Generator/src/gradio_interface.py`
            but imports everything from the reorganized `Mosaic_Gnerator_Improved` package.
            """
            )

        generate_btn.click(
            fn=generate_mosaic,
            inputs=[
                input_image,
                grid_size,
                tile_size,
                output_width,
                output_height,
                color_matching,
                use_uniform_quantization,
                quantization_levels,
                use_kmeans_quantization,
                kmeans_colors,
                normalize_tile_brightness,
            ],
            outputs=[mosaic_output, processed_output, metrics_output, timing_output],
        )

        benchmark_grid_btn.click(
            fn=benchmark_grid_sizes,
            inputs=[benchmark_image, grid_sizes_input],
            outputs=[benchmark_output],
        )

        use_uniform_quantization.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=use_uniform_quantization,
            outputs=quantization_levels,
        )
        use_kmeans_quantization.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=use_kmeans_quantization,
            outputs=kmeans_colors,
        )

    return demo


demo = create_interface()


if __name__ == "__main__":
    demo.launch()
