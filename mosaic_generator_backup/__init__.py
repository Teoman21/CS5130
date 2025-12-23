"""Mosaic_Gnerator_Improved package.

This reorganized module mirrors the functionality of the original
`Mosaic_Generator` project but exposes the public API under the new
flat module layout requested for the assignment.
"""

from .config import Config, Implementation, MatchSpace
from .image_processor import (
    load_image,
    prepare_image,
    analyze_grid_cells,
    apply_color_quantization,
    apply_uniform_quantization,
    apply_kmeans_quantization,
    analyze_quantization_effect,
)
from .metrics import (
    calculate_comprehensive_metrics,
    calculate_color_similarity,
    calculate_mae,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    interpret_metrics,
)
from .mosaic_builder import MosaicBuilder, MosaicPipeline
from .tile_manager import TileManager
from .utils import pil_to_np, np_to_pil, resize_and_crop_to_grid, cell_means

__all__ = [
    "Config",
    "Implementation",
    "MatchSpace",
    "load_image",
    "prepare_image",
    "analyze_grid_cells",
    "apply_color_quantization",
    "apply_uniform_quantization",
    "apply_kmeans_quantization",
    "analyze_quantization_effect",
    "calculate_comprehensive_metrics",
    "calculate_color_similarity",
    "calculate_mae",
    "calculate_mse",
    "calculate_psnr",
    "calculate_ssim",
    "interpret_metrics",
    "MosaicBuilder",
    "MosaicPipeline",
    "TileManager",
    "pil_to_np",
    "np_to_pil",
    "resize_and_crop_to_grid",
    "cell_means",
]
