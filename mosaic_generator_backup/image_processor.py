"""Image preparation utilities (loading, resizing, grid analysis, quantization)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

try:
    from .config import Config
    from .utils import cell_means, pil_to_np, np_to_pil, resize_and_crop_to_grid
except ImportError:  # pragma: no cover - script execution fallback
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from config import Config  # type: ignore  # noqa: E402
    from utils import cell_means, pil_to_np, np_to_pil, resize_and_crop_to_grid  # type: ignore  # noqa: E402


def _get_kmeans_cls():
    """Import KMeans lazily so the package can load without scikit-learn."""
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as exc:  # pragma: no cover - import error propagated to caller
        raise ImportError(
            "scikit-learn is required for K-means quantization. Install it to enable this option."
        ) from exc
    return KMeans


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk and convert it to RGB."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def prepare_image(image: Image.Image, config: Config) -> Image.Image:
    """Resize an image to the configured output size and apply optional quantization."""
    processed = resize_and_crop_to_grid(image, config.out_w, config.out_h, config.grid)
    if config.use_uniform_q or config.use_kmeans_q:
        processed = apply_color_quantization(processed, config)
    return processed


def analyze_grid_cells(image: Image.Image, grid_size: int) -> np.ndarray:
    """Return weighted mean colors for every grid cell."""
    img_array = pil_to_np(image)
    return cell_means(img_array, grid_size)


def apply_uniform_quantization(image: Image.Image, levels: int) -> Image.Image:
    """Reduce colors by mapping each channel to evenly spaced uniform buckets."""
    img_array = pil_to_np(image)
    quantized = np.zeros_like(img_array)
    for channel in range(3):
        channel_data = img_array[:, :, channel]
        quantized_channel = np.round(channel_data * (levels - 1)) / (levels - 1)
        quantized[:, :, channel] = np.clip(quantized_channel, 0, 1)
    return np_to_pil(quantized)


def apply_kmeans_quantization(image: Image.Image, k_colors: int) -> Image.Image:
    """Use K-means clustering to approximate the image with k representative colors."""
    img_array = pil_to_np(image)
    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, c)
    KMeans = _get_kmeans_cls()
    kmeans = KMeans(n_clusters=k_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_img = quantized_pixels.reshape(h, w, c)
    return np_to_pil(quantized_img)


def apply_color_quantization(image: Image.Image, config: Config) -> Image.Image:
    """Apply whichever quantization strategy is enabled in the configuration."""
    if config.use_uniform_q:
        return apply_uniform_quantization(image, config.q_levels)
    if config.use_kmeans_q:
        return apply_kmeans_quantization(image, config.k_colors)
    return image


def analyze_quantization_effect(original: Image.Image, quantized: Image.Image) -> dict:
    """Compute quality metrics that describe how quantization changed the image."""
    orig_array = pil_to_np(original)
    quant_array = pil_to_np(quantized)
    diff = np.abs(orig_array - quant_array)
    mse = np.mean((orig_array - quant_array) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
    orig_colors = len(np.unique(orig_array.reshape(-1, 3), axis=0))
    quant_colors = len(np.unique(quant_array.reshape(-1, 3), axis=0))
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "mean_difference": float(np.mean(diff)),
        "max_difference": float(np.max(diff)),
        "original_colors": orig_colors,
        "quantized_colors": quant_colors,
        "color_reduction_ratio": orig_colors / quant_colors if quant_colors > 0 else float("inf"),
    }
