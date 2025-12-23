"""Image quality metrics copied from the original Mosaic_Generator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

try:
    from .utils import pil_to_np
except ImportError:  # pragma: no cover - script execution fallback
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from utils import pil_to_np  # type: ignore  # noqa: E402


def calculate_mse(original: Image.Image, reconstructed: Image.Image) -> float:
    """Return the mean squared error between the original image and its reconstruction."""
    orig_array = pil_to_np(original)
    recon_array = pil_to_np(reconstructed)
    if orig_array.shape != recon_array.shape:
        recon_pil = reconstructed.resize(original.size, Image.LANCZOS)
        recon_array = pil_to_np(recon_pil)
    mse = np.mean((orig_array - recon_array) ** 2)
    return float(mse)


def calculate_psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    """Return the peak signal-to-noise ratio in decibels."""
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def calculate_ssim(original: Image.Image, reconstructed: Image.Image) -> float:
    """Return the structural similarity index between two images."""
    orig_array = pil_to_np(original)
    recon_array = pil_to_np(reconstructed)
    if orig_array.shape != recon_array.shape:
        recon_pil = reconstructed.resize(original.size, Image.LANCZOS)
        recon_array = pil_to_np(recon_pil)
    orig_gray = np.mean(orig_array, axis=2) if orig_array.ndim == 3 else orig_array
    recon_gray = np.mean(recon_array, axis=2) if recon_array.ndim == 3 else recon_array
    return float(ssim(orig_gray, recon_gray, data_range=1.0))


def calculate_color_similarity(original: Image.Image, reconstructed: Image.Image) -> Dict[str, float]:
    """Return per-channel color errors and histogram correlation."""
    orig_array = pil_to_np(original)
    recon_array = pil_to_np(reconstructed)
    if orig_array.shape != recon_array.shape:
        recon_pil = reconstructed.resize(original.size, Image.LANCZOS)
        recon_array = pil_to_np(recon_pil)
    channel_diffs = []
    for channel in range(3):
        orig_channel = orig_array[:, :, channel]
        recon_channel = recon_array[:, :, channel]
        channel_mse = np.mean((orig_channel - recon_channel) ** 2)
        channel_diffs.append(channel_mse)
    color_mse = np.mean(channel_diffs)
    orig_hist = np.histogram(orig_array.flatten(), bins=256, range=(0, 1))[0]
    recon_hist = np.histogram(recon_array.flatten(), bins=256, range=(0, 1))[0]
    orig_hist = orig_hist / np.sum(orig_hist)
    recon_hist = recon_hist / np.sum(recon_hist)
    hist_correlation = np.corrcoef(orig_hist, recon_hist)[0, 1]
    return {
        "color_mse": float(color_mse),
        "red_channel_mse": float(channel_diffs[0]),
        "green_channel_mse": float(channel_diffs[1]),
        "blue_channel_mse": float(channel_diffs[2]),
        "histogram_correlation": float(hist_correlation) if not np.isnan(hist_correlation) else 0.0,
    }


def calculate_mae(original: Image.Image, reconstructed: Image.Image) -> float:
    """Return mean absolute error between two images."""
    orig_array = pil_to_np(original)
    recon_array = pil_to_np(reconstructed)
    if orig_array.shape != recon_array.shape:
        recon_pil = reconstructed.resize(original.size, Image.LANCZOS)
        recon_array = pil_to_np(recon_pil)
    mae = np.mean(np.abs(orig_array - recon_array))
    return float(mae)


def calculate_comprehensive_metrics(original: Image.Image, reconstructed: Image.Image) -> Dict[str, float]:
    """Compute the aggregate suite of metrics used by the UI/tests."""
    metrics = {
        "mse": calculate_mse(original, reconstructed),
        "psnr": calculate_psnr(original, reconstructed),
        "ssim": calculate_ssim(original, reconstructed),
    }
    metrics.update(calculate_color_similarity(original, reconstructed))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = calculate_mae(original, reconstructed)
    return metrics


def interpret_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """Map raw metric values to qualitative descriptors for the UI."""
    interpretations = {}
    mse = metrics.get("mse", 0)
    interpretations["mse"] = "Excellent" if mse < 0.001 else "Good" if mse < 0.005 else "Needs improvement"
    psnr = metrics.get("psnr", 0)
    if psnr == float("inf"):
        interpretations["psnr"] = "Identical images"
    elif psnr > 30:
        interpretations["psnr"] = "Excellent fidelity"
    elif psnr > 20:
        interpretations["psnr"] = "Acceptable"
    else:
        interpretations["psnr"] = "Noticeable differences"
    ssim_value = metrics.get("ssim", 0)
    if ssim_value > 0.9:
        interpretations["ssim"] = "Very high structural similarity"
    elif ssim_value > 0.75:
        interpretations["ssim"] = "Moderate similarity"
    else:
        interpretations["ssim"] = "Significant structural differences"
    return interpretations
