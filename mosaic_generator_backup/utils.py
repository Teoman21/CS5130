"""Shared utility helpers for array/PIL interactions and grid math."""

from __future__ import annotations

import numpy as np
from PIL import Image


def pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized NumPy array in RGB format."""
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    if img.mode == "L":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr / 255.0


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a normalized NumPy array back into a PIL image."""
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def resize_and_crop_to_grid(img: Image.Image, width: int, height: int, grid: int) -> Image.Image:
    """Resize an image to the requested resolution and crop it so both sides divide evenly by the grid."""
    img = img.convert("RGB").resize((width, height), Image.LANCZOS)
    H, W = img.height, img.width
    H2, W2 = (H // grid) * grid, (W // grid) * grid
    if H2 != H or W2 != W:
        left = (W - W2) // 2
        top = (H - H2) // 2
        img = img.crop((left, top, left + W2, top + H2))
    return img


def block_view(arr: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """Return a contiguous view of the array split into (bh × bw) tiles."""
    H, W, C = arr.shape
    if H % bh or W % bw:
        raise ValueError("Array dimensions must be divisible by block size")
    grid_h = H // bh
    grid_w = W // bw
    # Reshape to (grid_h, bh, grid_w, bw, C) and swap axes so grid dims lead.
    return arr.reshape(grid_h, bh, grid_w, bw, C).swapaxes(1, 2)


def cell_means(arr: np.ndarray, grid: int) -> np.ndarray:
    """Compute weighted mean colors for every grid cell using pure NumPy ops."""
    H, W, _ = arr.shape
    bh, bw = H // grid, W // grid
    blocks = block_view(arr, bh, bw)  # (grid, grid, bh, bw, 3)

    center_h = (bh - 1) / 2.0
    center_w = (bw - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    dist = np.sqrt((yy - center_h) ** 2 + (xx - center_w) ** 2)
    max_dist = np.sqrt(center_h**2 + center_w**2) or 1.0
    weights = 1.0 - (dist / max_dist) * 0.5
    weights = weights.astype(np.float32)
    weights /= np.sum(weights)

    weighted = blocks * weights[None, None, :, :, None]
    return weighted.sum(axis=(2, 3))
