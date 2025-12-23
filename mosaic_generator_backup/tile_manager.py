"""Tile loading, caching, and feature extraction."""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image

try:
    from .config import Config, MatchSpace
    from .utils import pil_to_np
except ImportError:  # pragma: no cover - script execution fallback
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from config import Config, MatchSpace  # type: ignore  # noqa: E402
    from utils import pil_to_np  # type: ignore  # noqa: E402


@dataclass
class TileManager:
    """Manage tile loading, caching, and feature extraction."""
    config: Config
    _tiles_loaded: bool = field(default=False, init=False)
    tiles: List[np.ndarray] = field(default_factory=list, init=False)
    tile_colors: List[np.ndarray] = field(default_factory=list, init=False)
    tile_colors_lab: List[np.ndarray] = field(default_factory=list, init=False)

    _global_cache: ClassVar[Dict[str, dict]] = {}

    def _stable_cache_key(self) -> str:
        """Return a hash that uniquely represents the tile-loading configuration."""
        key = (
            f"ds={self.config.hf_dataset}|split={self.config.hf_split}|limit={self.config.hf_limit}"
            f"|tile={self.config.tile_size}|norm={self.config.tile_norm_brightness}"
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _ensure_tiles_loaded(self) -> None:
        """Populate `self.tiles` from cache, disk, or Hugging Face as needed."""
        if self._tiles_loaded:
            return

        config_hash = self._stable_cache_key()
        if config_hash in TileManager._global_cache:
            cached = TileManager._global_cache[config_hash]
            self.tiles = [tile.copy() for tile in cached["tiles"]]
            self.tile_colors = [color.copy() for color in cached["tile_colors"]]
            self.tile_colors_lab = [color.copy() for color in cached["tile_colors_lab"]]
            self._tiles_loaded = True
            return

        if self.config.tiles_cache_dir:
            os.makedirs(self.config.tiles_cache_dir, exist_ok=True)
            cache_path = os.path.join(self.config.tiles_cache_dir, f"tiles_{config_hash}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)
                self.tiles = cached["tiles"]
                self.tile_colors = cached["tile_colors"]
                self.tile_colors_lab = cached["tile_colors_lab"]
                self._tiles_loaded = True
                TileManager._global_cache[config_hash] = {
                    "tiles": [tile.copy() for tile in self.tiles],
                    "tile_colors": [color.copy() for color in self.tile_colors],
                    "tile_colors_lab": [color.copy() for color in self.tile_colors_lab],
                }
                return

        self._load_tiles_from_source()
        TileManager._global_cache[config_hash] = {
            "tiles": [tile.copy() for tile in self.tiles],
            "tile_colors": [color.copy() for color in self.tile_colors],
            "tile_colors_lab": [color.copy() for color in self.tile_colors_lab],
        }

        if self.config.tiles_cache_dir:
            os.makedirs(self.config.tiles_cache_dir, exist_ok=True)
            cache_path = os.path.join(self.config.tiles_cache_dir, f"tiles_{config_hash}.pkl")
            with open(cache_path, "wb") as fh:
                pickle.dump(
                    {
                        "tiles": self.tiles,
                        "tile_colors": self.tile_colors,
                        "tile_colors_lab": self.tile_colors_lab,
                    },
                    fh,
                )

        self._tiles_loaded = True

    def _load_tiles_from_source(self) -> None:
        """Fetch tiles from Hugging Face (streaming) and pre-compute representative colors."""
        print(f"Loading tiles from {self.config.hf_dataset} ...")
        try:
            dataset = load_dataset(
                self.config.hf_dataset,
                split=self.config.hf_split,
                cache_dir=self.config.hf_cache_dir,
                streaming=True,
            )
            tiles = []
            colors = []
            colors_lab = []
            limit = self.config.hf_limit if self.config.hf_limit is not None else 0
            limit = max(1, limit)
            for idx, sample in enumerate(dataset):
                if idx >= limit:
                    break
                pil_image = None
                if isinstance(sample, dict):
                    if "image" in sample:
                        pil_image = sample["image"]
                    else:
                        # fall back to any PIL Image in the sample
                        for value in sample.values():
                            if isinstance(value, Image.Image):
                                pil_image = value
                                break
                if pil_image is None:
                    continue
                pil_image = pil_image.convert("RGB")
                pil_image = pil_image.resize((self.config.tile_size, self.config.tile_size), Image.LANCZOS)
                tile_np = pil_to_np(pil_image)
                if self.config.tile_norm_brightness:
                    tile_np = self._normalize_brightness(tile_np)
                tiles.append(tile_np)
                colors.append(np.mean(tile_np, axis=(0, 1)))
                colors_lab.append(self._rgb_to_lab(colors[-1]))
            if tiles:
                self.tiles = tiles
                self.tile_colors = colors
                self.tile_colors_lab = colors_lab
                print(f"Loaded {len(self.tiles)} tiles successfully")
                return
            raise RuntimeError("No tiles fetched from dataset")
        except Exception as exc:
            print(f"Failed to load dataset tiles: {exc}")
            self._create_fallback_tiles()

    def _normalize_brightness(self, tile: np.ndarray) -> np.ndarray:
        """Scale the tile so that its mean brightness becomes ~1.0."""
        mean_brightness = np.mean(tile)
        if mean_brightness > 0:
            tile = tile / mean_brightness
        return np.clip(tile, 0, 1)

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert an RGB color (0-1 float) to CIE Lab."""
        r, g, b = rgb

        def gamma_correct(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)

        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        xn, yn, zn = 0.95047, 1.0, 1.08883
        fx, fy, fz = x / xn, y / yn, z / zn

        def f(t):
            return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

        fx, fy, fz = f(fx), f(fy), f(fz)
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_lab = 200 * (fy - fz)
        return np.array([L, a, b_lab], dtype=np.float32)

    def _calculate_perceptual_distance(self, targets: np.ndarray, tiles_lab: np.ndarray) -> np.ndarray:
        """Compute weighted Euclidean distances in Lab space."""
        weights = np.array([2.0, 1.0, 1.0], dtype=np.float32)
        diff = targets[:, None, :] - tiles_lab[None, :, :]
        weighted_diff = diff * weights[None, None, :]
        return np.sqrt(np.sum(weighted_diff**2, axis=2))

    def _calculate_rgb_distance(self, targets: np.ndarray, tiles_rgb: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances in RGB space."""
        diff = targets[:, None, :] - tiles_rgb[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    def find_best_tiles(self, cell_colors: np.ndarray, match_space: MatchSpace) -> np.ndarray:
        """Return the best tile index for each cell color using the requested match space."""
        self._ensure_tiles_loaded()
        if not self.tiles:
            raise RuntimeError("Tile bank is empty")

        cell_colors_flat = cell_colors.reshape(-1, 3)
        if match_space == MatchSpace.LAB:
            targets = np.array([self._rgb_to_lab(color) for color in cell_colors_flat])
            tile_colors = np.array(self.tile_colors_lab)
            distances = self._calculate_perceptual_distance(targets, tile_colors)
        else:
            targets = cell_colors_flat
            tile_colors = np.array(self.tile_colors)
            distances = self._calculate_rgb_distance(targets, tile_colors)

        noise_factor = 0.01
        distances = distances * (1 + noise_factor * np.random.random(distances.shape))
        best_indices = np.argmin(distances, axis=1)
        return best_indices.reshape(cell_colors.shape[:2])

    def tile_count(self) -> int:
        """Return the number of cached tiles (loading them if necessary)."""
        self._ensure_tiles_loaded()
        return len(self.tiles)

    def tile_stats(self) -> dict:
        """Summarize statistics about the loaded tile bank."""
        self._ensure_tiles_loaded()
        if not self.tiles:
            return {"count": 0}
        return {
            "count": len(self.tiles),
            "tile_size": self.config.tile_size,
            "color_range": {
                "min": np.min(self.tile_colors, axis=0).tolist(),
                "max": np.max(self.tile_colors, axis=0).tolist(),
                "mean": np.mean(self.tile_colors, axis=0).tolist(),
            },
        }

    def _create_fallback_tiles(self) -> None:
        """Populate the tile bank with a deterministic color palette."""
        print("Creating fallback tiles...")
        palette = [
            # Primary colors
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            # Grayscale spectrum
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7],
            [0.8, 0.8, 0.8],
            [0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0],
            # Extended palette
            [1.0, 0.5, 0.0],
            [1.0, 0.3, 0.0],
            [0.5, 0.0, 1.0],
            [0.3, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.8, 0.8],
            [0.0, 0.5, 0.5],
            [0.8, 0.0, 0.5],
            [0.5, 0.2, 0.7],
            [0.7, 0.4, 0.1],
            [0.9, 0.6, 0.3],
        ]
        tiles = []
        colors = []
        colors_lab = []
        tile_size = self.config.tile_size
        for color in palette:
            base = np.ones((tile_size, tile_size, 3), dtype=np.float32)
            tile_np = base * np.array(color, dtype=np.float32)
            if self.config.tile_norm_brightness:
                tile_np = self._normalize_brightness(tile_np)
            tiles.append(tile_np)
            colors.append(np.mean(tile_np, axis=(0, 1)))
            colors_lab.append(self._rgb_to_lab(colors[-1]))
        self.tiles = tiles
        self.tile_colors = colors
        self.tile_colors_lab = colors_lab
        print(f"Created {len(self.tiles)} fallback tiles")
