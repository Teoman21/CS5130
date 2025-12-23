"""High-level mosaic construction pipeline mirroring Mosaic_Generator/src."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from .config import Config, Implementation, MatchSpace
    from .image_processor import analyze_grid_cells, prepare_image
    from .metrics import calculate_comprehensive_metrics, interpret_metrics
    from .tile_manager import TileManager
    from .utils import np_to_pil
except ImportError:  # pragma: no cover - script execution fallback
    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from config import Config, Implementation, MatchSpace  # type: ignore[no-redef]
    from image_processor import analyze_grid_cells, prepare_image  # type: ignore[no-redef]
    from metrics import calculate_comprehensive_metrics, interpret_metrics  # type: ignore[no-redef]
    from tile_manager import TileManager  # type: ignore[no-redef]
    from utils import np_to_pil  # type: ignore[no-redef]


@dataclass
class MosaicBuilder:
    config: Config

    def __post_init__(self) -> None:
        """Instantiate the tile manager lazily so we cache tiles across builds."""
        self.tile_manager = TileManager(self.config)
        self.processing_time: Dict[str, float] = {}

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize and quantize the source image while recording timing data."""
        start = time.time()
        processed = prepare_image(image, self.config)
        self.processing_time["preprocessing"] = time.time() - start
        return processed

    def analyze_grid_cells(self, image: Image.Image) -> np.ndarray:
        """Compute weighted mean colors for each grid cell of the processed image."""
        start = time.time()
        cells = analyze_grid_cells(image, self.config.grid)
        self.processing_time["grid_analysis"] = time.time() - start
        return cells

    def map_tiles_to_grid(self, cell_colors: np.ndarray) -> np.ndarray:
        """Vectorize the tile selection step and assemble the final mosaic array."""
        start = time.time()
        tile_indices = self.tile_manager.find_best_tiles(cell_colors, self.config.match_space)
        grid = self.config.grid
        tile_size = self.config.tile_size
        tile_bank = np.stack(self.tile_manager.tiles, axis=0).astype(np.float32, copy=False)
        # Gather the selected tiles and reshape into the final mosaic in one go.
        selected_tiles = tile_bank[tile_indices]  # (grid, grid, tile, tile, 3)
        mosaic = (
            selected_tiles.transpose(0, 2, 1, 3, 4)
            .reshape(grid * tile_size, grid * tile_size, 3)
            .copy()
        )
        self.processing_time["tile_mapping"] = time.time() - start
        return mosaic

    def build(self, image: Image.Image) -> Tuple[Image.Image, Image.Image, Dict[str, float]]:
        """Run the entire preprocessing + mapping workflow for a single image."""
        start = time.time()
        self.processing_time = {}
        processed = self.preprocess_image(image)
        cell_colors = self.analyze_grid_cells(processed)
        mosaic_array = self.map_tiles_to_grid(cell_colors)
        mosaic_img = np_to_pil(mosaic_array)
        self.processing_time["total"] = time.time() - start
        stats = {
            "grid_size": self.config.grid,
            "tile_size": self.config.tile_size,
            "output_resolution": f"{mosaic_img.width}x{mosaic_img.height}",
            "processing_time": self.processing_time.copy(),
            "implementation": self.config.impl.value,
            "match_space": self.config.match_space.value,
        }
        return mosaic_img, processed, stats


class MosaicPipeline:
    """Orchestrates preprocessing, building, and metric calculation."""

    def __init__(self, config: Config):
        """Keep a shared builder instance around so we can reuse cached tiles."""
        self.config = config
        self.builder = MosaicBuilder(config)

    def run_full_pipeline(self, image: Image.Image) -> Dict:
        """Execute preprocessing, mosaic generation, and metric calculation."""
        results: Dict[str, dict] = {
            "timing": {},
            "metrics": {},
            "outputs": {},
            "config": self.config.__dict__.copy(),
        }
        mosaic_img, processed_image, stats = self.builder.build(image)
        results["timing"] = stats["processing_time"]
        results["outputs"]["mosaic"] = mosaic_img
        results["outputs"]["processed_image"] = processed_image
        metrics = calculate_comprehensive_metrics(image, mosaic_img)
        results["metrics"] = metrics
        results["metrics_interpretation"] = interpret_metrics(metrics)
        results["grid_info"] = {
            "grid_size": self.config.grid,
            "tile_size": self.config.tile_size,
            "total_tiles": self.config.grid**2,
        }
        return results

    # Backwards-compatible alias used elsewhere in the repo/tests.
    def run(self, image: Image.Image) -> Dict:
        """Alias for run_full_pipeline kept for compatibility with the original repo."""
        return self.run_full_pipeline(image)

    def benchmark_grid_sizes(self, image: Image.Image, grid_sizes: List[int]) -> Dict[int, dict]:
        """Time the pipeline for several grid sizes while keeping other settings fixed."""
        original_grid = self.config.grid
        original_w, original_h = self.config.out_w, self.config.out_h
        benchmarks: Dict[int, dict] = {}
        for grid_size in grid_sizes:
            self.config.grid = grid_size
            aspect_ratio = image.width / image.height
            self.config.out_w = (image.width // grid_size) * grid_size
            self.config.out_h = int(self.config.out_w / aspect_ratio // grid_size) * grid_size
            start = time.time()
            run_results = self.run_full_pipeline(image)
            total_time = time.time() - start
            benchmarks[grid_size] = {
                "processing_time": total_time,
                "output_resolution": f"{run_results['outputs']['mosaic'].width}x{run_results['outputs']['mosaic'].height}",
                "total_tiles": grid_size * grid_size,
                "tiles_per_second": (grid_size * grid_size) / total_time if total_time else 0,
                "metrics": run_results["metrics"],
            }
        self.config.grid = original_grid
        self.config.out_w, self.config.out_h = original_w, original_h
        return benchmarks

    def benchmark_implementations(self, image: Image.Image) -> Dict[str, dict]:
        """Mirrors the comparison helper from the original pipeline."""
        original_impl = self.config.impl
        results: Dict[str, dict] = {"vectorized": {}, "loop_based": {}, "comparison": {}}

        self.config.impl = Implementation.VECT
        start = time.time()
        vect_results = self.run_full_pipeline(image)
        vec_time = time.time() - start
        results["vectorized"] = {
            "processing_time": vec_time,
            "metrics": vect_results["metrics"],
            "mosaic": vect_results["outputs"]["mosaic"],
        }

        self.config.impl = Implementation.LOOPS
        start = time.time()
        loop_results = self.run_full_pipeline(image)
        loop_time = time.time() - start
        results["loop_based"] = {
            "processing_time": loop_time,
            "metrics": loop_results["metrics"],
            "mosaic": loop_results["outputs"]["mosaic"],
        }

        results["comparison"] = {
            "speedup_factor": (loop_time / vec_time) if vec_time else 0.0,
            "time_difference": loop_time - vec_time,
            "vectorized_faster": vec_time < loop_time,
        }

        self.config.impl = original_impl
        return results

    def analyze_performance_scaling(self, benchmark_results: Dict[int, dict]) -> Dict[str, List[float] | Dict[str, float]]:
        """Generate the same scaling summary helper offered by the original pipeline."""
        grid_sizes = sorted(benchmark_results.keys())
        processing_times = [benchmark_results[g]["processing_time"] for g in grid_sizes]
        total_tiles = [benchmark_results[g]["total_tiles"] for g in grid_sizes]
        tiles_per_second = [benchmark_results[g]["tiles_per_second"] for g in grid_sizes]

        scaling = {
            "grid_sizes": grid_sizes,
            "processing_times": processing_times,
            "total_tiles": total_tiles,
            "tiles_per_second": tiles_per_second,
            "scaling_factors": {},
        }

        if len(grid_sizes) >= 2:
            tile_ratio = total_tiles[-1] / total_tiles[0]
            time_ratio = processing_times[-1] / processing_times[0] if processing_times[0] else 0.0
            scaling["scaling_factors"] = {
                "tile_increase_ratio": tile_ratio,
                "time_increase_ratio": time_ratio,
                "scaling_efficiency": tile_ratio / time_ratio if time_ratio else 0.0,
                "is_linear_scaling": (abs(time_ratio - tile_ratio) / tile_ratio) < 0.1 if tile_ratio else False,
            }

        return scaling
