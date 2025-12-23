"""Configuration dataclasses and enums for the mosaic generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Implementation(Enum):
    VECT = "Vectorised"
    LOOPS = "Loop-based"


class MatchSpace(Enum):
    LAB = "Lab (perceptual)"
    RGB = "RGB (euclidean)"


@dataclass
class Config:
    grid: int = 32
    out_w: int = 768
    out_h: int = 768
    tile_size: int = 32

    hf_dataset: str = "Kratos-AI/KAI_car-images"
    hf_split: str = "train"
    hf_limit: int = 200
    hf_cache_dir: Optional[str] = None

    impl: Implementation = Implementation.VECT
    match_space: MatchSpace = MatchSpace.LAB

    use_uniform_q: bool = False
    q_levels: int = 8
    use_kmeans_q: bool = False
    k_colors: int = 8

    tile_norm_brightness: bool = False
    allow_rotations: bool = False

    tiles_cache_dir: Optional[str] = str((Path(__file__).resolve().parent / "tile_cache"))

    do_bench: bool = False
    bench_grids: Optional[List[int]] = None
