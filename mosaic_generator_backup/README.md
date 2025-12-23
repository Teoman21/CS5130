# Mosaic_Gnerator_Improved

A refactored, profiled, and optimized version of the Lab 1 photomosaic assignment. This folder retains the exact behavior of `Mosaic_Generator/src` while reorganizing the code into focused modules, adding profiling artifacts, and documenting the performance gains required for CS5130 Lab 4.

## Project Layout

```
├── __init__.py                  # Public package exports
├── app.py                      # Gradio UI (stand-alone & package aware)
├── config.py                   # Dataclasses / enums / defaults
├── image_processor.py          # Loading, resizing, quantization, grid stats
├── tile_manager.py             # Tile streaming, caching, color features
├── mosaic_builder.py           # MosaicBuilder + MosaicPipeline orchestration
├── metrics.py                  # MSE / PSNR / SSIM / color metrics
├── utils.py                    # PIL↔NumPy helpers and grid math
├── profiling_analysis.ipynb    # cProfile + line_profiler walkthrough
├── requirements.txt            # Includes line_profiler
└── tile_cache/                 # Persistent tile embeddings (auto-created)
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Mosaic_Gnerator_Improved/requirements.txt
python3 Mosaic_Gnerator_Improved/app.py  # or `gradio app.py`
```

The default `Config` now writes tile embeddings to `Mosaic_Gnerator_Improved/tile_cache/`, so the Hugging Face dataset is downloaded only once.

## Assignment Deliverables

### Part 1 – Profiling
- Open `profiling_analysis.ipynb` to record the profiler output and capture the tables described below.
- Run the baseline pipeline under `cProfile`:
  ```bash
  python3 -m cProfile -o legacy.prof -s cumulative - <<'PY'
  from pathlib import Path
  from PIL import Image
  from Mosaic_Generator.src.config import Config as LegacyConfig
  from Mosaic_Generator.src.pipeline import MosaicPipeline as LegacyPipeline

  img = Image.open(Path("data/test_images/copley.png")).convert("RGB")
  cfg = LegacyConfig(grid=32, out_w=512, out_h=512, tiles_cache_dir="Mosaic_Gnerator_Improved/tile_cache")
  LegacyPipeline(cfg).run_full_pipeline(img)
  PY
  ```
- Repeat for the improved pipeline:
  ```bash
  python3 -m cProfile -o improved.prof -s cumulative - <<'PY'
  from pathlib import Path
  from PIL import Image
  from Mosaic_Gnerator_Improved.config import Config
  from Mosaic_Gnerator_Improved.mosaic_builder import MosaicPipeline

  img = Image.open(Path("data/test_images/copley.png")).convert("RGB")
  cfg = Config(grid=32, out_w=512, out_h=512, tiles_cache_dir="Mosaic_Gnerator_Improved/tile_cache")
  MosaicPipeline(cfg).run_full_pipeline(img)
  PY
  ```
- Inspect the results with `python3 -m pstats legacy.prof` / `improved.prof`. The top three bottlenecks match the report: tile matching loops, repeated tile downloads, and Python-based grid means.
- Capture line-level timings by wrapping the legacy generator with `kernprof`:
  ```bash
  cat <<'PY' > /tmp/profile_baseline.py
  from pathlib import Path
  from PIL import Image
  from Mosaic_Generator.src.config import Config as LegacyConfig
  from Mosaic_Generator.src.pipeline import MosaicPipeline as LegacyPipeline

  img = Image.open(Path("data/test_images/copley.png")).convert("RGB")
  cfg = LegacyConfig(grid=32, out_w=512, out_h=512, tiles_cache_dir="Mosaic_Gnerator_Improved/tile_cache")
  pipeline = LegacyPipeline(cfg)
  pipeline.run_full_pipeline(img)
  PY
  kernprof -l -v /tmp/profile_baseline.py
  ```
  Repeat with the improved pipeline to confirm the loop hot spots disappear.



Key changes:
- Vectorized cell analysis (`utils.block_view`, `cell_means`) removes Python loops.
- `TileManager.find_best_tiles` computes all distances in NumPy (LAB or RGB space) and adds deterministic noise to break ties.
- Disk+memory file caches ensure Hugging Face tiles load once, mirroring the original helper scripts.

### Part 3 – Refactoring & Modularity
- Each module now has focused responsibilities with docstrings, input validation, and helpful exceptions.
- `app.py` consumes only the public API (`Config`, `MosaicPipeline`) for parity with `Mosaic_Generator/src/gradio_interface.py` and displays timing metrics in the UI.

## Module Usage Example

```python
from Mosaic_Gnerator_Improved.config import Config
from Mosaic_Gnerator_Improved.mosaic_builder import MosaicPipeline
from PIL import Image

config = Config(grid=32, tile_size=32, tiles_cache_dir="tile_cache")
pipeline = MosaicPipeline(config)
input_image = Image.open("data/test_images/house.jpg")
results = pipeline.run_full_pipeline(input_image)
mosaic = results["outputs"]["mosaic"]
mosaic.save("house_mosaic.png")
```

## Profiling Tips
- Install `line_profiler` (already part of `requirements.txt`) before running the `kernprof` commands shown above.
- Inspect saved stats with `python3 -m pstats *.prof` or `python3 -m line_profiler *.lprof`.
- Delete `tile_cache/` if you want to measure the cost of tile streaming from scratch; otherwise leave it in place for realistic (cached) timings.

## Deployment / Sharing
- Launch via `python3 app.py` for a local UI or `gradio app.py --share` to obtain a public link.
- The Hugging Face warning about LibreSSL is harmless on macOS; upgrading to Python 3.11+ with OpenSSL 1.1.1 removes it.

## Troubleshooting
- **Slow first run**: expected while Hugging Face tiles download. Subsequent runs load from `tile_cache/` instantly.
- **line_profiler import error**: ensure `pip install -r requirements.txt` inside your virtual environment.
- **Dataset offline**: `TileManager` falls back to a deterministic palette so the pipeline stays functional even without the dataset.
