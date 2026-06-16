"""Train: fit a 3D Gaussian splat with gsplat on the (swapped) COLMAP dataset.

Shells out to gsplat's ``examples/simple_trainer.py default`` (the reference 3DGS
strategy) from the pinned submodule. Because the COLMAP ``images/`` have already been
swapped for line drawings, the trained scene is stylized from every viewpoint.

The exported ``.ply`` is copied to ``<run_dir>/scene.ply``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .._proc import StageError, run
from ..presets import preset

GSPLAT = Path(__file__).resolve().parents[2] / "third_party" / "gsplat"
TRAINER = GSPLAT / "examples" / "simple_trainer.py"


def train(cfg, colmap_dir: Path, run_dir: Path) -> Path:
    if not TRAINER.exists():
        raise StageError(
            f"gsplat submodule missing at {GSPLAT}. Run `git submodule update --init` "
            "(or `./setup.sh`)."
        )
    max_steps = preset(cfg.resolution)["max_steps"]
    result_dir = run_dir / "splat"

    # Images were pre-downscaled to the target resolution at ingest, so data_factor=1.
    run(
        [
            sys.executable, str(TRAINER), "default",
            "--data_dir", str(colmap_dir),
            "--data_factor", "1",
            "--result_dir", str(result_dir),
            "--max_steps", str(max_steps),
            "--init_type", "sfm",
            "--save_ply",
            "--ply_steps", str(max_steps),
            "--disable_viewer",
        ],
        timeout=cfg.timeouts.train,
        log_path=run_dir / "logs" / "train.log",
        label="gsplat training",
        cwd=GSPLAT / "examples",  # so its local `import utils`/`datasets` resolve
    )

    plys = sorted((result_dir / "ply").glob("*.ply"))
    if not plys:
        raise StageError(
            f"training finished but no .ply was exported (see {run_dir / 'logs' / 'train.log'})"
        )
    scene = run_dir / "scene.ply"
    shutil.copy2(plys[-1], scene)
    return scene
