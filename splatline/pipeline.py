"""Pipeline orchestration: drive the stages in the right order for the swap point.

    pre-train (default): COLMAP on the ORIGINAL frames -> swap in line drawings -> train
    pre-sfm:             stylize first -> COLMAP on the drawings -> train
    --from-colmap:       reuse a prepared COLMAP dataset, just re-stylize + retrain

Each stage has a hard timeout (see config.Timeouts) and writes a log under
``<run_dir>/logs``. Preflight runs before any GPU work.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List

from .config import SceneConfig


def _log(msg: str) -> None:
    print(f"[splatline] {msg}", flush=True)


def _swap_images(drawings: List[Path], images_dir: Path) -> int:
    """Overwrite COLMAP's images with the same-named line drawings. Returns count."""
    by_name = {p.name: p for p in drawings}
    swapped = 0
    for target in images_dir.iterdir():
        src = by_name.get(target.name)
        if src is not None:
            shutil.copy2(src, target)
            swapped += 1
    if swapped == 0:
        from ._proc import StageError

        raise StageError(
            "image swap matched no files: stylized frames and COLMAP images have "
            "different names. This is a bug if you didn't use --from-colmap."
        )
    return swapped


def run_pipeline(cfg: SceneConfig, run_dir: Path) -> int:
    from .preflight import preflight
    from .stages.compress import compress
    from .stages.ingest import ingest
    from .stages.sfm import adopt_prepared_colmap, run_sfm
    from .stages.stylize import stylize
    from .stages.train import train

    start = time.monotonic()
    run_dir = Path(run_dir)
    colmap_dir = run_dir / "colmap"

    if cfg.from_colmap:
        _log(f"reusing prepared COLMAP from {cfg.from_colmap} (skipping ingest/preflight/SfM)")
        frames = adopt_prepared_colmap(Path(cfg.from_colmap), colmap_dir)
        _log(f"stylizing {len(frames)} frames ({cfg.summary()})")
        drawings = stylize(cfg, run_dir, frames)
        _swap_images(drawings, colmap_dir / "images")
    else:
        _log("ingesting input")
        frames = ingest(cfg, run_dir)
        _log(f"ingested {len(frames)} frames")

        _log("preflight checks")
        report = preflight(frames, strict=True)  # raises PreflightError on fatal issues
        for w in report.warnings:
            _log(f"warning: {w}")

        if cfg.swap_point == "pre-sfm":
            _log(f"stylizing first ({cfg.summary()})")
            drawings = stylize(cfg, run_dir, frames)
            _log("running SfM on the stylized frames (COLMAP)")
            run_sfm(drawings, colmap_dir, cfg.timeouts.sfm)
        else:  # pre-train (default)
            _log("running SfM on the original frames (COLMAP)")
            run_sfm(frames, colmap_dir, cfg.timeouts.sfm)
            _log(f"stylizing ({cfg.summary()})")
            drawings = stylize(cfg, run_dir, frames)
            n = _swap_images(drawings, colmap_dir / "images")
            _log(f"swapped {n} images for line drawings (pre-train swap)")

    _log("training the Gaussian splat (gsplat)")
    scene = train(cfg, colmap_dir, run_dir)

    artifacts = [scene]
    if cfg.ksplat:
        _log("compressing to .ksplat")
        artifacts.append(compress(scene, run_dir, cfg))

    mins = (time.monotonic() - start) / 60.0
    _log(f"done in {mins:.1f} min. Scene:")
    for a in artifacts:
        print(f"  {a}")
    return 0
