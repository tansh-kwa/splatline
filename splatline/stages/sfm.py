"""SfM: run COLMAP to recover camera poses + a sparse point cloud.

Produces a gsplat-readable dataset directory:

    <colmap_dir>/images/         the registered images (later swapped for drawings)
    <colmap_dir>/database.db
    <colmap_dir>/sparse/0/{cameras,images,points3D}.bin

Sequential matching is the default (it's the right choice for video / ordered
captures). COLMAP's SIFT can run on GPU; set $SPLATLINE_COLMAP_GPU=0 to force CPU.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from .._proc import StageError, run, which_or_raise


def _gpu_flag() -> str:
    return os.environ.get("SPLATLINE_COLMAP_GPU", "1")


def _populate_images(image_paths: List[Path], images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for p in image_paths:
        dst = images_dir / Path(p).name
        if Path(p).resolve() != dst.resolve():
            shutil.copy2(p, dst)


def run_sfm(
    image_paths: List[Path],
    colmap_dir: Path,
    timeout: int,
    matcher: str = "sequential",
) -> Path:
    """Run COLMAP over ``image_paths`` and return the dataset dir for training."""
    which_or_raise("colmap")
    images_dir = colmap_dir / "images"
    _populate_images(image_paths, images_dir)

    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    logs = colmap_dir.parent / "logs"
    gpu = _gpu_flag()

    run(
        ["colmap", "feature_extractor",
         "--database_path", db, "--image_path", images_dir,
         "--ImageReader.single_camera", "1",
         "--SiftExtraction.use_gpu", gpu],
        timeout=timeout, log_path=logs / "sfm_features.log",
        label="colmap feature_extractor",
    )

    matcher_cmd = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    run(
        ["colmap", matcher_cmd, "--database_path", db, "--SiftMatching.use_gpu", gpu],
        timeout=timeout, log_path=logs / "sfm_match.log",
        label=f"colmap {matcher_cmd}",
    )

    run(
        ["colmap", "mapper",
         "--database_path", db, "--image_path", images_dir,
         "--output_path", sparse],
        timeout=timeout, log_path=logs / "sfm_mapper.log",
        label="colmap mapper",
    )

    if not (sparse / "0" / "cameras.bin").exists():
        raise StageError(
            "COLMAP could not reconstruct a model (no sparse/0). The capture likely "
            "lacks enough overlapping, textured views; add more angles. See "
            f"{logs / 'sfm_mapper.log'}."
        )
    return colmap_dir


def adopt_prepared_colmap(src: Path, colmap_dir: Path) -> List[Path]:
    """Copy a user-provided COLMAP dataset into the run and return its image paths.

    Used by the --from-colmap tier so users can skip the slow, least-deterministic
    SfM stage and just re-stylize + retrain.
    """
    src = Path(src)
    sparse0 = src / "sparse" / "0"
    images = src / "images"
    if not (sparse0 / "cameras.bin").exists():
        raise StageError(f"--from-colmap: {src} has no sparse/0/cameras.bin")
    if not images.is_dir():
        raise StageError(f"--from-colmap: {src} has no images/ directory")
    colmap_dir.mkdir(parents=True, exist_ok=True)
    if colmap_dir.resolve() != src.resolve():
        shutil.copytree(images, colmap_dir / "images", dirs_exist_ok=True)
        shutil.copytree(src / "sparse", colmap_dir / "sparse", dirs_exist_ok=True)
    return sorted((colmap_dir / "images").iterdir())
