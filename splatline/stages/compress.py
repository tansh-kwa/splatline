"""Compress (optional): convert scene.ply -> scene.ksplat for portable sharing.

This is opt-in (``--ksplat``). The ``.ply`` is the primary, universal artifact; the
``.ksplat`` format is smaller and convenient but is produced by Mark Kellogg's
GaussianSplats3D Node tool, so this step needs Node on PATH. If it's missing we fail
with a clear message rather than silently skipping.

Override the converter command with $SPLATLINE_KSPLAT_CMD (use {ply} and {ksplat}
placeholders), e.g. a locally-installed create-ksplat script.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from .._proc import StageError, run

# Default: the npx-published converter from @mkkellogg/gaussian-splats3d.
DEFAULT_CMD = "npx --yes @mkkellogg/gaussian-splats3d create-ksplat {ply} {ksplat}"


def compress(scene_ply: Path, run_dir: Path, cfg) -> Path:
    ksplat = run_dir / "scene.ksplat"
    template = os.environ.get("SPLATLINE_KSPLAT_CMD", DEFAULT_CMD)
    cmd = shlex.split(template.format(ply=str(scene_ply), ksplat=str(ksplat)))

    if not shutil.which(cmd[0]):
        raise StageError(
            f"--ksplat needs {cmd[0]!r} on PATH (Node) to run the converter. "
            "Install Node, set $SPLATLINE_KSPLAT_CMD to a local converter, or drop "
            "--ksplat (scene.ply is the primary output)."
        )

    run(
        cmd,
        timeout=cfg.timeouts.compress,
        log_path=run_dir / "logs" / "compress.log",
        label="ksplat conversion",
    )
    if not ksplat.exists():
        raise StageError(
            f"ksplat conversion ran but {ksplat} was not produced (see compress.log)"
        )
    return ksplat
