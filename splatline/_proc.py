"""Subprocess helper with hard timeouts and per-stage log files.

Every external tool (ffmpeg, COLMAP, gsplat) is run through here so that a
pathological input can't hang forever and so the user gets the log path on failure
instead of a wall of stderr.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence


class StageError(RuntimeError):
    """A pipeline stage failed; message is human-readable, details are in the log."""


class StageTimeout(StageError):
    pass


def run(
    cmd: Sequence[str],
    *,
    timeout: int,
    log_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    label: Optional[str] = None,
) -> None:
    """Run ``cmd``, streaming stdout+stderr to ``log_path``.

    Raises StageTimeout on timeout, StageError on non-zero exit, both with a
    message pointing at the log.
    """
    label = label or cmd[0]
    cmd = [str(c) for c in cmd]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w") if log_path else None
    try:
        if log_fh:
            log_fh.write(f"$ {' '.join(cmd)}\n\n")
            log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise StageTimeout(
                f"{label} exceeded its {timeout}s timeout and was killed"
                + (f" (see {log_path})" if log_path else "")
            )
    finally:
        if log_fh:
            log_fh.close()
    if rc != 0:
        raise StageError(
            f"{label} failed (exit {rc})"
            + (f" (see {log_path})" if log_path else "")
        )


def which_or_raise(name: str) -> str:
    import shutil

    path = shutil.which(name)
    if not path:
        raise StageError(
            f"required tool {name!r} not found on PATH. "
            "Install it (see README) or run `./setup.sh`."
        )
    return path


def capture(cmd: Sequence[str], *, timeout: int = 60) -> str:
    """Run a quick command and return stdout (for probes like ffprobe)."""
    cmd = [str(c) for c in cmd]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if out.returncode != 0:
        raise StageError(f"{cmd[0]} failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.strip()
