"""Up-front environment checks so the tool fails clearly *before* a long run.

Covers the things that actually break self-install: missing NVIDIA driver,
unusable CUDA, not enough free VRAM, and missing host binaries (ffmpeg, colmap).
Used by ``splatline check`` and as a preflight inside ``splatline run``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fatal: bool = True  # a failed non-fatal check is a warning, not a hard stop


def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_nvidia_driver() -> CheckResult:
    out = _run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
    if out is None:
        return CheckResult(
            "nvidia-driver",
            False,
            "nvidia-smi not found or failed. Install the NVIDIA driver and make sure "
            "the GPU is visible (`nvidia-smi`).",
        )
    first = out.splitlines()[0]
    return CheckResult("nvidia-driver", True, first)


def check_free_vram(min_gb: int = 0) -> CheckResult:
    out = _run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    if out is None:
        return CheckResult("vram", False, "could not query free VRAM (no GPU?)")
    try:
        free_mb = max(int(x) for x in out.splitlines())
    except ValueError:
        return CheckResult("vram", False, f"unparseable nvidia-smi output: {out!r}")
    free_gb = free_mb / 1024
    ok = free_gb + 1e-6 >= min_gb
    need = f" (need >= {min_gb} GB for this resolution)" if min_gb else ""
    return CheckResult("vram", ok, f"{free_gb:.1f} GB free{need}")


def check_cuda() -> CheckResult:
    """Report the CUDA runtime torch was built against (the one that matters)."""
    try:
        import torch  # noqa: WPS433 (optional heavy import)
    except Exception as exc:  # pragma: no cover - torch missing
        return CheckResult("cuda", False, f"torch not importable: {exc}")
    if not torch.cuda.is_available():
        return CheckResult(
            "cuda", False, f"torch {torch.__version__} present but CUDA not available"
        )
    return CheckResult(
        "cuda",
        True,
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"device: {torch.cuda.get_device_name(0)}",
    )


def check_binary(name: str, fatal: bool = True) -> CheckResult:
    path = shutil.which(name)
    if path:
        return CheckResult(name, True, path, fatal=fatal)
    return CheckResult(name, False, f"{name} not on PATH", fatal=fatal)


def check_gsplat(fatal: bool = True) -> CheckResult:
    """The splat trainer library (native installs must `pip install gsplat`)."""
    try:
        import gsplat  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - gsplat missing on dev box
        return CheckResult("gsplat", False, f"not importable: {exc}", fatal=fatal)
    return CheckResult("gsplat", True, f"version {getattr(gsplat, '__version__', '?')}")


def check_trainer() -> CheckResult:
    """The vendored example trainer script (needs the gsplat submodule checked out)."""
    from .stages.train import TRAINER

    if TRAINER.exists():
        return CheckResult("gsplat-trainer", True, str(TRAINER))
    return CheckResult(
        "gsplat-trainer", False,
        f"missing {TRAINER}; run `git submodule update --init`", fatal=False,
    )


def check_langsam() -> CheckResult:
    """lang-sam (text-prompted SAM); only needed for --subject collage."""
    try:
        import lang_sam  # noqa: F401, WPS433
    except Exception as exc:  # pragma: no cover
        return CheckResult(
            "lang-sam", False,
            f"not importable ({exc}); only needed for --subject collage", fatal=False,
        )
    return CheckResult("lang-sam", True, "importable")


def check_weights() -> List[CheckResult]:
    from ._weights import find_gan_weights

    found = [s for s in ("contour", "anime") if find_gan_weights(s)]
    if found:
        return [CheckResult("gan-weights", True, "styles: " + ", ".join(found))]
    return [CheckResult(
        "gan-weights", False,
        "not found; run `scripts/fetch_weights.sh ./weights`", fatal=False,
    )]


def run_checks(min_vram_gb: int = 0, require_gpu: bool = True) -> List[CheckResult]:
    """Return all checks. GPU-related ones are skipped when require_gpu is False."""
    results: List[CheckResult] = []
    if require_gpu:
        results.append(check_nvidia_driver())
        results.append(check_cuda())
        results.append(check_free_vram(min_vram_gb))
        results.append(check_gsplat())
    results.append(check_binary("ffmpeg"))
    results.append(check_binary("colmap", fatal=require_gpu))
    results.append(check_trainer())
    results.append(check_langsam())
    results.extend(check_weights())
    return results


def format_report(results: List[CheckResult]) -> str:
    lines = []
    for r in results:
        mark = "OK  " if r.ok else ("FAIL" if r.fatal else "WARN")
        lines.append(f"[{mark}] {r.name:14s} {r.detail}")
    return "\n".join(lines)


def overall_ok(results: List[CheckResult]) -> bool:
    return all(r.ok or not r.fatal for r in results)
