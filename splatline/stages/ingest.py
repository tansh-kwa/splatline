"""Ingest: turn the user's input into a normalized set of frames.

Accepts either a video file (sampled to the frame budget, evenly across the clip)
or a directory of images. Output is always ``frames/frame_0001.png`` ... so that
filenames stay identical through COLMAP and the line-drawing swap.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .._proc import StageError, capture, run, which_or_raise
from ..presets import preset

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

FRAME_GLOB = "frame_*.png"


def _frame_name(i: int) -> str:
    return f"frame_{i:04d}.png"


def _ffmpeg_scale(long_edge: int) -> str:
    """ffmpeg scale filter: fit the long edge to ``long_edge``, downscale-only,
    preserve aspect ratio, keep dimensions even."""
    return (
        f"scale="
        f"w='if(gt(a,1),min({long_edge},iw),-2)':"
        f"h='if(gt(a,1),-2,min({long_edge},ih))'"
    )


def _downscale_keep_aspect(img, long_edge: int):
    """Downscale (never upscale) so the long edge == ``long_edge``."""
    import cv2

    h, w = img.shape[:2]
    cur = max(h, w)
    if cur <= long_edge:
        return img
    scale = long_edge / cur
    new = (max(2, round(w * scale)), max(2, round(h * scale)))
    return cv2.resize(img, new, interpolation=cv2.INTER_AREA)


def _video_frame_count(video: Path) -> int:
    """Best-effort frame count: nb_frames, else duration * avg frame rate."""
    try:
        nb = capture([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(video),
        ])
        if nb.isdigit() and int(nb) > 0:
            return int(nb)
    except StageError:
        pass
    # Fallback: duration * average frame rate
    dur = capture([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(video),
    ])
    rate = capture([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(video),
    ])
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        return max(1, int(float(dur) * fps))
    except (ValueError, ZeroDivisionError):
        raise StageError(f"could not determine frame count for {video}")


def sample_video(
    video: Path, frames_dir: Path, budget: int, long_edge: int, timeout: int
) -> List[Path]:
    which_or_raise("ffmpeg")
    which_or_raise("ffprobe")
    total = _video_frame_count(video)
    step = max(1, total // max(1, budget))
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Single decode pass: keep every `step`-th frame, downscale, cap at the budget.
    run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(video),
            "-vf", f"select='not(mod(n\\,{step}))',{_ffmpeg_scale(long_edge)}",
            "-vsync", "0", "-frames:v", str(budget),
            str(frames_dir / "frame_%04d.png"),
        ],
        timeout=timeout,
        log_path=frames_dir.parent / "logs" / "ingest.log",
        label="ffmpeg (frame sampling)",
    )
    out = sorted(frames_dir.glob(FRAME_GLOB))
    if not out:
        raise StageError(
            f"no frames extracted from {video}; is it a readable video? (see ingest.log)"
        )
    return out


def collect_images(image_dir: Path, frames_dir: Path, long_edge: int) -> List[Path]:
    import cv2

    srcs = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not srcs:
        raise StageError(f"no images found in {image_dir} (looked for {sorted(IMAGE_EXTS)})")
    frames_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for i, src in enumerate(srcs, start=1):
        img = cv2.imread(str(src))
        if img is None:
            raise StageError(f"could not read image {src}")
        img = _downscale_keep_aspect(img, long_edge)
        dst = frames_dir / _frame_name(i)
        cv2.imwrite(str(dst), img)
        out.append(dst)
    return out


def ingest(cfg, run_dir: Path) -> List[Path]:
    """Dispatch on the input type and return normalized, single-resolution frames.

    Frames are downscaled so their long edge == the resolution preset, making the
    whole pipeline single-resolution (so the line-drawing swap is dimensionally exact).
    """
    src = Path(cfg.input)
    frames_dir = run_dir / "frames"
    long_edge = preset(cfg.resolution)["long_edge"]
    if not src.exists():
        raise StageError(f"input not found: {src}")
    if src.is_dir():
        return collect_images(src, frames_dir, long_edge)
    if src.suffix.lower() in VIDEO_EXTS:
        return sample_video(src, frames_dir, cfg.frames, long_edge, cfg.timeouts.ingest)
    raise StageError(
        f"unsupported input {src.name!r}; pass a video ({sorted(VIDEO_EXTS)}) "
        f"or a directory of images"
    )
