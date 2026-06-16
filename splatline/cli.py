"""Thin CLI: maps user-facing flags to a generated config, then runs the pipeline.

    splatline run ./my-video.mp4 --style contour             # line drawing
    splatline run ./photos/ --style watercolor               # watercolor
    splatline run ./my-video.mp4 --subject "tractor"         # collage (subject over white)
    splatline run runs/my-scene/config.yaml                  # re-run / per-subject config
    splatline run ./video.mp4 --from-colmap ./colmap_out     # skip SfM
    splatline check                                          # environment report

Per-subject styling (different looks per subject, photo/white bases) is config-only;
see configs/per-subject.example.yaml.

Internals the user shouldn't need (swap-point, COLMAP details) are hidden;
swap-point defaults to pre-train.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__
from .config import SceneConfig, SubjectStyle
from .presets import preset


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "scene"


def _config_from_args(args: argparse.Namespace) -> SceneConfig:
    """Build a SceneConfig from either a saved .yaml or media path + flags."""
    src = Path(args.input)
    if src.suffix.lower() in {".yaml", ".yml"} and src.is_file():
        return SceneConfig.load(src)

    scene = args.scene or _slugify(src.stem if src.suffix else src.name)
    common = dict(
        scene=scene,
        input=str(src),
        resolution=args.resolution,
        frames=args.frames,
        from_colmap=args.from_colmap,
        ksplat=args.ksplat,
    )
    if args.subject:
        # collage: only the named subject, in --style, over a white base.
        cfg = SceneConfig(
            style="white",
            subjects=[SubjectStyle(prompt=args.subject, style=args.style)],
            **common,
        )
    else:
        cfg = SceneConfig(style=args.style, **common)
    if args.swap_point:
        cfg.swap_point = args.swap_point
    return cfg.validate()


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    run_dir = Path(args.runs_dir) / cfg.scene
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    print(f"scene '{cfg.scene}': {cfg.summary()} @ {cfg.resolution} -> {run_dir}/scene.ply")

    if args.dry_run:
        print("\n--dry-run: config generated; stopping before any compute.\n")
        print(cfg.to_yaml())
        return 0

    # Imported lazily so `check` / `--dry-run` work without heavy deps installed.
    from .pipeline import run_pipeline

    return run_pipeline(cfg, run_dir)


def cmd_check(args: argparse.Namespace) -> int:
    from .env_check import format_report, overall_ok, run_checks

    min_vram = preset(args.resolution)["min_vram_gb"] if args.resolution else 0
    results = run_checks(min_vram_gb=min_vram, require_gpu=not args.no_gpu)
    print(format_report(results))
    ok = overall_ok(results)
    print("\n" + ("Environment looks good." if ok else "Environment has problems (see above)."))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="splatline", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"splatline {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="generate a scene from a video, image dir, or config")
    r.add_argument("input", help="video file, image directory, or a saved config.yaml")
    r.add_argument("--style", choices=("contour", "anime", "watercolor"), default="contour",
                   help="the look to apply (default: contour)")
    r.add_argument("--resolution", choices=("512p", "720p", "1080p"), default="512p")
    r.add_argument("--frames", type=int, default=150, help="frame budget from video")
    r.add_argument("--subject", default=None,
                   help="collage: draw only this subject (in --style) over white (lang-sam)")
    r.add_argument("--ksplat", action="store_true", help="also emit a portable .ksplat")
    r.add_argument("--from-colmap", default=None, help="reuse prepared COLMAP output")
    r.add_argument("--scene", default=None, help="scene name (default: from input name)")
    r.add_argument("--runs-dir", default="runs", help="where run outputs are written")
    r.add_argument("--dry-run", action="store_true", help="generate config, run no compute")
    # Hidden power-user knob; default pre-train per the blog.
    r.add_argument("--swap-point", choices=("pre-train", "pre-sfm"), default=None,
                   help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", help="check the environment (GPU, CUDA, VRAM, binaries)")
    c.add_argument("--resolution", choices=("512p", "720p", "1080p"), default="512p",
                   help="check VRAM against this resolution's floor")
    c.add_argument("--no-gpu", action="store_true", help="skip GPU/CUDA checks")
    c.set_defaults(func=cmd_check)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surface a clean message, not a traceback, to users
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
