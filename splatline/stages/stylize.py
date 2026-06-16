"""Stylize: turn frames into line drawings (and optionally watercolor / collage).

Drives Chan et al.'s informative-drawings GAN from the pinned submodule (importing
``model.Generator`` directly, with no copied code or argparse). Then optionally applies
the watercolor blend and/or lang-sam collage (see ``splatline.effects``) on top.

Output filenames match the input frames exactly so the line-drawing swap into the
COLMAP image set is dimensionally and name-wise exact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from .._proc import StageError

# repo_root/third_party/informative-drawings
SUBMODULE = Path(__file__).resolve().parents[2] / "third_party" / "informative-drawings"


def _device(preferred: Optional[str] = None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_gan_weights(style: str) -> Path:
    """Weights layout: <root>/<style>_style/netG_A_latest.pth (informative-drawings)."""
    from .._weights import find_gan_weights, gan_weights_roots

    ckpt = find_gan_weights(style)
    if ckpt is None:
        searched = ", ".join(str(r) for r in gan_weights_roots())
        raise StageError(
            f"GAN weights for style {style!r} not found (looked in: {searched}). "
            "Run `scripts/fetch_weights.sh ./weights` (or `./setup.sh`), "
            "or set $SPLATLINE_GAN_WEIGHTS."
        )
    return ckpt


def _load_generator(style: str, device: str):
    if str(SUBMODULE) not in sys.path:
        sys.path.insert(0, str(SUBMODULE))
    if not (SUBMODULE / "model.py").exists():
        raise StageError(
            f"informative-drawings submodule missing at {SUBMODULE}. "
            "Run `git submodule update --init` (or `./setup.sh`)."
        )
    import torch
    from model import Generator  # from the submodule

    net = Generator(3, 1, 3)  # input_nc=3, output_nc=1, n_blocks=3 (as in test.py)
    net.load_state_dict(torch.load(_resolve_gan_weights(style), map_location=device))
    return net.to(device).eval()


def line_drawings(
    frames: List[Path], out_dir: Path, style: str, device: Optional[str] = None
) -> List[Path]:
    """Run the GAN over each frame, writing a same-size grayscale line drawing."""
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image

    device = _device(device)
    net = _load_generator(style, device)
    to_tensor = T.ToTensor()
    out_dir.mkdir(parents=True, exist_ok=True)

    out: List[Path] = []
    with torch.no_grad():
        for f in frames:
            img = Image.open(f).convert("RGB")
            w, h = img.size
            t = to_tensor(img).unsqueeze(0).to(device)
            drawing = net(t)[0, 0].clamp(0, 1).cpu().numpy()
            arr = (drawing * 255).astype(np.uint8)
            # Guarantee exact size match with the source frame (intrinsics alignment).
            if arr.shape[1] != w or arr.shape[0] != h:
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
            # Write 3-channel: COLMAP/gsplat expect RGB images.
            dst = out_dir / f.name
            cv2.imwrite(str(dst), cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))
            out.append(dst)
    return out


def _render_look(look: str, frames: List[Path], run_dir: Path, device: str, cache: dict):
    """Render (and cache) the per-frame images for a drawable look (LINE_LOOKS)."""
    if look in cache:
        return cache[look]
    import cv2

    if look == "watercolor":
        from ..effects.blend import watercolor_blend

        contour = _render_look("contour", frames, run_dir, device, cache)
        out_dir = run_dir / "looks_watercolor"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f, cpath in zip(frames, contour):
            color = cv2.imread(str(f))
            line = cv2.imread(str(cpath), cv2.IMREAD_GRAYSCALE)
            p = out_dir / f.name
            cv2.imwrite(str(p), watercolor_blend(color, line))
            paths.append(p)
        cache[look] = paths
        return paths

    # contour / anime: straight GAN line drawing
    cache[look] = line_drawings(frames, run_dir / f"looks_{look}", look, device)
    return cache[look]


def _layer(look: str, color, cache: dict, i: int):
    """The rendered image for ``look`` on frame ``i`` (always a fresh, writable array)."""
    import numpy as np

    if look == "white":
        return np.full_like(color, 255)
    if look == "photo":
        return color.copy()
    import cv2

    return cv2.imread(str(cache[look][i]))


def stylize(cfg, run_dir: Path, frames: List[Path]) -> List[Path]:
    """Render the base look + per-subject overrides into the swapped image set.

    One path for everything: the base look fills each frame; then, if there are
    subjects, each subject's look is pasted inside its lang-sam mask (later subjects
    over earlier ones). ``photo``/``white`` looks need no GAN.
    """
    import cv2

    from ..config import LINE_LOOKS

    device = _device()
    cache: dict = {}
    needed = {cfg.style} | {s.style for s in (cfg.subjects or [])}
    for look in needed:
        if look in LINE_LOOKS:
            _render_look(look, frames, run_dir, device, cache)

    out_dir = run_dir / "drawings"
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter = None
    if cfg.subjects:
        from ..effects.collage import load_segmenter, subject_mask

        segmenter = load_segmenter(device)

    out: List[Path] = []
    for i, f in enumerate(frames):
        color = cv2.imread(str(f))
        canvas = _layer(cfg.style, color, cache, i)
        if cfg.subjects:
            for subj in cfg.subjects:
                mask = subject_mask(segmenter, color, subj.prompt)
                if mask is None:
                    continue
                canvas[mask] = _layer(subj.style, color, cache, i)[mask]
        dst = out_dir / f.name
        cv2.imwrite(str(dst), canvas)
        out.append(dst)
    return out
