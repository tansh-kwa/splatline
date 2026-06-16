"""Locate the line-drawing GAN weights.

Search order (first existing wins):
  1. $SPLATLINE_GAN_WEIGHTS
  2. a repo-local ./weights/ dir (what scripts/fetch_weights.sh writes)

(SAM weights are handled by lang-sam, which downloads its own checkpoints.)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL = REPO_ROOT / "weights"


def gan_weights_roots() -> List[Path]:
    """Candidate roots that each contain <style>_style/netG_A_latest.pth."""
    roots: List[Path] = []
    env = os.environ.get("SPLATLINE_GAN_WEIGHTS")
    if env:
        roots.append(Path(env))
    roots.append(_LOCAL / "informative-drawings")
    return roots


def find_gan_weights(style: str) -> Optional[Path]:
    for root in gan_weights_roots():
        ckpt = root / f"{style}_style" / "netG_A_latest.pth"
        if ckpt.exists():
            return ckpt
    return None
