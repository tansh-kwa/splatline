"""Map user-facing resolution names to concrete per-stage parameters.

Hides the knobs a user shouldn't have to reason about (GAN long edge, training
step count). Lower resolution = fewer steps = faster + less VRAM.
"""

from __future__ import annotations

from typing import Dict

# long_edge:  pixels on the longer image side fed to the GAN / training
# max_steps:  gsplat training iterations
# min_vram_gb: rough floor for the training stage at this resolution
RESOLUTION_PRESETS: Dict[str, Dict[str, int]] = {
    "512p": {"long_edge": 512, "max_steps": 7_000, "min_vram_gb": 6},
    "720p": {"long_edge": 720, "max_steps": 15_000, "min_vram_gb": 10},
    "1080p": {"long_edge": 1080, "max_steps": 30_000, "min_vram_gb": 16},
}


def preset(resolution: str) -> Dict[str, int]:
    try:
        return RESOLUTION_PRESETS[resolution]
    except KeyError:
        raise KeyError(
            f"unknown resolution {resolution!r}; choose from {list(RESOLUTION_PRESETS)}"
        )
