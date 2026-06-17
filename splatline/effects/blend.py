"""Watercolor blend: soft color washes from the photo with ink lines on top.

The ink lines are composited *multiplicatively* over a low-frequency color wash, so
paper areas keep the photo's full color while the lines darken to ink. (An additive
blend, as in the original hybrid.py, mixes in the line drawing's white paper and washes
the color out.) ``color_strength`` lightly boosts saturation for a more painterly look.
"""

from __future__ import annotations

import cv2
import numpy as np


def _low_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass (keeps the soft color, drops fine detail)."""
    # ksize=(0,0) lets OpenCV derive the kernel from sigma; handles multi-channel.
    return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def watercolor_blend(
    color_bgr: np.ndarray,
    line_img: np.ndarray,
    sigma_low: float = 8.0,
    color_strength: float = 1.3,
) -> np.ndarray:
    """Ink lines multiplied over a soft, saturation-boosted color wash.

    Args:
        color_bgr:      original frame (H, W, 3), uint8.
        line_img:       line drawing, (H, W) or (H, W, 3), uint8 (white paper, dark ink).
        sigma_low:      blur strength for the color wash (smaller keeps more color detail).
        color_strength: saturation multiplier for the wash (1.0 = unchanged).

    Returns:
        uint8 BGR image the same size as ``color_bgr``.
    """
    color = color_bgr.astype(np.float32) / 255.0
    lines = line_img.astype(np.float32) / 255.0
    if lines.ndim == 2:
        lines = lines[:, :, None]

    wash = _low_pass(color, sigma_low)
    if color_strength != 1.0:
        hsv = cv2.cvtColor(np.clip(wash, 0, 1), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * color_strength, 0, 1)
        wash = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out = np.clip(wash * lines, 0.0, 1.0)  # multiply: paper keeps color, ink darkens
    return (out * 255).astype(np.uint8)
