"""Watercolor blend: low-frequency color from the photo + the line drawing on top.

Copied and tidied from the user's ``hybrid.py`` (their own code, not Chan's GAN).
Produces the "watercolor" look from the blog: soft color washes under ink lines.
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
    sigma_low: float = 15.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend ``alpha * lowpass(color) + (1 - alpha) * line_drawing``.

    Args:
        color_bgr: original frame (H, W, 3), uint8.
        line_img:  line drawing, (H, W) or (H, W, 3), uint8.
        sigma_low: blur strength for the color wash.
        alpha:     weight of the color wash vs. the lines.

    Returns:
        uint8 BGR image the same size as ``color_bgr``.
    """
    color = color_bgr.astype(np.float64) / 255.0
    lines = line_img.astype(np.float64) / 255.0
    if lines.ndim == 2:
        lines = np.repeat(lines[:, :, None], 3, axis=2)

    low = _low_pass(color, sigma_low)
    hybrid = alpha * low + (1.0 - alpha) * lines
    hybrid = np.clip(hybrid, 0.0, 1.0)
    return (hybrid * 255).astype(np.uint8)
