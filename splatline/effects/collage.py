"""Subject segmentation for per-subject / collage looks.

Uses lang-sam (text-prompted Segment Anything) so a subject is selected by name, e.g.
"tractor". lang-sam downloads its own SAM2 + GroundingDINO weights on first use, so
there is nothing for splatline to fetch here. The stylize stage does the compositing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_SAM_TYPE = "sam2.1_hiera_small"


def load_segmenter(device: str = "cuda", sam_type: str = DEFAULT_SAM_TYPE):
    """Load lang-sam once and reuse it across frames."""
    from lang_sam import LangSAM

    return LangSAM(sam_type=sam_type, device=device)


def subject_mask(
    model,
    color_bgr: np.ndarray,
    subject: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Optional[np.ndarray]:
    """Return a boolean mask of every region matching ``subject`` (or None)."""
    import cv2
    from PIL import Image

    image_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    results = model.predict([image_pil], [subject], box_threshold, text_threshold)
    masks = results[0].get("masks") if results else None
    if masks is None or len(masks) == 0:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    # Union all matched instances (e.g. several "tractor"s) into one subject mask.
    return np.any(masks > 0.5, axis=0).astype(bool)
