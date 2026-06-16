"""Preflight: reject input that COLMAP/training will choke on, *before* GPU time.

The expensive failure mode is paying to train on a scene SfM was never going to
solve. These are cheap CPU checks (count, resolution, blur, feature richness, and a
consecutive-frame overlap proxy) with human-readable messages, not a guarantee that
COLMAP will succeed, but a guard against the common, obvious ways it won't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Optional

# Thresholds (conservative; tuned to catch obvious failures, not borderline scenes)
HARD_MIN_FRAMES = 12          # COLMAP needs overlapping views to triangulate
RECOMMEND_MIN_FRAMES = 30
MIN_LONG_EDGE = 256           # pixels on the longer side
BLUR_VAR_THRESHOLD = 60.0     # variance of Laplacian; below = soft/blurry
MAX_BLURRY_FRACTION = 0.40
MIN_ORB_FEATURES = 150        # per frame; below = textureless
MAX_FEATUREPOOR_FRACTION = 0.40
MIN_OVERLAP_MATCHES = 30      # median good ORB matches between consecutive frames


class PreflightError(RuntimeError):
    """Input failed a fatal preflight check; message explains how to fix it."""


@dataclass
class PreflightReport:
    n_frames: int = 0
    min_long_edge: int = 0
    blurry_fraction: float = 0.0
    featurepoor_fraction: float = 0.0
    median_overlap: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [
            f"frames:        {self.n_frames}",
            f"min long edge: {self.min_long_edge}px",
            f"blurry:        {self.blurry_fraction:.0%}",
            f"feature-poor:  {self.featurepoor_fraction:.0%}",
            f"overlap (med): {self.median_overlap:.0f} matches/consecutive pair",
        ]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        return "\n".join(lines)


def _good_matches(matcher, des_a, des_b) -> int:
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return 0
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    good = 0
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good += 1
    return good


def analyze(frame_paths: List[Path]) -> PreflightReport:
    """Compute metrics over the frames. Pure analysis; never raises on bad scenes."""
    import cv2
    import numpy as np

    report = PreflightReport(n_frames=len(frame_paths))
    if not frame_paths:
        report.errors.append("no frames to analyze")
        return report

    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    long_edges: List[int] = []
    blur_vars: List[float] = []
    feature_counts: List[int] = []
    overlaps: List[int] = []
    prev_des = None

    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            report.warnings.append(f"could not read {p.name}; skipped")
            prev_des = None
            continue
        h, w = img.shape[:2]
        long_edges.append(max(h, w))
        blur_vars.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
        kp, des = orb.detectAndCompute(img, None)
        feature_counts.append(0 if des is None else len(des))
        if prev_des is not None:
            overlaps.append(_good_matches(matcher, prev_des, des))
        prev_des = des

    report.min_long_edge = min(long_edges) if long_edges else 0
    report.blurry_fraction = (
        sum(v < BLUR_VAR_THRESHOLD for v in blur_vars) / len(blur_vars)
        if blur_vars else 1.0
    )
    report.featurepoor_fraction = (
        sum(c < MIN_ORB_FEATURES for c in feature_counts) / len(feature_counts)
        if feature_counts else 1.0
    )
    report.median_overlap = float(median(overlaps)) if overlaps else 0.0

    _apply_rules(report)
    return report


def _apply_rules(r: PreflightReport) -> None:
    if r.n_frames < HARD_MIN_FRAMES:
        r.errors.append(
            f"only {r.n_frames} frames; need at least {HARD_MIN_FRAMES} overlapping "
            "views. Capture more angles (or raise the frame budget for a video)."
        )
    elif r.n_frames < RECOMMEND_MIN_FRAMES:
        r.warnings.append(
            f"{r.n_frames} frames is on the low side; {RECOMMEND_MIN_FRAMES}+ is safer."
        )

    if r.min_long_edge < MIN_LONG_EDGE:
        r.errors.append(
            f"smallest frame is {r.min_long_edge}px on its long edge; "
            f"need >= {MIN_LONG_EDGE}px. Use higher-resolution input."
        )

    if r.blurry_fraction > MAX_BLURRY_FRACTION:
        r.errors.append(
            f"{r.blurry_fraction:.0%} of frames look blurry; capture with less motion "
            "blur / better focus (move the camera slowly)."
        )

    if r.featurepoor_fraction > MAX_FEATUREPOOR_FRACTION:
        r.errors.append(
            f"{r.featurepoor_fraction:.0%} of frames are low on detail/texture; SfM needs "
            "trackable features. Avoid blank walls/sky-only shots; capture textured scenes."
        )

    if r.median_overlap < MIN_OVERLAP_MATCHES and r.n_frames >= HARD_MIN_FRAMES:
        r.errors.append(
            f"too little overlap between consecutive frames "
            f"(~{r.median_overlap:.0f} matches, need ~{MIN_OVERLAP_MATCHES}). "
            "Capture more views with smaller steps between them."
        )


def preflight(frame_paths: List[Path], strict: bool = True) -> PreflightReport:
    """Analyze frames and raise PreflightError if any fatal rule fails."""
    report = analyze(frame_paths)
    if strict and not report.ok:
        raise PreflightError(report.render())
    return report
