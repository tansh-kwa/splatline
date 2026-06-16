"""Per-scene configuration: one config = one scene = one reproducible run.

The CLI maps user-facing flags onto a ``SceneConfig`` and writes it to
``runs/<scene>/config.yaml``. Re-running with that file reproduces the run (modulo the
documented GPU nondeterminism).

A scene has one **base look** (``style``) plus optional **per-subject overrides**
(``subjects``). A "look" is one of: ``contour`` / ``anime`` / ``watercolor`` (rendered via
the line-drawing GAN), ``photo`` (keep the original image), or ``white`` (blank paper).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOOKS = ("contour", "anime", "watercolor", "photo", "white")
LINE_LOOKS = ("contour", "anime", "watercolor")  # looks rendered via the GAN
RESOLUTIONS = ("512p", "720p", "1080p")
SWAP_POINTS = ("pre-train", "pre-sfm")


class ConfigError(ValueError):
    """Raised when a config is structurally invalid (bad enum, missing field)."""


@dataclass
class Timeouts:
    """Hard per-stage wall-clock limits (seconds) so one bad input can't hang."""

    ingest: int = 600
    preflight: int = 300
    stylize: int = 1800
    sfm: int = 3600
    train: int = 10800
    compress: int = 600


@dataclass
class SubjectStyle:
    """One subject in per-subject mode: a text prompt and the look to render it in."""

    prompt: str
    style: str = "contour"


@dataclass
class SceneConfig:
    """Everything needed to turn one input into one scene."""

    scene: str
    input: str

    # Look: one base look for the whole scene, plus optional per-subject overrides.
    style: str = "contour"                        # base look (any of LOOKS)
    subjects: Optional[List[SubjectStyle]] = None  # per-subject overrides

    # Geometry / training
    resolution: str = "512p"                      # 512p | 720p | 1080p
    frames: int = 150                             # frame budget sampled from video
    swap_point: str = "pre-train"                 # pre-train | pre-sfm  (hidden default)
    from_colmap: Optional[str] = None             # reuse prepared COLMAP output, skip SfM

    # Output
    ksplat: bool = False                          # also emit portable .ksplat

    # Internal (not written to the per-run config; override only if you must)
    timeouts: Timeouts = field(default_factory=Timeouts)

    # ------------------------------------------------------------------ helpers
    @property
    def has_subjects(self) -> bool:
        return bool(self.subjects)

    def summary(self) -> str:
        """Short human description, e.g. 'contour' or 'per-subject[tractor=anime] on photo'."""
        if self.subjects:
            parts = ", ".join(f"{s.prompt}={s.style}" for s in self.subjects)
            return f"per-subject[{parts}] on {self.style}"
        return self.style

    def validate(self) -> "SceneConfig":
        if self.style not in LOOKS:
            raise ConfigError(f"style must be one of {LOOKS}, got {self.style!r}")
        if self.resolution not in RESOLUTIONS:
            raise ConfigError(
                f"resolution must be one of {RESOLUTIONS}, got {self.resolution!r}"
            )
        if self.swap_point not in SWAP_POINTS:
            raise ConfigError(
                f"swap_point must be one of {SWAP_POINTS}, got {self.swap_point!r}"
            )
        if self.frames < 8:
            raise ConfigError(
                f"frames must be >= 8 (COLMAP needs overlap), got {self.frames}"
            )
        if not self.scene:
            raise ConfigError("scene name is required")
        if not self.input:
            raise ConfigError("input path is required")
        if self.subjects:
            for s in self.subjects:
                if not s.prompt:
                    raise ConfigError("each subject needs a non-empty prompt")
                if s.style not in LOOKS:
                    raise ConfigError(
                        f"subject {s.prompt!r} style must be one of {LOOKS}, got {s.style!r}"
                    )
        elif self.style not in LINE_LOOKS:
            # photo/white only make sense as a *base* under subjects; alone they draw nothing.
            raise ConfigError(
                f"base look {self.style!r} only makes sense with subjects; for a plain "
                f"scene use one of {LINE_LOOKS}"
            )
        return self

    # ------------------------------------------------------------- (de)serialize
    def to_dict(self) -> Dict[str, Any]:
        """Minimal, human-facing config: required fields + any non-default choice.

        Defaults (timeouts, swap_point=pre-train, off flags) are omitted to keep the
        saved config.yaml easy to read; they're filled back in by from_dict().
        """
        d: Dict[str, Any] = {
            "scene": self.scene,
            "input": self.input,
            "style": self.style,
            "resolution": self.resolution,
            "frames": self.frames,
        }
        if self.subjects:
            d["subjects"] = [dataclasses.asdict(s) for s in self.subjects]
        if self.ksplat:
            d["ksplat"] = True
        if self.from_colmap:
            d["from_colmap"] = self.from_colmap
        if self.swap_point != "pre-train":
            d["swap_point"] = self.swap_point
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneConfig":
        data = dict(data)
        if "timeouts" in data and isinstance(data["timeouts"], dict):
            data["timeouts"] = Timeouts(**data["timeouts"])
        if data.get("subjects"):
            data["subjects"] = [
                SubjectStyle(**s) if isinstance(s, dict) else s for s in data["subjects"]
            ]
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path: Path) -> "SceneConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
