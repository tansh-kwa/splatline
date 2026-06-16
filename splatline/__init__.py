"""splatline: turn a video or image set into a stylized 3D Gaussian-splat scene.

Pipeline: ingest -> preflight -> stylize (line drawing) -> SfM (COLMAP) ->
train (gsplat, with the line-drawing image swap) -> scene.ply [-> scene.ksplat].

This tool only *generates* scenes; it does not ship a viewer.
"""

__version__ = "0.1.0"
