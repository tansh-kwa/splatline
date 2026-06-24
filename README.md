# splatline

splatline augments the standard pipeline for generating 3D Gaussian splats to create stylized 
3D scenes that resemble 3D line drawings. The work duct tapes together Caroline Chan's [informative line drawings](https://carolineec.github.io/informative_drawings/)
with [gsplat's](https://github.com/nerfstudio-project/gsplat) tools for creating 3D gaussian splats. This project originally used MrNeRF's gaussian-splatting-cuda, now [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio). 

Splatline works with videos and image collections, and creates .ply scenes. The full method and background are described in this blog post: **[3D Line Drawings](https://amritkwatra.com/experiments/3d-line-drawings)**.

> **AI Disclosure:** I used Claude Code to put this repository together.

## Setup

splatline orchestrates a few tools that you install **yourself** first. Splatline does not build them for you. You'll want a **modern NVIDIA GPU**, since COLMAP and splat-training stages aren't practical on CPU. Converting individual images to line drawings is however possible on modern CPUs. 

| Requirement | Used for | Where to install |
|---|---|---|
| **gsplat** | Gaussian-splat training | https://github.com/nerfstudio-project/gsplat |
| **COLMAP** | camera poses (Structure-from-Motion) | https://colmap.github.io/install.html |
| **ffmpeg** | sampling frames from video | https://ffmpeg.org/download.html |
| **lang-sam** | subject segmentation (only for `--subject`) | https://github.com/luca-medeiros/lang-segment-anything |

`splatline check` (after install) tells you which of these are present.

## Install

```bash
git clone --recurse-submodules git@github.com:tansh-kwa/splatline.git
cd splatline
./setup.sh          # checks the requirements above, installs splatline + its submodules
splatline check
```

`setup.sh` does **not** install the heavy requirements above. It installs splatline
itself and fetches the model weights.

**Model weights.** The line-drawing model is *informative-drawings* by Chan et al., and its
pretrained weights come from t
([carolineec/informative-drawings](https://github.com/carolineec/informative-drawings)).
`setup.sh` downloads them automatically into `./weights` (you can re-run
`scripts/fetch_weights.sh ./weights` at any time). lang-sam fetches its own segmentation
weights on first use.

## Usage

Point splatline at a video or a folder of images and pick a style. The result is written
to `runs/<scene>/scene.ply`.

```bash
# Line drawing (also: --style anime)
splatline run ./scene.mp4 --style contour

# Watercolor: line drawing with a soft color wash
splatline run ./scene.mp4 --style watercolor

# Collage: draw only the named subject (in --style), leave the rest blank (uses lang-sam)
splatline run ./scene.mp4 --subject "tractor" --style contour
```

`--subject` draws that subject in the chosen `--style` over a blank (white) background;
the example above renders the tractor as a contour line drawing.

Common options:

- `--resolution 512p|720p|1080p`: higher is sharper but slower and needs more VRAM (default `512p`).
- `--fps F`: frames per second to sample from a video (default `2`).
- `--ksplat`: also write a compact `scene.ksplat` for sharing (needs Node).
- `--from-colmap DIR`: reuse a COLMAP result you already have and skip Structure-from-Motion.

### Define a run in a config file

Instead of flags, a run can also be written as a YAML file and run directly:

```yaml
# my-run.yaml   ->   splatline run my-run.yaml
scene: my-scene
input: ./scene.mp4
style: watercolor
resolution: 720p
fps: 2
ksplat: true
```

Every run also writes its resolved config to `runs/<scene>/config.yaml`, so you can
reproduce or tweak a scene by re-running `splatline run runs/<scene>/config.yaml`.

**Different styles for different subjects.** For finer control, such as a photoreal
background with a line-drawn subject, write a config file and run it directly. `style` is
the base look; each subject overrides it:

```yaml
# my-scene.yaml   ->   splatline run my-scene.yaml
scene: tractor-yard
input: ./scene.mp4
style: photo                 # base look: contour | anime | watercolor | photo | white
subjects:
  - { prompt: "tractor", style: contour }
  - { prompt: "barn",    style: watercolor }
```

A full template is in [`configs/per-subject.example.yaml`](configs/per-subject.example.yaml).


## License

MIT. See [`LICENSE`](LICENSE). The required dependencies above keep their own licenses.
