# splatline

splatline turns a video (or a set of photos) of a scene into a **3D Gaussian-splat model
that looks hand-drawn** (a line drawing, a watercolor, or a subject-only collage). It
stays consistent as you move the camera, because the style is baked into the 3D
reconstruction rather than painted on at the end.

You give it footage and a style; it returns a `scene.ply` you can open in any 3D Gaussian
splat viewer. Under the hood it samples frames, redraws them with a line-drawing model,
recovers camera geometry with COLMAP, and trains a Gaussian splat on the redrawn images.

The full method and background are in the blog post:
**[3D Line Drawings](https://amritkwatra.com/experiments/3d-line-drawings)**.

> [!NOTE]
> **AI disclosure:** this repository was put together using Claude Code, primarily working through Opus 4.8.

## Setup

splatline orchestrates a few heavyweight, GPU/CUDA tools that you install **yourself**
first; it does not build them for you. You'll want a **modern NVIDIA GPU** (a recent
driver and a CUDA 12.x stack), since the line-drawing and splat-training stages aren't
practical on CPU.

| Requirement | Used for | Where to install |
|---|---|---|
| **PyTorch** (CUDA build) | the line-drawing model | https://pytorch.org/get-started/locally/ |
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
pretrained weights come from that project
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
