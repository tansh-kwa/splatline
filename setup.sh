#!/usr/bin/env bash
# splatline install. ASSUMES COLMAP and gsplat are already installed;
# this script does NOT build them. It installs splatline's own Python deps, checks
# out the pinned GAN + trainer submodules, fetches model weights into ./weights
# (auto-discovered at runtime), and installs the `splatline` CLI into the active env.
#
#   ./setup.sh                  # use the active python/pip
#   PY=python3.10 ./setup.sh    # pin the interpreter
#
# Tested-on: Ubuntu 22.04, Python 3.10, COLMAP 3.x, CUDA 12.1 torch + gsplat.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
PIP="$PY -m pip"

echo "==> splatline native setup (assumes COLMAP + gsplat already installed)"
echo "    python: $($PY --version 2>&1)"

# 1. Prerequisites this script will NOT install for you (the heavy / CUDA bits).
fail=0
command -v colmap >/dev/null 2>&1 || { echo "  MISSING: colmap  (https://colmap.github.io/install.html)"; fail=1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "  MISSING: ffmpeg"; fail=1; }
$PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || { echo "  MISSING: torch with CUDA (install the wheel matching your CUDA)"; fail=1; }
$PY -c "import gsplat" 2>/dev/null \
    || { echo "  MISSING: gsplat (pip install gsplat, matching your torch/CUDA)"; fail=1; }
if [ "$fail" -ne 0 ]; then
    echo "==> Resolve the MISSING prerequisites above, then re-run."
    exit 1
fi
# lang-sam is only needed for --subject (collage); warn rather than block.
$PY -c "import lang_sam" 2>/dev/null \
    || echo "  note: lang-sam not installed, needed only for --subject collage"

# 2. Pinned submodules (recursive: gsplat vendors glm and the example libs as nested
#    submodules; without --recursive the CUDA build fails on a missing glm header).
echo "==> fetching submodules (informative-drawings GAN, gsplat example trainer)"
git submodule update --init --recursive third_party/informative-drawings third_party/gsplat

# 3. Python deps. splatline core + the trainer's glue deps.
echo "==> installing splatline core deps"
$PIP install -r requirements.txt
# Build tools for the CUDA-extension deps below (they build against your installed torch).
$PIP install wheel setuptools ninja

echo "==> installing gsplat example-trainer deps (skipping NCore/lidar-only extras)"
# --no-build-isolation: the trainer's CUDA extensions (e.g. fused-ssim) import torch at
# build time, which an isolated build env doesn't have. Skip nvidia-ncore / ppisp: they
# are only needed for the lidar/NCore data path, not COLMAP.
grep -vE '^(nvidia-ncore|ppisp )' third_party/gsplat/examples/requirements.txt > /tmp/splatline-trainer-reqs.txt
$PIP install --no-build-isolation -r /tmp/splatline-trainer-reqs.txt
rm -f /tmp/splatline-trainer-reqs.txt

# The example trainer also needs gsplat's local scene/stage helper libraries.
echo "==> installing gsplat example scene/stage libs"
$PIP install --no-build-isolation -e third_party/gsplat/libs/scene -e third_party/gsplat/libs/stage

echo "==> installing the splatline CLI"
$PIP install -e .

# 4. Model weights -> ./weights (auto-discovered at runtime; no env vars needed).
echo "==> fetching model weights into ./weights"
bash scripts/fetch_weights.sh ./weights

# 5. Verify.
echo "==> verifying environment"
splatline check || true
echo
echo "==> done. Try:"
echo "    splatline run ./your-video.mp4 --style contour --resolution 512p"
