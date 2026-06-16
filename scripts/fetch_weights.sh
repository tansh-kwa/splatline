#!/usr/bin/env bash
# Fetch model weights into <dest> (default ./weights), verifying integrity.
# Run directly, or via setup.sh. Weights under ./weights are auto-discovered at runtime.
#
#   bash scripts/fetch_weights.sh ./weights
set -euo pipefail

DEST="${1:-./weights}"
GAN_DIR="$DEST/informative-drawings"
mkdir -p "$GAN_DIR"

# (lang-sam downloads its own SAM2 + GroundingDINO weights on first use, so the only
#  weights splatline fetches are the line-drawing GAN's.)

# --- informative-drawings GAN weights (Google Drive) ----------------------------
# Drive has no stable checksum to pin, so we verify the expected files exist instead.
GAN_ZIP_ID="1MIdHzecxz-z0uY3ARL_R40DlKcuQxiDk"
if ! ls "$GAN_DIR"/*_style/netG_A_latest.pth >/dev/null 2>&1; then
    echo "fetching informative-drawings GAN weights..."
    pip install --quiet gdown
    gdown --fuzzy "https://drive.google.com/uc?id=${GAN_ZIP_ID}" -O /tmp/model.zip
    unzip -o -q /tmp/model.zip -d "$GAN_DIR"
    rm -f /tmp/model.zip
fi

styles="$(ls -d "$GAN_DIR"/*_style 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/_style//' | tr '\n' ' ' || true)"
if [ -z "$styles" ]; then
    echo "ERROR: no <style>_style/netG_A_latest.pth found after unzip"; exit 1
fi
echo "GAN styles available: ${styles}"
echo "weights ready under ${DEST}"
