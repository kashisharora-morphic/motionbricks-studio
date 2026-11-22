#!/usr/bin/env bash
# One-time setup: clone MotionBricks next to this folder, fetch its Git LFS files
# (SOMA ONNX model + BVH clips) without needing git-lfs, and create the Python env.
set -euo pipefail
cd "$(dirname "$0")"
MB=../motionbricks
RAW=https://media.githubusercontent.com/media/NVlabs/motionbricks/main

[ -d "$MB" ] || git clone --depth 1 https://github.com/NVlabs/motionbricks.git "$MB"

fetch() { # replace an LFS pointer file with its real content
  local f="$1"
  if head -c 40 "$MB/$f" | grep -q "git-lfs.github.com"; then
    echo "fetching $f"
    curl -fL --retry 3 -o "$MB/$f.part" "$RAW/$f" && mv "$MB/$f.part" "$MB/$f"
  fi
}
fetch motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx
(cd "$MB" && find assets/motions/soma -name '*.bvh' ! -name '._*') | while read -r f; do fetch "$f"; done

[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy onnxruntime
(cd web && npm install --no-audit --no-fund)
echo "Setup done. Start with ./run.sh"
