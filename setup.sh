#!/usr/bin/env bash
# SignVision one-time setup for Linux / macOS / WSL.
set -euo pipefail
cd "$(dirname "$0")"
command -v conda >/dev/null || { echo "Conda not found. Install Miniconda first."; exit 1; }
eval "$(conda shell.bash hook)"
if conda env list | grep -q "^signvision "; then
  conda env update -n signvision -f environment.yml --prune
else
  conda env create -f environment.yml
fi
conda activate signvision
# MediaPipe on Linux needs OpenGL ES / EGL libraries (Windows and macOS already have them).
if [[ "$(uname)" == "Linux" ]]; then conda install -y -q -c conda-forge libgles libegl; fi
(cd frontend && npm install --no-audit --no-fund && npm run build)
(cd backend && python -m scripts.download_speech_model
python -m scripts.download_tunnel || echo "Could not download cloudflared now; the app retries on Invite.")
echo "Setup finished. Start the app with ./run.sh"
