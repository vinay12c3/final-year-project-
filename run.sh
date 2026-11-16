#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
conda activate signvision
[[ -f frontend/dist/index.html ]] || (cd frontend && npm run build)
echo "SignVision running at http://localhost:8000  (Ctrl+C to stop)"
cd backend && exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
