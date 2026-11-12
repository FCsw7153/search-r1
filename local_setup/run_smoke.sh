#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$SEARCH_R1_ROOT"
python -u local_setup/smoke_retriever.py
python -u local_setup/serve_smoke.py > local_setup/serve_smoke.log 2>&1 &
searchr1_server_pid=$!
trap 'kill "$searchr1_server_pid" 2>/dev/null || true' EXIT
python - <<'PY'
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen('http://127.0.0.1:18080/openapi.json',timeout=2)
        break
    except OSError: time.sleep(1)
else: raise RuntimeError('Smoke retriever failed to become ready')
PY
python -u local_setup/smoke_train.py
