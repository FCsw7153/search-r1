#!/usr/bin/env bash
# Run in the SAME 4x B300 container as run_grpo_4b300.sh. No extra environment.
# E5 and the FP32 FAISS index (~60 GiB) share the first visible GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/local_setup/env.sh"
VISIBLE_GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=${EMBEDDING_GPU:-${VISIBLE_GPUS%%,*}}
export RETRIEVER_BATCH_SIZE=${RETRIEVER_BATCH_SIZE:-128}
export RETRIEVER_BATCH_WAIT_MS=${RETRIEVER_BATCH_WAIT_MS:-20}
export RETRIEVER_INDEX_DEVICE=${RETRIEVER_INDEX_DEVICE:-gpu}
export OMP_NUM_THREADS=${RETRIEVER_THREADS:-4}
export OPENBLAS_NUM_THREADS=${RETRIEVER_THREADS:-4}
export MKL_NUM_THREADS=${RETRIEVER_THREADS:-4}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs/retriever"
cd "$ROOT/local_setup"
PYTHON_BIN="$ROOT/.venv/bin/python"
LOG_FILE="$ROOT/logs/retriever/server.log"
PID_FILE="$ROOT/logs/retriever/$(hostname).pid"
exec 9>"$PID_FILE.lock"
flock -n 9 || { echo "Another retriever launch is in progress."; exit 1; }
if [[ -s "$PID_FILE" ]]; then
    read -r existing_pid < "$PID_FILE"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null &&
       [[ -r /proc/$existing_pid/cmdline ]] &&
       grep -aq 'serve_retriever_4b300.py' "/proc/$existing_pid/cmdline"; then
        if [[ ${1:-} != --restart ]]; then
            echo "Retriever already running: PID $existing_pid; log: $LOG_FILE"
            exit 0
        fi
        echo "Restarting only retriever PID $existing_pid; training is untouched."
        kill -TERM "$existing_pid"
        for ((attempt=0; attempt<120; attempt++)); do
            if ! kill -0 "$existing_pid" 2>/dev/null; then break; fi
            if [[ -r /proc/$existing_pid/status ]] && grep -q '^State:.*Z' "/proc/$existing_pid/status"; then break; fi
            sleep 1
        done
        if [[ -r /proc/$existing_pid/status ]] && ! grep -q '^State:.*Z' "/proc/$existing_pid/status"; then
            echo "Retriever did not stop within 120 seconds; no second copy started." >&2
            exit 1
        fi
    fi
fi
# Also recognize an older foreground service without a PID file.
if "$PYTHON_BIN" - <<'PORT_CHECK'
import os, socket
with socket.socket() as sock:
    sock.settimeout(2)
    raise SystemExit(0 if sock.connect_ex(('127.0.0.1', int(os.environ.get('RETRIEVER_PORT', '8000')))) == 0 else 1)
PORT_CHECK
then
    echo "Retriever port is already occupied; no duplicate service started."
    exit 0
fi
"$PYTHON_BIN" - <<'FAISS_CHECK'
import os, faiss
if os.environ['RETRIEVER_INDEX_DEVICE'] == 'gpu':
    missing = [name for name in ('StandardGpuResources', 'GpuClonerOptions', 'index_cpu_to_gpu')
               if not hasattr(faiss, name)]
    if missing:
        raise SystemExit('Cannot start GPU retrieval: this environment lacks GPU FAISS bindings. '
                         'No packages were installed, and CPU fallback is disabled.')
FAISS_CHECK
nohup "$PYTHON_BIN" serve_retriever_4b300.py </dev/null >>"$LOG_FILE" 2>&1 9>&- &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
echo "Retriever launched in background: PID $pid"
echo "Log: $LOG_FILE"
echo "Initialization is complete only after 'Application startup complete' appears in the log."
