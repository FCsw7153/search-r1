# One environment for verl training, vLLM and retrieval.
SEARCH_R1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the existing uv environment; never install or synchronize during startup.
export VIRTUAL_ENV="$SEARCH_R1_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHON_BIN="$VIRTUAL_ENV/bin/python"
# Report the path resolved IN THIS container, so failures identify the actual environment.
printf 'Using existing uv environment: %s\n' "$VIRTUAL_ENV"
# Surface the actual interpreter error instead of misdiagnosing it as missing FAISS.
"$PYTHON_BIN" - <<'ENV_CHECK' || return 1
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Existing environment requires Python 3.12, but {sys.executable} runs "
        f"{sys.version.split()[0]}. No packages were installed or changed."
    )
ENV_CHECK
export PYTHONPATH="$SEARCH_R1_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_MODELSCOPE=False
export HF_HOME="$SEARCH_R1_ROOT/data/hf-cache"
export XDG_CACHE_HOME="$SEARCH_R1_ROOT/.cache"
export TORCH_HOME="$SEARCH_R1_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$SEARCH_R1_ROOT/.cache/triton"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export RAY_agent_register_timeout_ms=60000
unset RAY_TMPDIR
