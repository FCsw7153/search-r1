#!/usr/bin/env bash
set -euo pipefail
SEARCH_R1_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export UV_PROJECT_ENVIRONMENT="$SEARCH_R1_ROOT/.venv"
export UV_CACHE_DIR="$SEARCH_R1_ROOT/.cache/uv"
export UV_LINK_MODE=hardlink
export UV_SYSTEM_CERTS=true
mkdir -p "$SEARCH_R1_ROOT/local_setup/verl_new_env_spec"
cp "$SEARCH_R1_ROOT/verl/pyproject.toml" "$SEARCH_R1_ROOT/verl/uv.lock" "$SEARCH_R1_ROOT/local_setup/verl_new_env_spec/"
"$SEARCH_R1_ROOT/.tools/uv" sync --project "$SEARCH_R1_ROOT/local_setup/verl_new_env_spec" --frozen --no-install-project --extra fsdp --extra vllm
# The retrieval service shares this environment; faiss is not part of verl's lock.
# CPU/GPU distributions share the faiss namespace; never install both.
"$SEARCH_R1_ROOT/.tools/uv" pip uninstall --python "$SEARCH_R1_ROOT/.venv/bin/python" faiss-cpu faiss-gpu-cu12
"$SEARCH_R1_ROOT/.tools/uv" pip install --python "$SEARCH_R1_ROOT/.venv/bin/python" -r "$SEARCH_R1_ROOT/local_setup/retrieval-requirements.txt"
