#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_qwen35.sh"
cd "$SEARCH_R1_ROOT"
python - <<'PY'
import torch
assert torch.cuda.device_count() == 4, 'Expose exactly four B300 GPUs for this validation.'
PY
for model in Qwen3.5-2B Qwen3.5-4B; do
  torchrun --standalone --nnodes=1 --nproc-per-node=4 local_setup/validate_qwen35.py --model "$model"
done
