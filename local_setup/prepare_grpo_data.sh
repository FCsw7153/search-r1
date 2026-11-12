#!/usr/bin/env bash
# Full original train and independent evaluation splits; no smoke rows or sampling.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/local_setup/env.sh"
cd "$ROOT/local_setup"
for split in train test; do
    dst="$ROOT/data/grpo_qwen35/$split.parquet"
    if [[ ! -s "$dst" ]]; then
        mkdir -p "$(dirname "$dst")"
        "$ROOT/.venv/bin/python" "$ROOT/verl/examples/search_r1_qwen35/prepare_data.py" \
          --input "$ROOT/data/nq_hotpotqa_train/$split.parquet" --output "$dst.tmp"
        mv "$dst.tmp" "$dst"
    fi
done
