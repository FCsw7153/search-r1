#!/usr/bin/env bash
# Start the local E5 + FAISS retrieval server in the background.
# Output is appended to $SEARCH_R1_HOME/logs/retriever.log.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${CODE_DIR}/.." && pwd)}
DATA_DIR=${SEARCH_R1_HOME}/data
LOG=${SEARCH_R1_HOME}/logs/retriever.log
mkdir -p "${SEARCH_R1_HOME}/logs"

CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU:-0} nohup python3 "${SCRIPT_DIR}/retrieval_server.py" --faiss_gpu \
    --index_path "${DATA_DIR}/retrieval/e5_Flat.index" \
    --corpus_path "${DATA_DIR}/retrieval/wiki-18.jsonl" \
    --retriever_model "${DATA_DIR}/models/e5-base-v2" \
    >> "${LOG}" 2>&1 &
pid=$!

echo "retriever pid: ${pid}"
echo "log          : ${LOG}"
echo "check        : curl -s http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{\"queries\": [\"who wrote the wealth of nations\"], \"topk\": 3, \"return_scores\": false}'"
