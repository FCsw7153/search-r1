#!/usr/bin/env bash
# Start the local E5 + FAISS retrieval server in the background.
# Output is appended to $SEARCH_R1_HOME/logs/retriever.log.
#
# LOCAL_ROOT 里有 stage_local.sh 打好完成标记（data/.retrieval.ok、.venv.ok）的本地副本时
# 优先用本地盘检索数据和本地 venv 的 python（索引在共享盘上随机读很慢）；不完整的本地
# 副本回落到共享盘。设 RETRIEVAL_DATA_ROOT 可指定别的数据目录。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${CODE_DIR}/.." && pwd)}
LOCAL_ROOT=${LOCAL_ROOT-/home/work/search_r1_local}
RETRIEVAL_DATA_ROOT=${RETRIEVAL_DATA_ROOT:-${SEARCH_R1_HOME}/data}
LOG=${SEARCH_R1_HOME}/logs/retriever.log
mkdir -p "${SEARCH_R1_HOME}/logs"

DATA_DIR=${RETRIEVAL_DATA_ROOT}
if [ -n "${LOCAL_ROOT}" ] && [ -f "${LOCAL_ROOT}/data/.retrieval.ok" ]; then
    DATA_DIR=${LOCAL_ROOT}/data
    echo "retrieval data: local copy ${DATA_DIR}"
elif [ -n "${LOCAL_ROOT}" ] && [ -e "${LOCAL_ROOT}/data" ]; then
    echo "retrieval data: local copy incomplete（缺少 ${LOCAL_ROOT}/data/.retrieval.ok），回落到 ${DATA_DIR}"
else
    echo "retrieval data: ${DATA_DIR}    # 本地副本不完整时回落到共享盘"
fi

# VERL_PYTHON 显式设置时优先；否则本地 venv 有完成标记（.venv.ok）就用它。
PYTHON=${VERL_PYTHON:-python3}
if [ -z "${VERL_PYTHON:-}" ] && [ -n "${LOCAL_ROOT}" ] && [ -f "${LOCAL_ROOT}/.venv.ok" ]; then
    PYTHON=${LOCAL_ROOT}/.venv/bin/python
    echo "python        : local venv ${PYTHON}"
elif [ -z "${VERL_PYTHON:-}" ] && [ -n "${LOCAL_ROOT}" ] && [ -e "${LOCAL_ROOT}/.venv" ]; then
    echo "python        : 本地 venv 不完整（缺少 ${LOCAL_ROOT}/.venv.ok），回落到 ${PYTHON}"
else
    echo "python        : ${PYTHON}"
fi

CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU:-0} nohup "${PYTHON}" "${SCRIPT_DIR}/retrieval_server.py" --faiss_gpu \
    --index_path "${DATA_DIR}/retrieval/e5_Flat.index" \
    --corpus_path "${DATA_DIR}/retrieval/wiki-18.jsonl" \
    --retriever_model "${DATA_DIR}/models/e5-base-v2" \
    >> "${LOG}" 2>&1 &
pid=$!

echo "retriever pid: ${pid}"
echo "log          : ${LOG}"
echo "check        : curl -s http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{\"queries\": [\"who wrote the wealth of nations\"], \"topk\": 3, \"return_scores\": false}'"
