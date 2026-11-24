#!/usr/bin/env bash
# 在新 pod 上把 Search-R1 需要的一切准备到本地盘，并启动检索服务。
#
#   bash verl/examples/search_r1/pod_setup_local.sh
#   MODEL_REPO=Qwen/Qwen3-8B-Base RETRIEVER_GPU=0 bash verl/examples/search_r1/pod_setup_local.sh
#
# 布局（全部在容器本地盘）：
#   SEARCH_R1_HOME=/home/work/sr1           代码（git clone，branch verl-rebase）
#   LOCAL_ROOT=/home/work/search_r1_local   venv + data（模型、检索数据、parquet）
#
# 共享盘（默认 /home/work/mshop-agent-1/liang.mao/Search-R1）只作为 stage_local.sh 的
# 拷贝源，本脚本绝不往共享盘写：那块 spfs FUSE 盘在负载下挂死过，会拖住所有 job。
#
# 幂等：可以反复执行（git fast-forward、stage_local 有完成标记、HF snapshot 会跳过已有
# 文件、检索服务已在跑就复用）。最后脚本用 `tail --pid` 在**前台**持有检索服务进程，
# 所以要在持久终端（Herdr / hc run --bg）里跑，别用 hc exec（父进程退出会带走子进程）。
#
# 环境变量：
#   MODEL_REPO     要下载的 HF 模型（默认 Qwen/Qwen3-8B-Base）
#   MODEL_NAME     本地目录名（默认取 MODEL_REPO 的最后一段）
#   SHARED_ROOT    只读的共享工作区（默认 /home/work/mshop-agent-1/liang.mao/Search-R1）
#   REPO_URL       代码仓库（默认 https://github.com/FCsw7153/search-r1.git）
#   BRANCH         分支（默认 verl-rebase）
#   RETRIEVER_GPU  检索服务用哪张卡（默认 0）
#   UV_BIN         找不到 uv 时显式指定
set -euo pipefail

SEARCH_R1_HOME=${SEARCH_R1_HOME:-/home/work/sr1}
LOCAL_ROOT=${LOCAL_ROOT:-/home/work/search_r1_local}
SHARED_ROOT=${SHARED_ROOT:-/home/work/mshop-agent-1/liang.mao/Search-R1}
REPO_URL=${REPO_URL:-https://github.com/FCsw7153/search-r1.git}
BRANCH=${BRANCH:-verl-rebase}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3-8B-Base}
MODEL_NAME=${MODEL_NAME:-${MODEL_REPO##*/}}
RETRIEVER_GPU=${RETRIEVER_GPU:-0}

# pod 的 python 发 HTTPS 请求（HF 下载）需要显式 CA；不设置会报证书错误。
export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}

VENV=${LOCAL_ROOT}/.venv
PY=${VENV}/bin/python
MODEL_DIR=${LOCAL_ROOT}/data/models/${MODEL_NAME}
PTH_PATTERN=${VENV}/lib/python*/site-packages/__editable__.verl-*.pth

say() {
    printf '== %s %s\n' "$(date -u +%H:%M:%S)" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

########################### 代码 ###########################
say "code ${SEARCH_R1_HOME} (${BRANCH})"
if [ ! -d "${SEARCH_R1_HOME}/.git" ]; then
    if [ -e "${SEARCH_R1_HOME}" ] && [ -n "$(ls -A "${SEARCH_R1_HOME}" 2>/dev/null)" ]; then
        die "${SEARCH_R1_HOME} 已存在但不是 git 仓库；请先挪走或手动 clone"
    fi
    mkdir -p "$(dirname "${SEARCH_R1_HOME}")"
    git clone -q -b "${BRANCH}" "${REPO_URL}" "${SEARCH_R1_HOME}"
else
    # 直接从 REPO_URL fetch，不依赖 origin 是 HTTPS 还是 SSH。
    git -C "${SEARCH_R1_HOME}" fetch -q "${REPO_URL}" "${BRANCH}"
    if [ "$(git -C "${SEARCH_R1_HOME}" symbolic-ref --short -q HEAD || true)" != "${BRANCH}" ]; then
        if git -C "${SEARCH_R1_HOME}" show-ref --verify -q "refs/heads/${BRANCH}"; then
            git -C "${SEARCH_R1_HOME}" checkout -q "${BRANCH}"
        else
            git -C "${SEARCH_R1_HOME}" checkout -q -b "${BRANCH}" "FETCH_HEAD"
        fi
    fi
    # fast-forward only：pod 上有未提交修改或本地提交时明确报错，不静默覆盖。
    if ! git -C "${SEARCH_R1_HOME}" merge -q --ff-only FETCH_HEAD; then
        die "${SEARCH_R1_HOME} 无法 fast-forward 到 origin/${BRANCH}（本地有修改/提交？）"
    fi
fi
say "code at $(git -C "${SEARCH_R1_HOME}" log --oneline -1)"

########################### venv + 检索数据 ###########################
[ -x "${SHARED_ROOT}/.venv/bin/python" ] || die "共享盘上找不到 venv：${SHARED_ROOT}/.venv/bin/python"
say "stage venv + retrieval: ${SHARED_ROOT} -> ${LOCAL_ROOT}"
# stage_local.sh 的 SEARCH_R1_HOME 是**拷贝源**，不是代码目录；代码目录由脚本自身路径决定。
STAGE_LOCAL=${SEARCH_R1_HOME}/verl/examples/search_r1/stage_local.sh
SEARCH_R1_HOME="${SHARED_ROOT}" LOCAL_ROOT="${LOCAL_ROOT}" bash "${STAGE_LOCAL}" all

# 共享 venv 的可编辑安装指向共享盘的 verl；训练必须用本地 clone 的代码。
shopt -s nullglob
pths=(${PTH_PATTERN})
shopt -u nullglob
[ "${#pths[@]}" -ge 1 ] || die "找不到可编辑安装的 .pth：${PTH_PATTERN}"
printf '%s\n' "${SEARCH_R1_HOME}/verl" > "${pths[0]}"
say "editable .pth -> $(cat "${pths[0]}")"
verl_file=$(cd /tmp && "${PY}" -c 'import verl; print(verl.__file__)') \
    || die "本地 venv 无法 import verl（检查 .pth）"
case "${verl_file}" in
    "${SEARCH_R1_HOME}/verl/"*) say "import verl -> ${verl_file}" ;;
    *) die "import verl 没有走本地 clone：${verl_file}（应位于 ${SEARCH_R1_HOME}/verl）" ;;
esac

########################### swanlab ###########################
if "${PY}" -c 'import swanlab' >/dev/null 2>&1; then
    say "swanlab already installed"
else
    uv_bin=${UV_BIN:-}
    if [ -z "${uv_bin}" ]; then
        for candidate in "$(command -v uv || true)" "${SHARED_ROOT}/.tools/uv" \
            "${SEARCH_R1_HOME}/.tools/uv" "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
            if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
                uv_bin=${candidate}
                break
            fi
        done
    fi
    if [ -z "${uv_bin}" ]; then
        say "WARNING: 找不到 uv，跳过 swanlab 安装；有 ~/.swanlab_key 时训练会因为缺 swanlab 失败" >&2
    else
        say "install swanlab with ${uv_bin}"
        # venv 里没有 pip，只能用 uv pip --python；不要用 uv run/sync（会按 uv.lock 重建环境）。
        "${uv_bin}" pip install --python "${PY}" swanlab
    fi
fi

########################### 模型 ###########################
mkdir -p "${LOCAL_ROOT}/data/models"
model_ready=1
[ -f "${MODEL_DIR}/config.json" ] || model_ready=0
[ -f "${MODEL_DIR}/generation_config.json" ] || model_ready=0
[ "$(cat "${MODEL_DIR}/.source_repo" 2>/dev/null || true)" = "${MODEL_REPO}" ] || model_ready=0
if [ "${model_ready}" = 1 ] && ! ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    model_ready=0
fi
if [ "${model_ready}" = 1 ]; then
    say "model already at ${MODEL_DIR}"
else
    say "download ${MODEL_REPO} -> ${MODEL_DIR}"
    if ! MODEL_REPO="${MODEL_REPO}" MODEL_DIR="${MODEL_DIR}" "${PY}" - <<'PY'
import os

from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id=os.environ["MODEL_REPO"],
        local_dir=os.environ["MODEL_DIR"],
    )
)
PY
    then
        die "模型下载失败：${MODEL_REPO}（检查网络 / SSL_CERT_FILE）"
    fi
    # 记录来源，MODEL_REPO 变了（即使目录名相同）下次会重新下载。
    printf '%s\n' "${MODEL_REPO}" > "${MODEL_DIR}/.source_repo"
fi

# Search-R1 用纯文本协议；Qwen3 generation_config 默认只有 <|endoftext|>，加上 <|im_end|>。
MODEL_DIR="${MODEL_DIR}" "${PY}" - <<'PY'
import json
import os

path = os.path.join(os.environ["MODEL_DIR"], "generation_config.json")
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
else:
    config = {}
config["eos_token_id"] = [151645, 151643]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print("eos_token_id ->", config["eos_token_id"])
PY

########################### 训练数据 ###########################
copy_dataset() {
    local name=$1
    local src=${SHARED_ROOT}/data/${name}
    local dst=${LOCAL_ROOT}/data/${name}
    if [ ! -d "${src}" ]; then
        say "data: 共享盘没有 ${name}，跳过（用到时再自己放到 ${dst}）"
        return 0
    fi
    mkdir -p "${LOCAL_ROOT}/data"
    if command -v rsync >/dev/null 2>&1; then
        # 不用 --delete：本地可能还有自己生成的 parquet。
        rsync -a "${src}/" "${dst}/"
    elif [ ! -d "${dst}" ]; then
        cp -a "${src}" "${LOCAL_ROOT}/data/"
    else
        cp -an "${src}/." "${dst}/"
    fi
    say "data: ${dst}"
    ls -lh "${dst}" | head -n 10
}

copy_dataset search_r1
# 过滤后的数据是本地产物，共享盘上通常没有；没有也不影响 setup。
copy_dataset search_r1_filtered

if [ ! -f "${LOCAL_ROOT}/data/search_r1/train.parquet" ]; then
    say "WARNING: 缺少 ${LOCAL_ROOT}/data/search_r1/train.parquet，训练前需要准备数据" >&2
fi

########################### 检索服务 ###########################
retriever_pid=""
if pgrep -f "[r]etrieval_server.py" >/dev/null 2>&1; then
    retriever_pid=$(pgrep -f "[r]etrieval_server.py" | head -n 1 || true)
    say "retriever already running: pid ${retriever_pid}（复用，不重复启动）"
else
    say "start retriever (GPU ${RETRIEVER_GPU})"
    if ! retriever_out=$(cd "${SEARCH_R1_HOME}" && \
        SEARCH_R1_HOME="${SEARCH_R1_HOME}" LOCAL_ROOT="${LOCAL_ROOT}" RETRIEVER_GPU="${RETRIEVER_GPU}" \
        bash "${SEARCH_R1_HOME}/verl/examples/search_r1/start_retriever.sh"); then
        die "start_retriever.sh 失败，看 ${SEARCH_R1_HOME}/logs/retriever.log"
    fi
    printf '%s\n' "${retriever_out}"
    retriever_pid=$(printf '%s\n' "${retriever_out}" | awk '/retriever pid:/ {print $3; exit}')
fi

[ -n "${retriever_pid}" ] || die "无法确定 retriever pid，看 ${SEARCH_R1_HOME}/logs/retriever.log"
sleep 2
if ! kill -0 "${retriever_pid}" 2>/dev/null; then
    tail -n 50 "${SEARCH_R1_HOME}/logs/retriever.log" >&2 || true
    die "retriever 进程 ${retriever_pid} 已退出"
fi

say "SETUP_DONE retriever pid ${retriever_pid} log ${SEARCH_R1_HOME}/logs/retriever.log"
say "检查: curl -s http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
-d '{\"queries\": [\"who wrote the wealth of nations\"], \"topk\": 3, \"return_scores\": false}'"
say "模型: ${MODEL_DIR}"
say "训练数据: ${LOCAL_ROOT}/data/search_r1 (filtered: ${LOCAL_ROOT}/data/search_r1_filtered)"
say "下一步（另开终端）: bash ${SEARCH_R1_HOME}/verl/examples/search_r1/run_qwen3_base_local.sh <EXP> 0"

# 前台持有检索服务 pid：这个终端活着，检索服务就不会被父进程退出带走。
tail --pid="${retriever_pid}" -f /dev/null
