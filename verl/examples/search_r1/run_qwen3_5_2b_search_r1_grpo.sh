#!/usr/bin/env bash
# GRPO | Qwen3.5-2B | FSDP2 | Search-R1 with a local E5 + FAISS retriever
# This file lives in <workspace>/verl/examples/search_r1/, so the code dir = ../..
# and SEARCH_R1_HOME = ../../.. (the workspace/git root with data/, experiments/, logs/).
# Requires the venv (source "$SEARCH_R1_HOME/.venv/bin/activate"); never use `uv run`,
# which would replace the environment's faiss-gpu without it being in uv.lock.
#
#   EXP=<name> [BG=1] [RESUME=1] [GPUS=0,1] [VAL=sub|full] [CKPT_ROOT=<dir>] [LOCAL_ROOT=<dir>] \
#       bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
#   SMOKE=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
#   [DRY_RUN=1] ...            print resolved paths and the command, create nothing
#
# CKPT_ROOT（默认空 = 旧行为）把 checkpoint 写到本地盘（见 README）；LOCAL_ROOT 里有
# stage_local.sh 完成标记的 .venv 时，未显式设置 VERL_PYTHON 就用它（Ray worker 继承
# sys.executable，所以整个集群都走本地 venv）。

set -euo pipefail

########################### workspace ###########################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_FILE=${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")
CODE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${CODE_DIR}/.." && pwd)}

# GPUS picks the devices for this job (e.g. 0,1 or 2,3 for two jobs on one node);
# trainer.n_gpus_per_node is derived from the count.
GPUS=${GPUS:-0,1,2,3}
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPUS 必须是逗号分隔的设备号，例如 0,1: ${GPUS}" >&2
    exit 1
fi
GPU_COUNT=$(awk -F, '{print NF}' <<< "${GPUS}")
# VAL=full uses the full 7-dataset test.parquet, VAL=sub the stratified test_sub.parquet.
VAL=${VAL:-full}
case "${VAL}" in
    full) TEST_FILE=${TEST_FILE:-${SEARCH_R1_HOME}/data/search_r1/test.parquet} ;;
    sub) TEST_FILE=${TEST_FILE:-${SEARCH_R1_HOME}/data/search_r1/test_sub.parquet} ;;
    *)
        echo "VAL 只能是 full 或 sub: ${VAL}" >&2
        exit 1
        ;;
esac

MODEL_PATH=${MODEL_PATH:-${SEARCH_R1_HOME}/data/models/Qwen3.5-2B}
TRAIN_FILE=${TRAIN_FILE:-${SEARCH_R1_HOME}/data/search_r1/train.parquet}
TOOL_CONFIG=${TOOL_CONFIG:-${CODE_DIR}/examples/search_r1/config/search_tool_config.yaml}
# 本地盘（容器 overlay，pod 重建即清空）根目录；stage_local.sh 往这里放 venv 和检索数据。
# 默认 /home/work/search_r1_local（不在慢速的 spfs 工作区里）；设成空字符串可禁用。
LOCAL_ROOT=${LOCAL_ROOT-/home/work/search_r1_local}
# 为空 = checkpoint 仍在 $EXP_DIR/checkpoints（旧行为）；设置后放本地盘并在实验目录留符号链接。
CKPT_ROOT=${CKPT_ROOT:-}
# VERL_PYTHON 显式设置时优先；否则 stage_local.sh 打好完成标记（.venv.ok）才用本地 venv。
# 只看 bin/python 不够：它是复制刚开始就出现的符号链接，半截 venv 会被误用。
VERL_PYTHON_SET=${VERL_PYTHON:+1}
VERL_PYTHON=${VERL_PYTHON:-python3}
if [ -z "${VERL_PYTHON_SET}" ] && [ -n "${LOCAL_ROOT}" ]; then
    if [ -f "${LOCAL_ROOT}/.venv.ok" ]; then
        VERL_PYTHON=${LOCAL_ROOT}/.venv/bin/python
    elif [ -e "${LOCAL_ROOT}/.venv" ]; then
        echo "本地 venv 不完整（缺少完成标记 ${LOCAL_ROOT}/.venv.ok）：回落到共享盘的 python"
    fi
fi
if command -v "${VERL_PYTHON}" >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v "${VERL_PYTHON}")
else
    PYTHON_BIN=${VERL_PYTHON}
fi
export SEARCH_R1_HOME MODEL_PATH TRAIN_FILE TEST_FILE TOOL_CONFIG VERL_PYTHON LOCAL_ROOT CKPT_ROOT
# Ray workers inherit this, so each job only sees its own GPUs.
export CUDA_VISIBLE_DEVICES="${GPUS}"

SMOKE=${SMOKE:-0}
RESUME=${RESUME:-0}
BG=${BG:-0}
DRY_RUN=${DRY_RUN:-0}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
EXTRA_OVERRIDES=("$@")  # user overrides, kept for meta.json

########################### experiment ###########################
if [ "${SMOKE}" = 1 ] && [ "${RESUME}" = 1 ]; then
    echo "SMOKE=1 与 RESUME=1 不能同时使用" >&2
    exit 1
fi

if [ "${SMOKE}" = 1 ]; then
    EXP=${EXP:-smoke}
    EXP_DIR=${SEARCH_R1_HOME}/experiments/smoke/${TS}-${EXP}
    RESUME_MODE=disable
else
    if [ -z "${EXP:-}" ]; then
        echo "需要 EXP=<实验名>（或用 SMOKE=1 跑冒烟训练）" >&2
        exit 1
    fi
    EXP_DIR=${SEARCH_R1_HOME}/experiments/${EXP}
    if [ "${RESUME}" = 1 ]; then
        if [ ! -d "${EXP_DIR}" ]; then
            echo "RESUME=1 但实验目录不存在: ${EXP_DIR}" >&2
            exit 1
        fi
        RESUME_MODE=auto
    elif [ "${BG_CHILD:-0}" = 1 ]; then
        RESUME_MODE=disable  # BG parent 已经建好了 EXP_DIR/runs
    elif [ -d "${EXP_DIR}" ]; then
        echo "实验 ${EXP} 已存在：换一个 EXP，或加 RESUME=1 继续" >&2
        exit 1
    else
        RESUME_MODE=disable
    fi
fi

if [[ ! "${EXP}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EXP 只能包含字母、数字、点、下划线和连字符: ${EXP}" >&2
    exit 1
fi

# 实验目录相对 experiments/ 的路径（EXP 本身，或 smoke/<TS>-smoke）；
# CKPT_ROOT 下用同样的相对路径，两个实验的 checkpoint 不会互相覆盖。
EXP_REL=${EXP_DIR#"${SEARCH_R1_HOME}"/experiments/}
if [ "${EXP_REL}" = "${EXP_DIR}" ]; then
    echo "内部错误：EXP_DIR 不在 ${SEARCH_R1_HOME}/experiments/ 下: ${EXP_DIR}" >&2
    exit 1
fi
CKPT_LINK=${EXP_DIR}/checkpoints
if [ -n "${CKPT_ROOT}" ]; then
    CKPT_DIR=${CKPT_ROOT}/${EXP_REL}
    CKPT_LOCAL=1
else
    CKPT_DIR=${CKPT_LINK}
    CKPT_LOCAL=0
fi
CKPT_REAL=${CKPT_DIR}  # setup_ckpt_dir 里解析成物理路径
LOG_REL=runs/${TS}.log
METRICS_REL=runs/${TS}.metrics.jsonl
LOG=${EXP_DIR}/${LOG_REL}
METRICS=${EXP_DIR}/${METRICS_REL}
HYDRA_DIR=${EXP_DIR}/runs/${TS}.hydra
COMMAND_FILE=${EXP_DIR}/runs/${TS}.command.sh
PID_FILE=${EXP_DIR}/runs/${TS}.pid
META_FILE=${EXP_DIR}/meta.json

# SMOKE forces a 2-step run with no validation and no checkpoints.
TOTAL_TRAINING_STEPS=500
SAVE_FREQ=25
TEST_FREQ=50
VAL_BEFORE_TRAIN=True
if [ "${SMOKE}" = 1 ]; then
    TOTAL_TRAINING_STEPS=2
    SAVE_FREQ=-1
    TEST_FREQ=-1
    VAL_BEFORE_TRAIN=False
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # token-level IS guards the residual train/infer mismatch; threshold stays the default 2.0
    algorithm.rollout_correction.rollout_is=token
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=512
    data.max_prompt_length=1024
    data.max_response_length=3072
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=16
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    # rmpad=True gave rollout_corr/kl~0.05 (vs 6e-4) with Qwen3.5 GatedDeltaNet; the run collapsed at ~step 60
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # without rmpad, 24576 peaked at ~219GB
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # chunked/checkpointed entropy keeps the extra logits pass from blowing up memory
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    # GPU0 also hosts the retriever (~32GB for the fp16 FAISS index).
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.max_num_seqs=512
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.max_model_len=4096
    actor_rollout_ref.rollout.enable_prefix_caching=False
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image: 0, video: 0}'
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    # Truncation is by characters (~500 tokens), matching the paper's retrieval budget.
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=3000
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}"
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=8
    # Validation follows the paper: greedy, every 50 steps, full 7-dataset test.
    # Point TEST_FILE at test_sub.parquet for a faster stratified validation set.
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.n=1
)

TRAINER=(
    trainer.project_name=search_r1
    trainer.experiment_name="${EXP}"
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.n_gpus_per_node=${GPU_COUNT}
    trainer.nnodes=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.total_epochs=15
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=4
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.resume_mode=${RESUME_MODE}
    trainer.logger=['console','file']
    trainer.log_val_generations=10
)

# Two jobs on one node each start their own local Ray cluster. A fresh _temp_dir means
# ray.init() finds no other cluster to attach to, and address=local also ignores a stale
# ray_current_cluster left by a crashed run. Keep it on local disk with a short name: Ray
# creates AF_UNIX sockets under <temp_dir>/session_*/sockets (107-byte path limit, and
# network/FUSE mounts often do not support Unix sockets). VERL_ZMQ_JOB_TOKEN keeps the
# colocated weight-sync IPC socket unique: two independent clusters both name their first
# job 01000000, and the token also keeps the socket path short.
RAY_TMP_SHORT=$(printf '%s' "${EXP_DIR}" | cksum | cut -d' ' -f1)
RAY_TEMP_DIR=/tmp/ray-${RAY_TMP_SHORT}
ZMQ_TOKEN="${RAY_TMP_SHORT}-$$"
export VERL_ZMQ_JOB_TOKEN="${ZMQ_TOKEN}"
RAY=(
    +ray_kwargs.ray_init.include_dashboard=False
    +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}"
    +ray_kwargs.ray_init.address=local
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ZMQ_JOB_TOKEN="${ZMQ_TOKEN}"
)

HYDRA=(
    hydra.run.dir="${HYDRA_DIR}"
)

OVERRIDES=(
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${REF[@]}"
    "${ROLLOUT[@]}"
    "${TRAINER[@]}"
    "${RAY[@]}"
    "${HYDRA[@]}"
    "$@"
)

# Effective values after overrides (the last assignment wins), recorded in meta.json so
# the dashboard can compute samples seen (train_batch_size * rollout_n per step).
last_override() {
    local key=$1 item value=""
    for item in "${OVERRIDES[@]}"; do
        if [[ "${item}" == "${key}="* ]]; then
            value=${item#*=}
        fi
    done
    printf '%s' "${value}"
}
TRAIN_BATCH_SIZE=$(last_override data.train_batch_size)
ROLLOUT_N=$(last_override actor_rollout_ref.rollout.n)

########################### derived ###########################
RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-120000}
export VERL_FILE_LOGGER_PATH="${METRICS}"
export RAY_agent_register_timeout_ms

GIT_COMMIT=$(git -C "${CODE_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git -C "${CODE_DIR}" status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY=1
else
    GIT_DIRTY=0
fi
HOSTNAME_VAL=$(hostname)

print_command() {
    printf '%s -m verl.trainer.main_ppo' "${VERL_PYTHON}"
    for override in "${OVERRIDES[@]}"; do
        printf ' \\\n    %q' "${override}"
    done
    printf '\n'
}

########################### dry run ###########################
if [ "${DRY_RUN}" = 1 ]; then
    echo "SEARCH_R1_HOME=${SEARCH_R1_HOME}"
    echo "EXP=${EXP}"
    echo "GPUS=${GPUS}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "GPU_COUNT=${GPU_COUNT}"
    echo "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
    echo "ZMQ_TOKEN=${ZMQ_TOKEN}"
    echo "VAL=${VAL}"
    echo "TEST_FILE=${TEST_FILE}"
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
    echo "ROLLOUT_N=${ROLLOUT_N}"
    echo "EXP_DIR=${EXP_DIR}"
    echo "CKPT_ROOT=${CKPT_ROOT}"
    echo "LOCAL_ROOT=${LOCAL_ROOT}"
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "CKPT_LINK=${CKPT_LINK}"
    echo "CKPT_LOCAL=${CKPT_LOCAL}"
    echo "CKPT_REAL=${CKPT_REAL}"
    echo "VERL_PYTHON=${VERL_PYTHON}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "LOG=${LOG}"
    echo "METRICS=${METRICS}"
    echo "HYDRA_DIR=${HYDRA_DIR}"
    echo "COMMAND_FILE=${COMMAND_FILE}"
    echo "PID_FILE=${PID_FILE}"
    echo "META_FILE=${META_FILE}"
    echo "RESUME_MODE=${RESUME_MODE}"
    echo "---"
    print_command
    exit 0
fi

########################### launch ###########################
mkdir -p "${EXP_DIR}/runs"

# CKPT_ROOT 设置时：实验目录下留一个 checkpoints -> 本地盘目录 的符号链接，
# RESUME 和人肉找 checkpoint 都走这个固定路径。旧符号链接直接替换；真实目录只有在
# 空的情况下才替换，非空目录拒绝（避免把已有 checkpoint 藏起来）。
setup_ckpt_dir() {
    if [ -z "${CKPT_ROOT}" ]; then
        # 之前用 CKPT_ROOT 跑过、pod 重建后本地盘被清空时，checkpoints 会变成失效的
        # 符号链接；mkdir -p 只会报一句含糊的 "File exists"。这里不自动删除，交给用户决定。
        if [ -L "${CKPT_DIR}" ] && [ ! -e "${CKPT_DIR}" ]; then
            echo "拒绝：${CKPT_DIR} 是失效的符号链接 -> $(readlink "${CKPT_DIR}")" >&2
            echo "      链接目标已不存在（pod 重建会清空本地盘）。两种做法：" >&2
            echo "      1) 设置 CKPT_ROOT 重跑：会从头开始训练，原来的 checkpoint 已无法续训；" >&2
            echo "      2) 手动删除这个链接后重试：rm '${CKPT_DIR}'" >&2
            return 1
        fi
        mkdir -p "${CKPT_DIR}"
        CKPT_REAL=$(cd "${CKPT_DIR}" && pwd -P)
        return 0
    fi
    mkdir -p "${CKPT_DIR}"
    if [ -L "${CKPT_LINK}" ]; then
        rm -f "${CKPT_LINK}"
    elif [ -e "${CKPT_LINK}" ]; then
        if [ -d "${CKPT_LINK}" ] && [ -z "$(ls -A "${CKPT_LINK}" 2>/dev/null)" ]; then
            rmdir "${CKPT_LINK}"
        else
            echo "拒绝：${CKPT_LINK} 已存在，且不是空目录或符号链接。" >&2
            echo "checkpoint 要写到 ${CKPT_DIR}；请先手动挪走旧的 checkpoints 目录再重试。" >&2
            return 1
        fi
    fi
    ln -s "${CKPT_DIR}" "${CKPT_LINK}"
    CKPT_REAL=$(cd "${CKPT_DIR}" && pwd -P)
    if [ "${RESUME_MODE}" = auto ] && [ -z "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
        echo "警告：RESUME=1，但本地 checkpoint 目录是空的：${CKPT_DIR}" >&2
        echo "      pod 重建后本地盘会被清空；这种情况无法续训，trainer 会从第 0 步开始。" >&2
    fi
}

# BG 父进程和前台都先做一次：路径/权限/旧目录的问题立刻报错，而不是等到子进程的 log 里。
setup_ckpt_dir

if [ "${BG}" = 1 ]; then
    # setsid makes the child a process-group leader, so the pid file stops this job
    # (python + its local Ray cluster) without touching the other job on the node.
    TS="${TS}" BG=0 BG_CHILD=1 EXP="${EXP}" SMOKE="${SMOKE}" RESUME="${RESUME}" \
        GPUS="${GPUS}" VAL="${VAL}" \
        setsid nohup bash "${SCRIPT_FILE}" "$@" >> "${LOG}" 2>&1 &
    pid=$!
    echo "${pid}" > "${PID_FILE}"
    echo "EXP dir : ${EXP_DIR}"
    echo "ray tmp : ${RAY_TEMP_DIR}    # Ray session/logs live here (local disk)"
    echo "ckpt    : ${CKPT_DIR}"
    if [ "${CKPT_LOCAL}" = 1 ]; then
        echo "ckpt link: ${CKPT_LINK} -> ${CKPT_DIR}    # 本地盘，pod 重建会丢"
    fi
    echo "python  : ${PYTHON_BIN}"
    echo "log     : ${LOG}"
    echo "metrics : ${METRICS}"
    echo "pid     : ${pid} (${PID_FILE})"
    echo "tail -f ${LOG}"
    echo "stop    : kill -- -$(cat "${PID_FILE}")    # 只杀本实验的进程组，别用 ray stop --force"
    exit 0
fi

write_command() {
    {
        printf '#!/usr/bin/env bash\n'
        printf '# Generated by %s at %s\n' "${SCRIPT_FILE}" "${TS}"
        printf 'cd %q\n' "${CODE_DIR}"
        printf 'export SEARCH_R1_HOME=%q\n' "${SEARCH_R1_HOME}"
        printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
        printf 'export GPUS=%q\n' "${GPUS}"
        printf 'export VAL=%q\n' "${VAL}"
        printf 'export EXP=%q\n' "${EXP}"
        printf 'export SMOKE=%q\n' "${SMOKE}"
        printf 'export RESUME=%q\n' "${RESUME}"
        printf 'export CKPT_ROOT=%q\n' "${CKPT_ROOT}"
        printf 'export LOCAL_ROOT=%q\n' "${LOCAL_ROOT}"
        printf 'export VERL_PYTHON=%q\n' "${VERL_PYTHON}"
        printf 'export VERL_FILE_LOGGER_PATH=%q\n' "${METRICS}"
        printf 'export VERL_ZMQ_JOB_TOKEN=%q\n' "${ZMQ_TOKEN}"
        printf 'export RAY_agent_register_timeout_ms=%q\n' "${RAY_agent_register_timeout_ms}"
        printf '\n'
        printf '%s -m verl.trainer.main_ppo' "${VERL_PYTHON}"
        for override in "${OVERRIDES[@]}"; do
            printf ' \\\n    %q' "${override}"
        done
        printf '\n'
    } > "${COMMAND_FILE}"
    chmod +x "${COMMAND_FILE}"
}

update_meta() {
    local status=$1
    shift
    META_FILE="${META_FILE}" \
    META_NAME="${EXP}" \
    META_STATUS="${status}" \
    META_SMOKE="${SMOKE}" \
    META_GIT_COMMIT="${GIT_COMMIT}" \
    META_GIT_DIRTY="${GIT_DIRTY}" \
    META_HOST="${HOSTNAME_VAL}" \
    META_MODEL_PATH="${MODEL_PATH}" \
    META_TS="${TS}" \
    META_RESUME="${RESUME_MODE}" \
    META_LOG="${LOG_REL}" \
    META_METRICS="${METRICS_REL}" \
    META_CHECKPOINT_DIR="${CKPT_REAL}" \
    META_CHECKPOINT_LOCAL="${CKPT_LOCAL}" \
    META_PYTHON="${PYTHON_BIN}" \
    META_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    META_ROLLOUT_N="${ROLLOUT_N}" \
    python3 - "$@" <<'PY'
import datetime
import json
import os
import sys
import tempfile

path = os.environ["META_FILE"]
now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
else:
    meta = {"name": os.environ["META_NAME"], "created": now, "notes": ""}

meta["status"] = os.environ["META_STATUS"]
meta["smoke"] = os.environ["META_SMOKE"] == "1"
meta["git_commit"] = os.environ["META_GIT_COMMIT"]
meta["git_dirty"] = os.environ["META_GIT_DIRTY"] == "1"
meta["host"] = os.environ["META_HOST"]
meta["model_path"] = os.environ["META_MODEL_PATH"]
meta["checkpoint_dir"] = os.environ["META_CHECKPOINT_DIR"]
meta["checkpoint_local"] = os.environ["META_CHECKPOINT_LOCAL"] == "1"
meta["python"] = os.environ["META_PYTHON"]
meta["train_batch_size"] = int(os.environ["META_TRAIN_BATCH_SIZE"])
meta["rollout_n"] = int(os.environ["META_ROLLOUT_N"])
meta["overrides"] = sys.argv[1:]
meta.setdefault("notes", "")

launch = {
    "ts": os.environ["META_TS"],
    "resume": os.environ["META_RESUME"] == "auto",
    "log": os.environ["META_LOG"],
    "metrics": os.environ["META_METRICS"],
}
launches = meta.setdefault("launches", [])
for index, item in enumerate(launches):
    if item.get("ts") == launch["ts"]:
        launches[index] = launch
        break
else:
    launches.append(launch)

# 原子写：SIGTERM 落在 json.dump 中间时，meta.json 不能变成 0 字节或半截 JSON。
# 同目录临时文件 + fsync + os.replace；读 meta 的一方要么看到旧文件，要么看到新文件。
directory = os.path.dirname(path) or "."
# 保留原文件权限（首次创建时用 0644，和普通 open() 一致）；mkstemp 默认是 0600。
mode = (os.stat(path).st_mode & 0o777) if os.path.exists(path) else 0o644
fd, tmp_path = tempfile.mkstemp(prefix=".meta.", suffix=".tmp", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
PY
}

CHILD_PID=""
CHILD_STATUS=""
SIGNAL_NUMBER=""

finish() {
    local code=$?
    if [ -n "${CHILD_STATUS}" ]; then
        code=${CHILD_STATUS}
    fi
    if [ -n "${SIGNAL_NUMBER}" ]; then
        update_meta "failed (signal ${SIGNAL_NUMBER})" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
    elif [ "${code}" -eq 0 ]; then
        update_meta finished ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
    else
        update_meta "failed (exit ${code})" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
    fi
}

# SIGTERM/SIGINT/SIGHUP：转发给 python 子进程，并记住信号。EXIT trap 据此写
# failed (signal N)；否则被杀掉的 job 会因为最后一条命令返回 0 而被写成 finished。
forward_signal() {
    local signo=$1 signame=$2
    if [ -z "${SIGNAL_NUMBER}" ]; then
        SIGNAL_NUMBER=${signo}
        echo "收到 SIG${signame}：转发给子进程，状态记为 failed (signal ${signo})" >&2
    fi
    if [ -n "${CHILD_PID}" ] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        kill -s "${signame}" "${CHILD_PID}" 2>/dev/null || true
    fi
}

trap finish EXIT
trap 'forward_signal 15 TERM' TERM
trap 'forward_signal 2 INT' INT
trap 'forward_signal 1 HUP' HUP

write_command
update_meta running ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}

if [ -n "${SIGNAL_NUMBER}" ]; then
    exit 143  # 启动前就收到信号；EXIT trap 会记 failed (signal N)
fi

# python 放后台再 wait：这样才能拿到它自己的退出码（原来 `python | tee` 让 EXIT trap
# 看到的是 tee 的 0，被判成 finished）。前台模式仍用 process substitution 把输出同时
# 写到终端和 log，$! 仍是 python 的 pid。
run_child() {
    local status=0 waited=0 killed=0
    if [ "${BG_CHILD:-0}" = 1 ]; then
        # setsid makes this process the group leader; the pid file stops the whole job.
        echo "$$" > "${PID_FILE}"
        # Detached child: the BG parent already redirects stdout/stderr to $LOG.
        "${VERL_PYTHON}" -m verl.trainer.main_ppo "${OVERRIDES[@]}" &
    else
        "${VERL_PYTHON}" -m verl.trainer.main_ppo "${OVERRIDES[@]}" > >(tee -a "${LOG}") 2>&1 &
    fi
    CHILD_PID=$!

    while :; do
        status=0
        wait "${CHILD_PID}" || status=$?
        if [ -z "${SIGNAL_NUMBER}" ]; then
            break  # 没有信号：wait 只在子进程退出时返回
        fi
        # 收到信号时 wait 可能只是被打断（子进程还活着）。轮询到它消失；
        # 15s 不退就 SIGKILL，最多再等 5s 收尸。
        waited=0
        killed=0
        while kill -0 "${CHILD_PID}" 2>/dev/null; do
            waited=$((waited + 1))
            if [ "${killed}" = 0 ] && [ "${waited}" -ge 15 ]; then
                echo "子进程 15s 内没有退出，发送 SIGKILL" >&2
                kill -KILL "${CHILD_PID}" 2>/dev/null || true
                killed=1
            fi
            if [ "${waited}" -ge 20 ]; then
                break
            fi
            sleep 1
        done
        wait "${CHILD_PID}" 2>/dev/null || status=$?  # 收尸；状态已被信号接管
        break
    done

    CHILD_STATUS=${status}
    # 子进程被信号直接杀掉（外部 kill 了 python）时，wait 返回 128+N。
    if [ -z "${SIGNAL_NUMBER}" ] && [ "${status}" -gt 128 ] && [ "${status}" -lt 160 ]; then
        SIGNAL_NUMBER=$((status - 128))
    fi
}

run_child
exit "${CHILD_STATUS}"
