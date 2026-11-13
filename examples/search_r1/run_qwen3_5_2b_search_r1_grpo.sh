#!/usr/bin/env bash
# GRPO | Qwen3.5-2B | FSDP2 | Search-R1 with a local E5 + FAISS retriever
# This file lives in <repo>/examples/search_r1/, so repo root = ../.. and
# SEARCH_R1_HOME = ../../.. (the workspace holding verl/, data/, experiments/, logs/).
# Requires the venv (source "$SEARCH_R1_HOME/.venv/bin/activate"); never use `uv run`,
# which would replace the environment's faiss-gpu without it being in uv.lock.
#
#   EXP=<name> [BG=1] [RESUME=1] bash examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
#   SMOKE=1 bash examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
#   [DRY_RUN=1] ...            print resolved paths and the command, create nothing

set -xeuo pipefail

########################### workspace ###########################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_FILE=${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${REPO_ROOT}/.." && pwd)}

MODEL_PATH=${MODEL_PATH:-${SEARCH_R1_HOME}/data/models/Qwen3.5-2B}
TRAIN_FILE=${TRAIN_FILE:-${SEARCH_R1_HOME}/data/search_r1/train.parquet}
TEST_FILE=${TEST_FILE:-${SEARCH_R1_HOME}/data/search_r1/test.parquet}
TOOL_CONFIG=${TOOL_CONFIG:-${REPO_ROOT}/examples/search_r1/config/search_tool_config.yaml}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
VERL_PYTHON=${VERL_PYTHON:-python3}
export SEARCH_R1_HOME MODEL_PATH TRAIN_FILE TEST_FILE TOOL_CONFIG NDEVICES_PER_NODE VERL_PYTHON

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

CKPT_DIR=${EXP_DIR}/checkpoints
LOG=${EXP_DIR}/runs/${TS}.log
METRICS=${EXP_DIR}/runs/${TS}.metrics.jsonl
HYDRA_DIR=${EXP_DIR}/runs/${TS}.hydra
COMMAND_FILE=${EXP_DIR}/runs/${TS}.command.sh
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
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
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

RAY=(
    +ray_kwargs.ray_init.include_dashboard=False
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

########################### derived ###########################
RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-120000}
export VERL_FILE_LOGGER_PATH="${METRICS}"
export RAY_agent_register_timeout_ms

GIT_COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]; then
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
    echo "EXP_DIR=${EXP_DIR}"
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "LOG=${LOG}"
    echo "METRICS=${METRICS}"
    echo "HYDRA_DIR=${HYDRA_DIR}"
    echo "COMMAND_FILE=${COMMAND_FILE}"
    echo "META_FILE=${META_FILE}"
    echo "RESUME_MODE=${RESUME_MODE}"
    echo "---"
    print_command
    exit 0
fi

########################### launch ###########################
mkdir -p "${EXP_DIR}/runs"

if [ "${BG}" = 1 ]; then
    TS="${TS}" BG=0 BG_CHILD=1 EXP="${EXP}" SMOKE="${SMOKE}" RESUME="${RESUME}" \
        nohup bash "${SCRIPT_FILE}" "$@" >> "${LOG}" 2>&1 &
    pid=$!
    echo "EXP dir : ${EXP_DIR}"
    echo "log     : ${LOG}"
    echo "metrics : ${METRICS}"
    echo "pid     : ${pid}"
    echo "tail -f ${LOG}"
    exit 0
fi

write_command() {
    {
        printf '#!/usr/bin/env bash\n'
        printf '# Generated by %s at %s\n' "${SCRIPT_FILE}" "${TS}"
        printf 'cd %q\n' "${REPO_ROOT}"
        printf 'export SEARCH_R1_HOME=%q\n' "${SEARCH_R1_HOME}"
        printf 'export EXP=%q\n' "${EXP}"
        printf 'export SMOKE=%q\n' "${SMOKE}"
        printf 'export RESUME=%q\n' "${RESUME}"
        printf 'export VERL_FILE_LOGGER_PATH=%q\n' "${METRICS}"
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
    META_LOG="${LOG}" \
    META_METRICS="${METRICS}" \
    python3 - "$@" <<'PY'
import datetime
import json
import os
import sys

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

with open(path, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

finish() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        update_meta finished ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
    else
        update_meta "failed (exit ${code})" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
    fi
}
trap finish EXIT

write_command
update_meta running ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}

if [ "${BG_CHILD:-0}" = 1 ]; then
    # Detached child: the BG parent already redirects stdout/stderr to $LOG.
    "${VERL_PYTHON}" -m verl.trainer.main_ppo "${OVERRIDES[@]}"
else
    "${VERL_PYTHON}" -m verl.trainer.main_ppo "${OVERRIDES[@]}" 2>&1 | tee -a "${LOG}"
fi
