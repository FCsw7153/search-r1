#!/usr/bin/env bash
# 单卡 pod（全本地盘）的 Qwen3 base 模型 Search-R1 训练入口。
#
#   bash verl/examples/search_r1/run_qwen3_base_local.sh <EXP> <GPU> [hydra overrides...]
#
# 例：
#   bash verl/examples/search_r1/run_qwen3_base_local.sh qwen3-8b-klcov 0 \
#       actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov
#
# 约定（每个都来自一次丢过的实验）：
#   * 代码在本地 git clone $SEARCH_R1_HOME（默认 /home/work/sr1），venv/data 在
#     $LOCAL_ROOT（默认 /home/work/search_r1_local），rollout 在 $ROLLOUT_ROOT/<EXP>。
#     共享盘 /home/work/mshop-agent-1 不写（FUSE 挂死会拖住所有 job）。
#   * vLLM 默认 FlashInfer attention 在 B300 + Qwen3 上会 GPU kernel spin 卡死，
#     固定用 FLASH_ATTN。
#   * 前台运行：watchdog 用 py-spy 时不要 sudo（root 没有 CAP_SYS_PTRACE）。
#   * 没有 checkpoint（save_freq=-1）：非信号失败最多重试 2 次，每次从第 0 步开始。
#   * 可写盘超限会被平台驱逐；后台 watchdog 每 5 分钟量化我们的占用，超限就 SIGTERM
#     本实验的进程组，让训练干净停止（并写 reason 文件），而不是整个 pod 被关掉。
#
# 可调环境变量：
#   MODEL_NAME     data/models 下的目录名（默认 Qwen3-8B-Base）
#   TRAIN_FILE     训练 parquet（默认 $LOCAL_ROOT/data/search_r1/train.parquet）
#   TEST_FILE      验证 parquet（默认 $LOCAL_ROOT/data/search_r1/test_sub.parquet）
#   STEPS          总步数（默认 300）
#   TEST_FREQ      验证间隔（默认 50；-1 关闭）
#   GPU_MEM        vLLM gpu_memory_utilization（默认 0.6）
#   REF_OFFLOAD   1 -> actor_rollout_ref.ref.fsdp_config.param_offload=True（默认 0）
#   OPT_OFFLOAD   1 -> actor_rollout_ref.actor.fsdp_config.optimizer_offload=True（默认 0）
#   DISK_LIMIT_GB  可写盘上限（默认 180）
#   RESUME         首次启动是否 RESUME=1（默认 0；实验目录已存在时必须先手动设 1）
#   MAX_RETRIES    非信号失败的重试次数（默认 2）
#   DRY_RUN=1      只打印解析后的配置和 override 列表，不创建目录、不启动
#
# 日志：终端 + $SEARCH_R1_HOME/experiments/<EXP>/runs/<TS>.log（launcher 写）；
#       watchdog 的动静另记在 $ROLLOUT_ROOT/<EXP>/watchdog.log。
set -euo pipefail

SEARCH_R1_HOME=${SEARCH_R1_HOME:-/home/work/sr1}
LOCAL_ROOT=${LOCAL_ROOT:-/home/work/search_r1_local}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-/home/work/rollouts}

MODEL_NAME=${MODEL_NAME:-Qwen3-8B-Base}
STEPS=${STEPS:-300}
TEST_FREQ=${TEST_FREQ:-50}
GPU_MEM=${GPU_MEM:-0.6}
REF_OFFLOAD=${REF_OFFLOAD:-0}
OPT_OFFLOAD=${OPT_OFFLOAD:-0}
DISK_LIMIT_GB=${DISK_LIMIT_GB:-180}
DISK_CHECK_SECS=${DISK_CHECK_SECS:-300}
HANG_STALE_SECS=${HANG_STALE_SECS:-900}
HANG_DUMP_COOLDOWN_SECS=${HANG_DUMP_COOLDOWN_SECS:-1800}
MAX_RETRIES=${MAX_RETRIES:-2}
RETRY_DELAY_SECS=${RETRY_DELAY_SECS:-60}
RESUME=${RESUME:-0}
DRY_RUN=${DRY_RUN:-0}

MODEL_PATH=${MODEL_PATH:-${LOCAL_ROOT}/data/models/${MODEL_NAME}}
TRAIN_FILE=${TRAIN_FILE:-${LOCAL_ROOT}/data/search_r1/train.parquet}
TEST_FILE=${TEST_FILE:-${LOCAL_ROOT}/data/search_r1/test_sub.parquet}
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}

# 我们自己的可写占用：/home/work 上除去共享 spfs 挂载，加上 /tmp。
# -x 不跨文件系统（共享盘是单独的 FUSE 挂载，会被剪掉）；GNU du 再显式 exclude 一次。
DISK_WATCH_PATHS=(/home/work /tmp)
if du --version >/dev/null 2>&1; then
    DU_CMD=(du -s -x -k --exclude=/home/work/mshop-agent-1)
else
    DU_CMD=(du -s -x -k)
fi

WATCHDOG_PIDS=()
RUN_PGID=""
WRAPPER_SIGNAL=""
EXP=""
R=""

usage() {
    printf 'usage: %s <EXP> <GPU> [hydra overrides...]\n' "${BASH_SOURCE[0]}" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# 1/true/yes/on -> True，其余 -> False（Hydra 的布尔字段要大写）。
bool_flag() {
    case "$1" in
        1 | true | True | TRUE | yes | on) printf 'True' ;;
        *) printf 'False' ;;
    esac
}

# 输出若干目录的总 KiB；不存在的目录按 0 算。可单独测试：du_usage_kb <dir>...
du_usage_kb() {
    local dir kb out total=0
    for dir in "$@"; do
        [ -d "${dir}" ] || continue
        out=$("${DU_CMD[@]}" "${dir}" 2>/dev/null || true)
        kb=$(printf '%s\n' "${out}" | awk 'NF {print $1; exit}')
        [ -n "${kb}" ] || kb=0
        total=$((total + kb))
    done
    printf '%s\n' "${total}"
}

log_msg() {
    local line
    line="$(date -u +%Y-%m-%dT%H:%M:%SZ) [${EXP}] $*"
    printf '%s\n' "${line}"
    if [ -n "${R}" ] && [ -d "${R}" ]; then
        # 盘满时日志自己失败也不能让 watchdog 退出（否则就没人停训练了）。
        printf '%s\n' "${line}" >>"${R}/watchdog.log" 2>/dev/null || true
    fi
}

############### watchdog：metrics 停更 >15 分钟就抓栈 ###############
hang_watchdog() {
    local f now age out pid cmd pyspy last_dump=0 p
    pyspy=${PYSPY:-${LOCAL_ROOT}/.venv/bin/py-spy}
    while sleep 60; do
        f=$(ls -t "${EXP_DIR}"/runs/*.metrics.jsonl 2>/dev/null | head -n 1 || true)
        [ -n "${f}" ] || continue
        now=$(date +%s)
        age=$((now - $(date -r "${f}" +%s 2>/dev/null || printf '%s' "${now}")))
        [ "${age}" -gt "${HANG_STALE_SECS}" ] || continue
        [ $((now - last_dump)) -ge "${HANG_DUMP_COOLDOWN_SECS}" ] || continue

        out="${R}/hang/$(date -u +%Y%m%d_%H%M%S).txt"
        {
            printf 'search_r1 hang dump at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'metrics file: %s (stale %ss)\n' "${f}" "${age}"
            printf 'EXP=%s GPU=%s SEARCH_R1_HOME=%s\n' "${EXP}" "${GPU}" "${SEARCH_R1_HOME}"
            printf '===== nvidia-smi\n'
            nvidia-smi 2>&1 || true
            # 只抓这个实验的卡上的进程：/proc/<pid>/environ 里有本次的 CUDA_VISIBLE_DEVICES。
            # 不用 sudo：root 反而没有 CAP_SYS_PTRACE，同 uid 的 py-spy 才能 attach。
            for p in /proc/[0-9]*; do
                pid=${p#/proc/}
                [ -r "${p}/environ" ] || continue
                tr '\0' '\n' <"${p}/environ" 2>/dev/null \
                    | grep -qx "CUDA_VISIBLE_DEVICES=${GPU}" || continue
                cmd=$(tr '\0' ' ' <"${p}/cmdline" 2>/dev/null || true)
                case "${cmd}" in
                    *python* | *Python* | *ray* | *Ray* | *vllm* | *VLLM* | *EngineCore* | *main_ppo*) ;;
                    *) continue ;;
                esac
                printf '===== pid %s: %.200s\n' "${pid}" "${cmd}"
                if [ -x "${pyspy}" ]; then
                    "${pyspy}" dump --pid "${pid}" 2>&1 | head -n 120
                else
                    printf 'py-spy 不可执行: %s\n' "${pyspy}"
                fi
            done
        } >"${out}" 2>&1
        log_msg "hang: metrics 已停更 ${age}s，栈写入 ${out}"
        last_dump=${now}
    done
}

############### watchdog：可写盘超限就停掉本实验 ###############
# 返回 0 = 未超限；1 = 已对训练进程组发 SIGTERM；2 = 超限但当前没有在跑的训练。
check_disk_once() {
    local used_kb used_gb pgid reason tmp
    used_kb=$(du_usage_kb "${DISK_WATCH_PATHS[@]}")
    used_gb=$((used_kb / 1024 / 1024))
    log_msg "disk: 本实验可写占用 ${used_gb} GiB（上限 ${DISK_LIMIT_GB} GiB；du -sx ${DISK_WATCH_PATHS[*]}）"
    [ "${used_gb}" -gt "${DISK_LIMIT_GB}" ] || return 0

    pgid=$(tr -d '[:space:]' <"${R}/run.pgid" 2>/dev/null || true)
    if [ -z "${pgid}" ]; then
        log_msg "disk 超限但没有在跑的训练进程（启动前检查会拦住下一次启动）"
        return 2
    fi

    reason="${R}/DISK_LIMIT_EXCEEDED.txt"
    tmp="${reason}.tmp.$$"
    # 盘满时写 reason 文件本身也可能失败；无论如何都要把 SIGTERM 发出去。
    if {
        printf 'writable usage %s GiB > DISK_LIMIT_GB=%s at %s\n' \
            "${used_gb}" "${DISK_LIMIT_GB}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'measured with: %s %s (shared spfs mount excluded)\n' \
            "${DU_CMD[*]}" "${DISK_WATCH_PATHS[*]}"
        printf 'raw usage: %s KiB\n' "${used_kb}"
        printf 'run process group: %s\n' "${pgid}"
        printf 'action: SIGTERM to the process group so the pod is not evicted\n'
        printf '清理后换一个 EXP（或删掉本文件）再继续。\n'
    } >"${tmp}" 2>/dev/null; then
        mv -f "${tmp}" "${reason}" 2>/dev/null || rm -f "${tmp}" || true # 原子替换
    else
        rm -f "${tmp}" 2>/dev/null || true
        log_msg "disk 超限但 reason 文件写不出来（盘满？），仍然发 SIGTERM"
    fi
    log_msg "disk 超限：reason=${reason}，向进程组 ${pgid} 发 SIGTERM"
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    return 1
}

disk_watchdog() {
    local rc
    while :; do
        rc=0
        check_disk_once || rc=$?
        # 1 = 已经发过 SIGTERM，停止本 watchdog；2 = 超限但没在跑，继续盯着。
        [ "${rc}" -eq 1 ] && return 0
        sleep "${DISK_CHECK_SECS}"
    done
}

start_watchdogs() {
    # watchdog 的 stdout/stderr 不能继承调用者的管道：清理时被 kill 的 subshell 可能留下
    # sleep/du 子进程，它们会继续持有管道，让 `... | grep` 之类一直等不到 EOF。
    hang_watchdog >>"${R}/watchdog.stdout" 2>&1 &
    WATCHDOG_PIDS+=("$!")
    disk_watchdog >>"${R}/watchdog.stdout" 2>&1 &
    WATCHDOG_PIDS+=("$!")
}

cleanup() {
    local pid
    for pid in ${WATCHDOG_PIDS[@]+"${WATCHDOG_PIDS[@]}"}; do
        pkill -TERM -P "${pid}" 2>/dev/null || true # 正在跑的 sleep/du
        kill "${pid}" 2>/dev/null || true
    done
    for pid in ${WATCHDOG_PIDS[@]+"${WATCHDOG_PIDS[@]}"}; do
        wait "${pid}" 2>/dev/null || true
    done
}

on_signal() {
    local signo=$1 signame=$2
    WRAPPER_SIGNAL=${signo}
    log_msg "收到 SIG${signame}，转发给训练进程组 ${RUN_PGID:-（尚未启动）}"
    if [ -n "${RUN_PGID}" ]; then
        kill -s "${signame}" -- "-${RUN_PGID}" 2>/dev/null || true
    fi
}

# wait 会被 trap 打断（子进程还活着时也返回 >128）；轮询到真正退出为止。
wait_for_pid() {
    local pid=$1 code=0
    while :; do
        code=0
        wait "${pid}" || code=$?
        if kill -0 "${pid}" 2>/dev/null; then
            continue
        fi
        return "${code}"
    done
}

############### 一次启动：独立进程组，方便磁盘 watchdog 只杀训练 ###############
run_attempt() {
    local resume=$1 pid code=0 actual_pgid
    RUN_PGID=""
    if command -v setsid >/dev/null 2>&1; then
        BG=0 RESUME="${resume}" EXP="${EXP}" GPUS="${GPU}" \
            setsid bash "${LAUNCHER}" ${OVERRIDES[@]+"${OVERRIDES[@]}"} &
    else
        # 没有 setsid 时打开 job control，让后台 job 自己成为进程组组长。
        set -m
        BG=0 RESUME="${resume}" EXP="${EXP}" GPUS="${GPU}" \
            bash "${LAUNCHER}" ${OVERRIDES[@]+"${OVERRIDES[@]}"} &
        set +m
    fi
    pid=$!
    # 先记下 pid（setsid 下 PGID == pid），避免刚启动就收到信号时转发不到。
    RUN_PGID=${pid}

    # 等 setsid() 生效后再确认进程组；正常情况下 PGID == pid。
    sleep 1
    actual_pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)
    RUN_PGID=${actual_pgid:-${pid}}
    printf '%s\n' "${RUN_PGID}" >"${R}/run.pgid"
    log_msg "训练进程组 ${RUN_PGID}（launcher pid ${pid}）"

    wait_for_pid "${pid}" || code=$?
    RUN_PGID=""
    : >"${R}/run.pgid"
    return "${code}"
}

main() {
    if [ "$#" -lt 2 ]; then
        usage
        exit 2
    fi
    EXP=$1
    GPU=$2
    shift 2

    [[ "${EXP}" =~ ^[A-Za-z0-9._-]+$ ]] || die "EXP 只能包含字母、数字、点、下划线和连字符: ${EXP}"
    [[ "${GPU}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPU 必须是逗号分隔的设备号，例如 0 或 0,1: ${GPU}"

    R="${ROLLOUT_ROOT}/${EXP}"
    EXP_DIR="${SEARCH_R1_HOME}/experiments/${EXP}"
    LAUNCHER="${SEARCH_R1_HOME}/verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh"

    export SEARCH_R1_HOME LOCAL_ROOT MODEL_PATH TRAIN_FILE TEST_FILE
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
    export GPUS="${GPU}"

    # SwanLab 只在 ~/.swanlab_key 存在时打开（key 不打印）；metrics.jsonl 始终保留。
    LOGGERS="['console','file']"
    swanlab=0
    if [ -s "${HOME}/.swanlab_key" ]; then
        export SWANLAB_API_KEY="$(cat "${HOME}/.swanlab_key")"
        export SWANLAB_MODE=cloud
        export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/home/work/swanlog}
        LOGGERS="['console','file','swanlab']"
        swanlab=1
    fi

    local ref_offload opt_offload
    ref_offload=$(bool_flag "${REF_OFFLOAD}")
    opt_offload=$(bool_flag "${OPT_OFFLOAD}")

    OVERRIDES=(
        # base 模型的 tag 协议；最后一轮不再检索
        "actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent"
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5"
        # 本 loop 不套 chat template，关掉 tool_agent 的 chat-template sanity check
        "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable"
        "algorithm.rollout_correction.rollout_is=null"
        "actor_rollout_ref.model.path=${MODEL_PATH}"
        "actor_rollout_ref.model.use_remove_padding=True"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576"
        "actor_rollout_ref.actor.ppo_mini_batch_size=64"
        "data.train_batch_size=128"
        "actor_rollout_ref.rollout.enable_prefix_caching=False"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM}"
        # B300 + Qwen3 上 FlashInfer 会 kernel spin 卡死；必须显式 FLASH_ATTN
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN"
        "actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload}"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${opt_offload}"
        "trainer.total_training_steps=${STEPS}"
        "trainer.test_freq=${TEST_FREQ}"
        "trainer.save_freq=-1"
        "trainer.rollout_data_dir=${R}/train"
        "trainer.validation_data_dir=${R}/val"
        "trainer.logger=${LOGGERS}"
        "trainer.project_name=search_r1_qwen3"
    )
    OVERRIDES+=("$@")

    if [ "${DRY_RUN}" = 1 ]; then
        printf 'SEARCH_R1_HOME=%s\n' "${SEARCH_R1_HOME}"
        printf 'LOCAL_ROOT=%s\n' "${LOCAL_ROOT}"
        printf 'EXP=%s\n' "${EXP}"
        printf 'GPU=%s\n' "${GPU}"
        printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
        printf 'TRAIN_FILE=%s\n' "${TRAIN_FILE}"
        printf 'TEST_FILE=%s\n' "${TEST_FILE}"
        printf 'ROLLOUT_DIR=%s\n' "${R}"
        printf 'STEPS=%s TEST_FREQ=%s GPU_MEM=%s\n' "${STEPS}" "${TEST_FREQ}" "${GPU_MEM}"
        printf 'REF_OFFLOAD=%s (%s) OPT_OFFLOAD=%s (%s)\n' \
            "${REF_OFFLOAD}" "${ref_offload}" "${OPT_OFFLOAD}" "${opt_offload}"
        printf 'DISK_LIMIT_GB=%s DISK_CHECK_SECS=%s\n' "${DISK_LIMIT_GB}" "${DISK_CHECK_SECS}"
        printf 'HANG_STALE_SECS=%s MAX_RETRIES=%s\n' "${HANG_STALE_SECS}" "${MAX_RETRIES}"
        printf 'LOGGERS=%s SWANLAB=%s\n' "${LOGGERS}" "${swanlab}"
        printf '%s\n' '--- overrides（后面的覆盖前面的，用户参数在最后）：'
        printf '  %s\n' "${OVERRIDES[@]}"
        printf '%s\n' '--- launcher：'
        printf '  EXP=%q GPUS=%q bash %q ...\n' "${EXP}" "${GPU}" "${LAUNCHER}"
        exit 0
    fi

    [ -f "${LAUNCHER}" ] || die "找不到 launcher：${LAUNCHER}"
    [ -f "${TRAIN_FILE}" ] || die "训练数据不存在：${TRAIN_FILE}（先在 pod 上跑 pod_setup_local.sh）"
    if [ "${TEST_FREQ}" != "-1" ]; then
        [ -f "${TEST_FILE}" ] || die "验证数据不存在：${TEST_FILE}"
    fi
    if [ -d "${EXP_DIR}" ] && [ "${RESUME}" != 1 ]; then
        die "实验目录已存在：${EXP_DIR}。换一个 EXP，或用 RESUME=1 继续"
    fi
    [ -f "${MODEL_PATH}/config.json" ] \
        || printf 'WARNING: %s 没有 config.json（模型还没下载？先跑 pod_setup_local.sh）\n' "${MODEL_PATH}" >&2

    mkdir -p "${R}/train" "${R}/val" "${R}/hang"
    : >"${R}/run.pgid"
    # 上一次（同一个 EXP）的磁盘超限记录不能影响本次的重试判断。
    if [ -f "${R}/DISK_LIMIT_EXCEEDED.txt" ]; then
        mv -f "${R}/DISK_LIMIT_EXCEEDED.txt" "${R}/DISK_LIMIT_EXCEEDED.previous.txt" 2>/dev/null || true
    fi

    trap cleanup EXIT
    trap 'on_signal 15 TERM' TERM
    trap 'on_signal 2 INT' INT
    trap 'on_signal 1 HUP' HUP
    start_watchdogs

    log_msg "watchdog: hang(>${HANG_STALE_SECS}s 无 metrics → py-spy), disk(每 ${DISK_CHECK_SECS}s, 上限 ${DISK_LIMIT_GB} GiB)"
    log_msg "swanlab=$([ "${swanlab}" = 1 ] && printf enabled || printf disabled)；rollouts=${R}"

    local attempt=0 resume code final=1
    while :; do
        # 还没启动就收到信号：不要再启动新的训练。
        if [ -n "${WRAPPER_SIGNAL}" ]; then
            final=$((128 + WRAPPER_SIGNAL))
            log_msg "wrapper 收到信号，退出码 ${final}"
            break
        fi
        # 启动前同步查一次盘：重试的 60s 间隔里后台 watchdog 可能看不到在跑的进程组。
        if check_disk_once; then
            :
        else
            log_msg "启动前磁盘已超限，放弃"
            final=1
            break
        fi

        resume=${RESUME}
        if [ "${attempt}" -gt 0 ]; then
            resume=1 # 重试时实验目录已存在；没有 checkpoint，trainer 仍从第 0 步开始
        fi
        log_msg "启动 attempt=$((attempt + 1)) RESUME=${resume}（${#OVERRIDES[@]} 个 override）"
        code=0
        run_attempt "${resume}" || code=$?
        log_msg "launcher 退出码 ${code}（attempt $((attempt + 1))）"

        if [ "${code}" -eq 0 ]; then
            final=0
            break
        fi
        if [ -n "${WRAPPER_SIGNAL}" ]; then
            final=$((128 + WRAPPER_SIGNAL))
            log_msg "wrapper 收到信号，退出码 ${final}"
            break
        fi
        if [ -f "${R}/DISK_LIMIT_EXCEEDED.txt" ]; then
            log_msg "磁盘超限停止，见 ${R}/DISK_LIMIT_EXCEEDED.txt；不重试"
            final=${code}
            break
        fi
        if [ "${code}" -gt 128 ] && [ "${code}" -lt 160 ]; then
            log_msg "训练被信号 $((code - 128)) 终止；不自动重试"
            final=${code}
            break
        fi

        attempt=$((attempt + 1))
        if [ "${attempt}" -gt "${MAX_RETRIES}" ]; then
            log_msg "已重试 ${MAX_RETRIES} 次仍失败，放弃"
            final=${code}
            break
        fi
        log_msg "非信号失败（exit ${code}）；${RETRY_DELAY_SECS}s 后第 ${attempt}/${MAX_RETRIES} 次重试。save_freq=-1 没有 checkpoint，重试从第 0 步开始"
        sleep "${RETRY_DELAY_SECS}"
    done

    exit "${final}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
