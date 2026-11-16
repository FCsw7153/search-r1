#!/usr/bin/env bash
# Copy a checkpoint from the run's checkpoint dir (often the local overlay, see CKPT_ROOT
# in the launcher) to the shared workspace so it survives pod recreation.
#
#   bash verl/examples/search_r1/keep_checkpoint.sh <EXP> <step|latest|best> [--move]
#
#   step     exact global_step_<N>
#   latest   highest saved step
#   best     val step with the highest unweighted mean of the 7
#            val-core/searchR1_*/acc/mean@1 metrics (runs/*.metrics.jsonl);
#            saves the closest saved step <= that val step and says which.
#   --move   delete the local copy after the copy is verified (resume from that step
#            will then fail — only archive when you do not plan to resume).
#
# The destination is $EXP_DIR/kept_checkpoints/global_step_<N> (shared disk).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${CODE_DIR}/.." && pwd)}

usage() {
    echo "usage: bash $0 <EXP|path under experiments/> <step|latest|best> [--move]" >&2
    exit 2
}

target=${1:-}
which=${2:-}
if [ -z "${target}" ] || [ -z "${which}" ]; then
    usage
fi
move=0
case "${3:-}" in
    "") ;;
    --move) move=1 ;;
    *) usage ;;
esac
if [ "$#" -gt 3 ]; then
    usage
fi

if [ -d "${SEARCH_R1_HOME}/experiments/${target}" ]; then
    EXP_DIR=$(cd "${SEARCH_R1_HOME}/experiments/${target}" && pwd)
elif [ -d "${target}" ]; then
    EXP_DIR=$(cd "${target}" && pwd)
else
    echo "找不到实验: ${target}" >&2
    exit 1
fi

CKPT_LINK=${EXP_DIR}/checkpoints
if [ ! -d "${CKPT_LINK}" ]; then
    echo "没有 checkpoint 目录: ${CKPT_LINK}" >&2
    exit 1
fi
# checkpoints 可能是指向本地盘的符号链接（CKPT_ROOT），cd -P 解析成真实目录。
CKPT_DIR=$(cd "${CKPT_LINK}" && pwd -P)
echo "experiment    : ${EXP_DIR}"
echo "checkpoint dir: ${CKPT_DIR}"

saved=()
for d in "${CKPT_DIR}"/global_step_*; do
    [ -d "${d}" ] || continue
    n=${d##*/global_step_}
    case "${n}" in
        '' | *[!0-9]*) continue ;;
    esac
    saved+=("${n}")
done
if [ "${#saved[@]}" -eq 0 ]; then
    echo "没有已保存的 global_step_*: ${CKPT_DIR}" >&2
    exit 1
fi
# 数字排序（macOS/Linux 的 sort 都支持 -n）。
IFS=$'\n' saved=($(printf '%s\n' "${saved[@]}" | sort -n))
unset IFS
latest=${saved[${#saved[@]} - 1]}
echo "saved steps   : ${saved[*]}"

case "${which}" in
    latest)
        STEP=${latest}
        echo "latest        : global_step_${STEP}"
        ;;
    best)
        # 从 runs/*.metrics.jsonl 里读 val 步的 7 个 acc/mean@1，按未加权平均挑最好的。
        # 每行格式：{"step": N, "data": {"val-core/searchR1_<ds>/acc/mean@1": x, ...}}
        best_out=$(python3 - "${EXP_DIR}" <<'PY'
import glob
import json
import os
import sys

exp_dir = sys.argv[1]
rows = {}  # step -> (mean, n)
files = sorted(glob.glob(os.path.join(exp_dir, "runs", "*.metrics.jsonl")))
for path in files:
    try:
        handle = open(path, encoding="utf-8")
    except OSError:
        continue
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            data = rec.get("data")
            if not isinstance(step, int) or not isinstance(data, dict):
                continue
            vals = [
                float(v)
                for k, v in data.items()
                if k.startswith("val-core/searchR1_")
                and k.endswith("/acc/mean@1")
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            ]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            old = rows.get(step)
            if old is None or mean > old[0]:
                rows[step] = (mean, len(vals))

if not rows:
    print("metrics 里没有 val-core/searchR1_*/acc/mean@1 记录", file=sys.stderr)
    sys.exit(3)

print("val steps（未加权平均）:", file=sys.stderr)
for step in sorted(rows):
    mean, n = rows[step]
    print("  step %6d  mean@1=%.4f  (n=%d)" % (step, mean, n), file=sys.stderr)

# 平均值相同取更早的 step。
best_step = max(rows, key=lambda s: (rows[s][0], -s))
mean, n = rows[best_step]
print("BEST %d %.6f %d" % (best_step, mean, n))
PY
)
        read -r _ best_step best_mean best_n <<< "${best_out}"
        STEP=""
        for n in "${saved[@]}"; do
            if [ "${n}" -le "${best_step}" ]; then
                STEP=${n}
            else
                break
            fi
        done
        if [ -z "${STEP}" ]; then
            echo "最好的验证 step ${best_step} 之前没有已保存的 checkpoint（已保存: ${saved[*]}）" >&2
            exit 1
        fi
        echo "val best      : step ${best_step}, mean@1=${best_mean}（${best_n} 个指标）"
        echo "closest saved : global_step_${STEP}"
        ;;
    *)
        case "${which}" in
            '' | *[!0-9]*)
                echo "step 必须是数字、latest 或 best: ${which}" >&2
                exit 2
                ;;
        esac
        STEP=${which}
        ;;
esac

SRC=${CKPT_DIR}/global_step_${STEP}
if [ ! -d "${SRC}" ]; then
    echo "找不到 ${SRC}（已保存: ${saved[*]}）" >&2
    exit 1
fi

KEEP_ROOT=${EXP_DIR}/kept_checkpoints
DEST=${KEEP_ROOT}/global_step_${STEP}
PART=${DEST}.part
mkdir -p "${KEEP_ROOT}"
if [ -e "${DEST}" ] || [ -L "${DEST}" ]; then
    echo "已存在: ${DEST}（不覆盖；需要更新先手动删掉它）" >&2
    exit 1
fi
if [ -e "${PART}" ]; then
    echo "发现上次未完成的 ${PART}，请确认后手动删掉再重试" >&2
    exit 1
fi

echo "copy          : ${SRC} -> ${DEST}"
echo "free space    : $(df -h "${KEEP_ROOT}" | awk 'NR==2 {print $4}') on ${KEEP_ROOT}"
start=${SECONDS}
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${SRC}/" "${PART}/"
else
    cp -a "${SRC}" "${PART}"
fi

# 校验文件数 + 字节数（lstat，符号链接按链接自身算，不跟着目标走）。
tree_stat() {
    python3 - "$1" <<'PY'
import os
import sys

root = sys.argv[1]
files = 0
size = 0
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        path = os.path.join(dirpath, name)
        files += 1
        try:
            size += os.lstat(path).st_size
        except OSError:
            pass
print(files, size)
PY
}
read -r src_files src_bytes <<< "$(tree_stat "${SRC}")"
read -r dst_files dst_bytes <<< "$(tree_stat "${PART}")"
if [ "${src_files}" -ne "${dst_files}" ] || [ "${src_bytes}" -ne "${dst_bytes}" ]; then
    echo "校验失败：${src_files} files/${src_bytes} bytes -> ${dst_files} files/${dst_bytes} bytes" >&2
    echo "不完整的副本留在 ${PART}，确认后删掉再重试。" >&2
    exit 1
fi
mv "${PART}" "${DEST}"
secs=$((SECONDS - start))
if [ "${secs}" -le 0 ]; then
    secs=1
fi
echo "files         : ${src_files}（${src_bytes} bytes）"
echo "sizes         : $(du -sh "${SRC}" | cut -f1) -> $(du -sh "${DEST}" | cut -f1)    # ${secs}s"

if [ "${move}" = 1 ]; then
    case "${SRC}" in
        "${CKPT_DIR}"/global_step_*)
            rm -rf "${SRC}"
            echo "已从本地 checkpoint 目录删除 ${SRC}"
            if [ "${STEP}" = "${latest}" ]; then
                echo "警告：删掉的是最新 checkpoint；用 RESUME=1 续训会找不到 step ${STEP}。" >&2
            fi
            ;;
        *)
            echo "拒绝删除不在 checkpoint 目录里的路径: ${SRC}" >&2
            exit 1
            ;;
    esac
fi

echo "kept          : ${DEST}"
