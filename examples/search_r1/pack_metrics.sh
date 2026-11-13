#!/usr/bin/env bash
# Pack the small metadata of an experiment (meta/metrics/command/hydra configs) into
# $SEARCH_R1_HOME/experiments/_export/<name>_<TS>.tar.gz. Checkpoints and *.log are excluded.
#
#   bash examples/search_r1/pack_metrics.sh <EXP | path under experiments/> | all

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${REPO_ROOT}/.." && pwd)}
EXP_ROOT=${SEARCH_R1_HOME}/experiments
TS=$(date +%Y%m%d_%H%M%S)
target=${1:?usage: bash pack_metrics.sh <EXP|path under experiments/|all>}

find_in() {
    local rel=$1
    (
        cd "${EXP_ROOT}"
        find "${rel}" -type f \
            \( -path "${rel}/meta.json" -o -path "${rel}/runs/*.metrics.jsonl" \
            -o -path "${rel}/runs/*.command.sh" -o -path "${rel}/runs/*.hydra/.hydra/*.yaml" \) \
            -print
    )
}

list=$(mktemp)
trap 'rm -f "${list}"' EXIT

if [ "${target}" = all ]; then
    name=all
    if [ ! -d "${EXP_ROOT}" ]; then
        echo "experiments 目录不存在: ${EXP_ROOT}" >&2
        exit 1
    fi
    for dir in "${EXP_ROOT}"/*/; do
        rel=$(basename "${dir}")
        [ "${rel}" = "_export" ] && continue
        find_in "${rel}"
    done > "${list}"
else
    if [ -d "${EXP_ROOT}/${target}" ]; then
        dir=$(cd "${EXP_ROOT}/${target}" && pwd)
    elif [ -d "${target}" ]; then
        dir=$(cd "${target}" && pwd)
    else
        echo "找不到实验: ${target}" >&2
        exit 1
    fi
    case "${dir}" in
        "${EXP_ROOT}"/*) rel=${dir#"${EXP_ROOT}"/} ;;
        *)
            echo "实验不在 experiments/ 下: ${dir}" >&2
            exit 1
            ;;
    esac
    name=${rel//\//_}
    find_in "${rel}" > "${list}"
fi

if [ ! -s "${list}" ]; then
    echo "没有可打包的 meta/metrics/command/hydra 文件" >&2
    exit 1
fi

mkdir -p "${EXP_ROOT}/_export"
out=${EXP_ROOT}/_export/${name}_${TS}.tar.gz
tar -czf "${out}" -C "${EXP_ROOT}" -T "${list}"
echo "${out}"
du -h "${out}" | cut -f1
