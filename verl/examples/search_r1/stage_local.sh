#!/usr/bin/env bash
# Stage the shared workspace's venv and retrieval data onto the container's local disk
# (pod-local overlay, engine $LOCAL_ROOT, default /home/work/search_r1_local).
#
#   bash verl/examples/search_r1/stage_local.sh [venv] [retrieval] [all]
#
# Why: $SEARCH_R1_HOME lives on spfs, a slow network FUSE mount. Importing torch/vllm
# from the shared .venv took >2 minutes on a cold node, and every Ray worker re-imports.
# The local overlay is fast but is wiped when the pod/run ends — this script is
# idempotent, so just rerun it on a new pod.
#
#   venv       copy $SEARCH_R1_HOME/.venv -> $LOCAL_ROOT/.venv, rewrite the paths baked
#              into console-script shebangs / activate scripts, precompile .pyc, verify
#              imports, and time `import torch, vllm` for local vs shared.
#   retrieval  copy data/retrieval/{e5_Flat.index,wiki-18.jsonl} and data/models/e5-base-v2,
#              skipping files whose size already matches.
#
# Never deletes anything under $SEARCH_R1_HOME; refuses LOCAL_ROOT inside the workspace.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
SEARCH_R1_HOME=${SEARCH_R1_HOME:-$(cd "${CODE_DIR}/.." && pwd)}
LOCAL_ROOT=${LOCAL_ROOT-/home/work/search_r1_local}
if [ -z "${LOCAL_ROOT}" ]; then
    echo "LOCAL_ROOT 为空：请指定本地盘目录（默认 /home/work/search_r1_local）" >&2
    exit 1
fi

SRC_VENV=${SEARCH_R1_HOME}/.venv
DST_VENV=${LOCAL_ROOT}/.venv
SRC_DATA=${SEARCH_R1_HOME}/data
DST_DATA=${LOCAL_ROOT}/data
# 完成标记：launcher / start_retriever.sh 只认这两个文件，不认“目录存在”。
# bin/python 是复制刚开始就出现的符号链接，半截 venv 会被误用。venv 标记放在 .venv
# 外面，rsync --delete / cp -a 不会碰到它。
VENV_MARKER=${LOCAL_ROOT}/.venv.ok
RETRIEVAL_MARKER=${DST_DATA}/.retrieval.ok

if [ "$#" -eq 0 ]; then
    set -- all
fi

want_venv=0
want_retrieval=0
for target in "$@"; do
    case "${target}" in
        venv) want_venv=1 ;;
        retrieval) want_retrieval=1 ;;
        all) want_venv=1; want_retrieval=1 ;;
        *)
            echo "usage: bash $0 [venv] [retrieval] [all]" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${LOCAL_ROOT}"
local_real=$(cd "${LOCAL_ROOT}" && pwd -P)
home_real=$(cd "${SEARCH_R1_HOME}" && pwd -P)
case "${local_real}" in
    "${home_real}" | "${home_real}"/*)
        echo "LOCAL_ROOT 不能指向工作区 ${home_real}（共享盘慢，而且脚本只往本地盘写）: ${local_real}" >&2
        exit 1
        ;;
esac

echo "SEARCH_R1_HOME=${SEARCH_R1_HOME}"
echo "LOCAL_ROOT    =${local_real}    # pod 重建会清空，重跑本脚本即可"

############### venv ###############
# 复制过来的 venv 里有两类绝对路径写死了旧位置：bin/* 的 shebang 和 activate* 里的
# VIRTUAL_ENV。pyvenv.cfg 的 home 指向系统 python，不用改。可编辑安装的 .pth/finder
# 指向 $SEARCH_R1_HOME/verl 的代码目录，和 venv 位置无关，也不要改。
fix_venv_paths() {
    local f first tmp
    for f in "${DST_VENV}"/bin/*; do
        [ -f "${f}" ] || continue
        # 只看真正以 "#!" 开头的文本文件，避免碰二进制。
        if [ "$(head -c 2 "${f}" 2>/dev/null || true)" != "#!" ]; then
            continue
        fi
        first=$(head -n 1 "${f}")
        case "${first}" in
            "#!${SRC_VENV}/"*)
                tmp=$(mktemp)
                sed "1s|^#!${SRC_VENV}/|#!${DST_VENV}/|" "${f}" > "${tmp}"
                cat "${tmp}" > "${f}"   # 保留原文件的权限/属主/inode
                rm -f "${tmp}"
                echo "  shebang: bin/${f##*/}"
                ;;
        esac
    done
    for f in "${DST_VENV}"/bin/activate "${DST_VENV}"/bin/activate.*; do
        [ -f "${f}" ] || continue
        if grep -qF -- "${SRC_VENV}" "${f}" 2>/dev/null; then
            tmp=$(mktemp)
            sed "s|${SRC_VENV}|${DST_VENV}|g" "${f}" > "${tmp}"
            cat "${tmp}" > "${f}"
            rm -f "${tmp}"
            echo "  activate: bin/${f##*/}"
        fi
    done
}

stage_venv() {
    rm -f "${VENV_MARKER}"   # 先删标记：中途失败/中断时不能留下“已完成”
    if [ ! -x "${SRC_VENV}/bin/python" ]; then
        echo "找不到共享 venv: ${SRC_VENV}" >&2
        return 1
    fi
    echo "== venv: ${SRC_VENV} -> ${DST_VENV}"
    local start=${SECONDS}
    mkdir -p "${DST_VENV}"
    if command -v rsync >/dev/null 2>&1; then
        # 共享 .venv 里可能有指向 .cache 的硬链接，复制会实体化（占空间但更快）。
        rsync -a --delete "${SRC_VENV}/" "${DST_VENV}/"
    else
        cp -a "${SRC_VENV}/." "${DST_VENV}/"
    fi
    echo "  copy: $((SECONDS - start))s"
    fix_venv_paths

    local site_packages
    site_packages=$(ls -d "${DST_VENV}"/lib/python*/site-packages 2>/dev/null | head -n 1 || true)
    if [ -n "${site_packages}" ]; then
        start=${SECONDS}
        # 只是预热 __pycache__，个别文件编译失败不影响 import。
        "${DST_VENV}/bin/python" -m compileall -q -j 16 "${site_packages}" \
            || echo "  compileall 有失败项（不影响使用，首次 import 会重新编译）" >&2
        echo "  compileall $(du -sh "${site_packages}" 2>/dev/null | cut -f1): $((SECONDS - start))s"
    fi

    echo "  verify: import verl, torch, vllm, faiss"
    if ! "${DST_VENV}/bin/python" -c 'import verl, torch, vllm, faiss; print(verl.__file__)'; then
        echo "  验证失败：本地 venv 不能 import verl/torch/vllm/faiss。" >&2
        echo "  editable 安装的 .pth/finder 应指向 ${SEARCH_R1_HOME}/verl；检查一下再重跑。" >&2
        return 1
    fi
    # 复制、路径修正、import 验证都通过后才写标记。
    : > "${VENV_MARKER}"
    echo "  marker: ${VENV_MARKER}"

    TIMEFORMAT='%3R s'
    echo "== time python -c 'import torch, vllm'"
    echo -n "  local  ${DST_VENV}: "
    time "${DST_VENV}/bin/python" -c 'import torch, vllm' \
        || echo "  （local import 失败）" >&2
    if [ -x "${SRC_VENV}/bin/python" ]; then
        echo -n "  shared ${SRC_VENV}: "
        time "${SRC_VENV}/bin/python" -c 'import torch, vllm' \
            || echo "  （shared import 失败）" >&2
    fi
}

############### retrieval ###############
copy_file() {
    local src=$1 dst=$2 ssize dsize start secs
    if [ ! -f "${src}" ]; then
        echo "缺少 ${src}" >&2
        return 1
    fi
    ssize=$(wc -c < "${src}" | tr -d ' ')
    if [ -f "${dst}" ]; then
        dsize=$(wc -c < "${dst}" | tr -d ' ')
        if [ "${ssize}" -eq "${dsize}" ]; then
            echo "  skip ${dst}（大小一致，$((ssize / 1048576)) MB）"
            return 0
        fi
        echo "  redo ${dst}（${dsize} -> ${ssize} bytes）"
    fi
    start=${SECONDS}
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${src}" "${dst}.part"
    else
        cp -a "${src}" "${dst}.part"
    fi
    mv -f "${dst}.part" "${dst}"
    dsize=$(wc -c < "${dst}" | tr -d ' ')
    if [ "${ssize}" -ne "${dsize}" ]; then
        echo "校验失败：${dst} 大小 ${dsize} != ${ssize}" >&2
        return 1
    fi
    secs=$((SECONDS - start))
    if [ "${secs}" -le 0 ]; then
        secs=1
    fi
    echo "  copied ${dst}: $((ssize / 1048576)) MB in ${secs}s（$((ssize / 1048576 / secs)) MB/s）"
}

copy_tree() {
    local src=$1 dst=$2 f rel bytes=0 copied=0 skipped=0 start=${SECONDS}
    if [ ! -d "${src}" ]; then
        echo "缺少 ${src}" >&2
        return 1
    fi
    while IFS= read -r -d '' f; do
        rel=${f#"${src}"/}
        if [ -f "${dst}/${rel}" ] \
            && [ "$(wc -c < "${f}")" -eq "$(wc -c < "${dst}/${rel}")" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        mkdir -p "$(dirname "${dst}/${rel}")"
        cp -a "${f}" "${dst}/${rel}"
        bytes=$((bytes + $(wc -c < "${f}")))
        copied=$((copied + 1))
    done < <(find "${src}" -type f -print0)
    echo "  tree ${dst}: copied ${copied} files（$((bytes / 1048576)) MB）, skipped ${skipped}, $((SECONDS - start))s"
}

stage_retrieval() {
    rm -f "${RETRIEVAL_MARKER}"   # 先删标记：中途失败/中断时不能留下“已完成”
    echo "== retrieval: ${SRC_DATA} -> ${DST_DATA}"
    mkdir -p "${DST_DATA}/retrieval" "${DST_DATA}/models"
    copy_file "${SRC_DATA}/retrieval/e5_Flat.index" "${DST_DATA}/retrieval/e5_Flat.index"
    copy_file "${SRC_DATA}/retrieval/wiki-18.jsonl" "${DST_DATA}/retrieval/wiki-18.jsonl"
    copy_tree "${SRC_DATA}/models/e5-base-v2" "${DST_DATA}/models/e5-base-v2"

    # copy_file 已校验两个大文件；模型树再逐文件比对大小，确认没有半截目录。
    local f rel
    while IFS= read -r -d '' f; do
        rel=${f#"${SRC_DATA}/models/e5-base-v2"/}
        if [ ! -f "${DST_DATA}/models/e5-base-v2/${rel}" ] \
            || [ "$(wc -c < "${f}")" -ne "$(wc -c < "${DST_DATA}/models/e5-base-v2/${rel}")" ]; then
            echo "校验失败：${DST_DATA}/models/e5-base-v2/${rel}" >&2
            return 1
        fi
    done < <(find "${SRC_DATA}/models/e5-base-v2" -type f -print0)

    : > "${RETRIEVAL_MARKER}"
    echo "  marker: ${RETRIEVAL_MARKER}"
}

if [ "${want_venv}" = 1 ]; then
    stage_venv
fi
if [ "${want_retrieval}" = 1 ]; then
    stage_retrieval
fi

echo "== done. LOCAL_ROOT=${LOCAL_ROOT}"
echo "   标记位置：${VENV_MARKER} / ${RETRIEVAL_MARKER}"
echo "   之后 launcher / start_retriever.sh 只认标记存在且完整的本地副本；"
echo "   想强制用共享盘：LOCAL_ROOT= bash <script>"
