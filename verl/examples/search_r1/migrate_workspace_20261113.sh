#!/usr/bin/env bash
# One-off migration of the old flat layout on the server (repo root == verl content,
# commit 7bc4201, branch verl-rebase tracking remote "publish") to the new layout
# (repo root == workspace, code under verl/). Safe to delete after the migration.
#
# Run from the workspace root, e.g. /home/work/mshop-agent-1/liang.mao/Search-R1:
#   bash /tmp/migrate.sh --dry-run   # default; prints the plan, changes nothing
#   bash /tmp/migrate.sh --execute

set -euo pipefail

MODE=${1:-}
[ -z "${MODE}" ] && MODE=--dry-run
case "${MODE}" in
    --dry-run|--execute) ;;
    *) echo "usage: $0 [--dry-run|--execute]" >&2; exit 2 ;;
esac

WORKSPACE=$(pwd)
cd "${WORKSPACE}"

########################### guards ###########################
for required in data .venv .git; do
    if [ ! -d "${WORKSPACE}/${required}" ]; then
        echo "ABORT: 当前目录不是 workspace 根（缺少 ${required}/）: ${WORKSPACE}" >&2
        exit 1
    fi
done
if [ -e "${WORKSPACE}/experiments" ]; then
    echo "ABORT: experiments/ 已存在，迁移不重复执行: ${WORKSPACE}/experiments" >&2
    exit 1
fi

LEFTOVERS=(verl/.git verl/verl verl/docs verl/recipe verl/.vscode)
tracked=$(git ls-files -- "${LEFTOVERS[@]}" || true)
if [ -n "${tracked}" ]; then
    echo "ABORT: 以下 leftover 路径仍有被跟踪的文件:" >&2
    echo "${tracked}" >&2
    exit 1
fi

########################### plan ###########################
# src|dst, one per line, in execution order.
MOVES=(
    "logs/search_r1/metrics_.jsonl|experiments/20261113-rmpad-on-collapsed/runs/20261112_180922.metrics.jsonl"
    "logs/search_r1/qwen3_5-2b-search-r1-grpo/qwen3_5-2b-search-r1-20261112_180922.log|experiments/20261113-rmpad-on-collapsed/runs/20261112_180922.log"
    "search_r1/smoke-2step.jsonl|experiments/smoke/20261112_174402-first-2step/runs/20261112_174402.metrics.jsonl"
    "logs/search_r1/qwen3_5-2b-search-r1-grpo/qwen3_5-2b-search-r1-20261112_174402.log|experiments/smoke/20261112_174402-first-2step/runs/20261112_174402.log"
    "search_r1/ab-normpad.jsonl|experiments/smoke/20261113_023705-ab-normpad/runs/20261113_023705.metrics.jsonl"
    "logs/search_r1/qwen3_5-2b-search-r1-grpo/qwen3_5-2b-search-r1-20261113_023705.log|experiments/smoke/20261113_023705-ab-normpad/runs/20261113_023705.log"
    "logs/qwen35-2B-search-grpo-paper/20261112_141839_201088/metrics.jsonl|experiments/20261112-codex-baseline/runs/20261112_141839.metrics.jsonl"
    "logs/qwen35-2B-search-grpo-paper/20261112_141839_201088/train.log|experiments/20261112-codex-baseline/runs/20261112_141839.log"
    "logs/qwen35-2B-search-grpo-paper/20261112_141839_201088/config.yaml|experiments/20261112-codex-baseline/runs/20261112_141839.config.yaml"
    "logs/qwen35-2B-search-grpo-paper/20261112_141839_201088/rollouts|experiments/20261112-codex-baseline/runs/20261112_141839.rollouts"
    # A running retriever holds it open; mv on the same FS is fine.
    "logs/retriever/server.log|logs/retriever.log"
)

FILE_DELETES=(
    logs/retriever/requests.jsonl
    logs/retriever/batches.jsonl
)

DIR_DELETES=(
    checkpoints
    logs/search_r1
    logs/qwen35-2B-search-grpo-paper
    local_setup
    outputs
    search_r1
    verl.egg-info
    data/grpo_qwen35
    data/verl_new_smoke
    verl/.git
    verl/verl
    verl/docs
    verl/recipe
    verl/.vscode
)

META_FILES=(
    experiments/20261113-rmpad-on-collapsed/meta.json
    experiments/20261112-codex-baseline/meta.json
    experiments/smoke/20261112_174402-first-2step/meta.json
    experiments/smoke/20261113_023705-ab-normpad/meta.json
)

PYTHON_FOR_INSTALL=${PYTHON_FOR_INSTALL:-.venv-mig/bin/python}

size_of() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        du -sh -- "$1" | cut -f1
    else
        echo MISSING
    fi
}

# Untracked top-level paths (data/, experiments/, .tools/, ...) are fine: the new
# root .gitignore allowlist ignores them after the pull. Only tracked edits and
# untracked files inside verl/ (which could collide with the incoming tree) abort.
prepull_check() {
    local mode=$1 tracked_changes untracked_verl
    tracked_changes=$(git status --porcelain --untracked-files=no)
    untracked_verl=$(git ls-files --others --exclude-standard -- verl)
    echo "=== PRE-PULL CHECK: tracked changes ==="
    if [ -n "${tracked_changes}" ]; then echo "${tracked_changes}"; else echo "(none)"; fi
    echo "=== PRE-PULL CHECK: untracked under verl/ ==="
    if [ -n "${untracked_verl}" ]; then echo "${untracked_verl}"; else echo "(none)"; fi
    if [ "${mode}" = abort ]; then
        if [ -n "${tracked_changes}" ]; then
            echo "ABORT: 有已跟踪文件的改动，先处理再 pull" >&2
            exit 1
        fi
        if [ -n "${untracked_verl}" ]; then
            echo "ABORT: verl/ 下有未跟踪文件，可能与 pull 冲突，先处理" >&2
            exit 1
        fi
    fi
}

write_metas() {
    python3 - "${WORKSPACE}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
model_path = os.path.join(root, "data", "models", "Qwen3.5-2B")


def build(name, created, status, notes, smoke, git_commit, overrides, ts):
    return {
        "name": name,
        "created": created,
        "status": status,
        "notes": notes,
        "smoke": smoke,
        "git_commit": git_commit,
        "git_dirty": False,
        "host": "",
        "model_path": model_path,
        "overrides": overrides,
        "launches": [
            {
                "ts": ts,
                "resume": False,
                "log": f"runs/{ts}.log",
                "metrics": f"runs/{ts}.metrics.jsonl",
            }
        ],
    }


# name follows the run script: for smoke, EXP is the suffix after <TS>-.
metas = [
    (
        "experiments/20261113-rmpad-on-collapsed/meta.json",
        build(
            name="20261113-rmpad-on-collapsed",
            created="2026-11-12T18:09:22",
            status="collapsed",
            notes="rmpad=True + CUDA graph。step 50 验证 EM 最好（NQ .397 / TriviaQA .585 / PopQA .403 / HotpotQA .355 / 2Wiki .327 / Musique .120 / Bamboogle .232），约 step 60 起 rollout_corr/kl 失控、截断到 3072、不再搜索，step 85 后 reward≈0。原因：remove padding 与 Qwen3.5 GatedDeltaNet 训推不一致（step1 KL 0.049）。",
            smoke=False,
            git_commit="7bc4201",
            overrides=["actor_rollout_ref.model.use_remove_padding=True"],
            ts="20261112_180922",
        ),
    ),
    (
        "experiments/20261112-codex-baseline/meta.json",
        build(
            name="20261112-codex-baseline",
            created="2026-11-12T14:18:39",
            status="stopped",
            notes="Codex 原配置：rmpad=False、eager、param/optimizer offload、micro_batch=1、tool response 截断 500 字符。约 900 s/step，KL≈7e-4。",
            smoke=False,
            git_commit="db65c14",
            overrides=[],
            ts="20261112_141839",
        ),
    ),
    (
        "experiments/smoke/20261112_174402-first-2step/meta.json",
        build(
            name="first-2step",
            created="2026-11-12T17:44:02",
            status="finished",
            notes="第一次 4 卡试跑：117 s/step，但 KL 0.049",
            smoke=True,
            git_commit="7bc4201",
            overrides=["actor_rollout_ref.model.use_remove_padding=True"],
            ts="20261112_174402",
        ),
    ),
    (
        "experiments/smoke/20261113_023705-ab-normpad/meta.json",
        build(
            name="ab-normpad",
            created="2026-11-13T02:37:05",
            status="finished",
            notes="实验 A：KL 回到 6e-4，182 s/step，但显存峰值 ~219GB",
            smoke=True,
            git_commit="7bc4201",
            overrides=["actor_rollout_ref.model.use_remove_padding=False"],
            ts="20261113_023705",
        ),
    ),
]

for rel, payload in metas:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"WROTE {rel}")
PY
}

########################### dry run ###########################
if [ "${MODE}" = "--dry-run" ]; then
    echo "=== MOVE ==="
    for pair in "${MOVES[@]}"; do
        printf 'MOVE %s -> %s  [%s]\n' "${pair%%|*}" "${pair#*|}" "$(size_of "${pair%%|*}")"
    done
    echo "=== DELETE ==="
    for path in "${FILE_DELETES[@]}"; do
        printf 'DELETE %s  [%s]\n' "${path}" "$(size_of "${path}")"
    done
    for path in "${DIR_DELETES[@]}"; do
        printf 'DELETE %s  [%s]\n' "${path}" "$(size_of "${path}")"
    done
    printf 'RMDIR %s\n' logs/retriever
    printf 'DELETE %s (find -name __pycache__ -type d -prune)\n' 'verl/**/__pycache__'
    echo "=== WRITE ==="
    for path in "${META_FILES[@]}"; do
        printf 'WRITE meta.json %s\n' "${path}"
    done
    echo "=== GIT ==="
    prepull_check report
    echo "GIT PULL publish verl-rebase"
    echo "=== REINSTALL ==="
    echo "REINSTALL .tools/uv pip install -e verl --no-deps --python ${PYTHON_FOR_INSTALL}"
    exit 0
fi

########################### execute ###########################
echo "=== MOVE ==="
for pair in "${MOVES[@]}"; do
    src=${pair%%|*}
    dst=${pair#*|}
    if [ ! -e "${src}" ] && [ ! -L "${src}" ]; then
        echo "SKIP missing ${src}"
        continue
    fi
    mkdir -p "$(dirname "${dst}")"
    mv -- "${src}" "${dst}"
    echo "MOVED ${src} -> ${dst}"
done

echo "=== WRITE ==="
write_metas

echo "=== DELETE ==="
for path in "${FILE_DELETES[@]}"; do
    if [ -e "${path}" ] || [ -L "${path}" ]; then
        rm -f -- "${path}"
        echo "DELETED ${path}"
    else
        echo "SKIP missing ${path}"
    fi
done
for path in "${DIR_DELETES[@]}"; do
    if [ -e "${path}" ] || [ -L "${path}" ]; then
        rm -rf -- "${path}"
        echo "DELETED ${path}"
    else
        echo "SKIP missing ${path}"
    fi
done
if [ -d logs/retriever ]; then
    if rmdir -- logs/retriever 2>/dev/null; then
        echo "RMDIR logs/retriever"
    else
        echo "NOTE: logs/retriever 非空，保留"
    fi
fi
if [ -d verl ]; then
    find verl -name __pycache__ -type d -prune -exec rm -rf {} +
    echo "DELETED verl/**/__pycache__"
fi

echo "=== PRE-PULL CHECKS ==="
prepull_check abort

echo "=== GIT PULL ==="
git pull --ff-only publish verl-rebase
git log --oneline -3

echo "=== REINSTALL ==="
.tools/uv pip install -e verl --no-deps --python "${PYTHON_FOR_INSTALL}"
module_file=$("${PYTHON_FOR_INSTALL}" -c "import verl; print(verl.__file__)")
case "${module_file}" in
    */verl/verl/__init__.py) echo "OK verl -> ${module_file}" ;;
    *)
        echo "ABORT: verl 解析到 ${module_file}，期望 .../verl/verl/__init__.py" >&2
        exit 1
        ;;
esac

echo "=== FINAL ==="
ls -la
find experiments -maxdepth 3 | sort
du -sh experiments logs data
git status --short
