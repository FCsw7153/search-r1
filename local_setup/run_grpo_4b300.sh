#!/usr/bin/env bash
# Search-R1 arXiv:2503.09516v3 Appendix B.2 (500 steps), migrated to Qwen3.5.
# Paper: batch 512, mini 256, n=5, lr=1e-6, warmup .285, KL .001, clip .2,
# temp/top_p=1, TP=1, 4 actions, 500 generated tokens/action, 500 retrieved tokens,
# context 4096, top3 E5/wiki18, binary EM, retrieved-token loss masking.
# Differences: 4 B300 instead of 8 H100; Qwen3.5 native tool format; microbatch=1
# with accumulation; FSDP2; eager vLLM; GPU FAISS shares GPU 0; vLLM memory .4.
# GPU retrieval reserves ~60 GiB, so the paper's vLLM .6 budget is reduced to .4.
# Usage: bash local_setup/run_grpo_4b300.sh [Hydra overrides]
# Config-only check (does not start Ray/GPU): DRY_RUN=1 bash local_setup/run_grpo_4b300.sh
# MODEL_SIZE=4B switches checkpoint; RUN_NAME selects checkpoint namespace for resume.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Background is the default. DRY_RUN remains synchronous and starts no training.
if [[ ${GRPO_FOREGROUND:-0} != 1 && ${DRY_RUN:-0} != 1 ]]; then
    run_name=${RUN_NAME:-qwen35-${MODEL_SIZE:-2B}-search-grpo-paper}
    run_dir="$ROOT/logs/$run_name"
    mkdir -p "$run_dir"
    pid_file="$run_dir/$(hostname).pid"
    exec 9>"$pid_file.lock"
    flock -n 9 || { echo "Another GRPO launch is in progress."; exit 1; }
    if [[ -s "$pid_file" ]]; then
        read -r existing_pid < "$pid_file"
        if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null &&
           [[ -r /proc/$existing_pid/cmdline ]] &&
           grep -aq 'run_grpo_4b300.sh' "/proc/$existing_pid/cmdline"; then
            echo "GRPO already running: PID $existing_pid"
            exit 0
        fi
    fi
    session_dir="$run_dir/$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "$session_dir"
    nohup env GRPO_FOREGROUND=1 GRPO_SESSION_DIR="$session_dir" \
      bash "$ROOT/local_setup/run_grpo_4b300.sh" "$@" \
      </dev/null >"$session_dir/launcher.log" 2>&1 9>&- &
    pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    echo "GRPO launched in background: PID $pid"
    echo "Log: $session_dir/launcher.log"
    echo "Metrics: $session_dir/metrics.jsonl"
    exit 0
fi
source "$ROOT/local_setup/env.sh"
export PYTHONPATH="$ROOT/local_setup:$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_MODELSCOPE=False HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8000/retrieve}
MODEL_SIZE=${MODEL_SIZE:-2B}
MODEL_PATH=${MODEL_PATH:-$ROOT/data/models/Qwen3.5-$MODEL_SIZE}
RUN_NAME=${RUN_NAME:-qwen35-${MODEL_SIZE}-search-grpo-paper}
# Each invocation has its own metrics file; resuming cannot overwrite previous logs.
SESSION_DIR=${GRPO_SESSION_DIR:-$ROOT/logs/$RUN_NAME/$(date +%Y%m%d_%H%M%S)_$$}
export VERL_FILE_LOGGER_PATH="$SESSION_DIR/metrics.jsonl"
export TENSORBOARD_DIR="$SESSION_DIR/tensorboard"
export TRAIN_CPU_THREADS=${TRAIN_CPU_THREADS:-4}
BATCH=${BATCH:-512}
MINIBATCH=${MINIBATCH:-256}
EVAL_K=${EVAL_K:-5}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/data/grpo_qwen35/train.parquet}
VAL_FILE=${VAL_FILE:-$ROOT/data/grpo_qwen35/test.parquet}
cd "$ROOT/local_setup"
if [[ ${DRY_RUN:-0} != 1 ]]; then
    "$ROOT/.venv/bin/python" - <<'PY'
import torch
if torch.cuda.device_count() != 4:
    raise SystemExit('Expected exactly four visible GPUs. Set CUDA_VISIBLE_DEVICES before running.')
PY
    bash "$ROOT/local_setup/prepare_grpo_data.sh"
    "$ROOT/.venv/bin/python" - <<'PY'
import json, os, time, urllib.request, urllib.error
health = os.environ['SEARCH_URL'].rsplit('/',1)[0]+'/health'
deadline = time.monotonic()+int(os.environ.get('RETRIEVER_WAIT_SECONDS', '1800'))
print('Waiting for retriever readiness:', health, flush=True)
while True:
    try:
        with urllib.request.urlopen(health, timeout=10) as response:
            status = json.load(response)
            if status.get('ready'):
                if status.get('index_device') != 'gpu' or status.get('max_batch') != 128:
                    raise SystemExit(f'Expected GPU FAISS with batch 128, got: {status}')
                print('GPU retriever ready (batch 128); starting GRPO.', flush=True)
                break
    except (OSError, ValueError):
        pass
    if time.monotonic() >= deadline:
        raise SystemExit('Retriever did not become ready; inspect logs/retriever/server.log.')
    time.sleep(5)
PY
fi
mkdir -p "$SESSION_DIR"
args=(
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False
 "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE"
 data.return_raw_chat=True "data.train_batch_size=$BATCH" data.val_batch_size=32
 data.max_prompt_length=2048 data.max_response_length=4096
 data.filter_overlong_prompts=True data.truncation=error data.seed=42
 data.dataloader_num_workers=2 data.filter_overlong_prompts_workers=2
 "actor_rollout_ref.model.path=$MODEL_PATH"
 actor_rollout_ref.model.use_remove_padding=False
 actor_rollout_ref.model.enable_gradient_checkpointing=True
 actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-6
 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
 "actor_rollout_ref.actor.ppo_mini_batch_size=$MINIBATCH"
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_epochs=1
 actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.clip_ratio_low=0.2
 actor_rollout_ref.actor.clip_ratio_high=0.2
 actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001
 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0
 actor_rollout_ref.actor.calculate_entropy=True actor_rollout_ref.actor.use_torch_compile=False
 actor_rollout_ref.actor.fsdp_config.param_offload=True
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
 actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
 actor_rollout_ref.ref.fsdp_config.param_offload=True
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4
 actor_rollout_ref.rollout.max_model_len=4096
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
 actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.temperature=1.0
 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1
 actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.enforce_eager=True
 actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.max_num_seqs=32
 actor_rollout_ref.rollout.enable_prefix_caching=False
 '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image: 0, video: 0}'
 actor_rollout_ref.rollout.agent.default_agent_loop=paper_tool_agent
 actor_rollout_ref.rollout.agent.num_workers=4
 "actor_rollout_ref.rollout.agent.agent_loop_config_path=$ROOT/local_setup/paper_agent.yaml"
 actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
 "actor_rollout_ref.rollout.multi_turn.tool_config_path=$ROOT/local_setup/search_tool_4b300.yaml"
 actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
 actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
 actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500
 actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
 actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False
 actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict
 "actor_rollout_ref.rollout.val_kwargs.n=$EVAL_K"
 actor_rollout_ref.rollout.val_kwargs.do_sample=True
 actor_rollout_ref.rollout.val_kwargs.temperature=1.0
 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.top_k=-1
 "reward.custom_reward_function.path=$ROOT/local_setup/paper_reward.py"
 reward.custom_reward_function.name=compute_score reward.num_workers=4
 trainer.use_v1=True trainer.v1.trainer_mode=sync trainer.nnodes=1 trainer.n_gpus_per_node=4
 trainer.total_epochs=15 trainer.total_training_steps=500
 trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=100
 trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep=6
 trainer.project_name=search_r1_qwen35 "trainer.experiment_name=$RUN_NAME"
 "trainer.default_local_dir=$ROOT/checkpoints/$RUN_NAME"
 "trainer.validation_data_dir=$SESSION_DIR/validation"
 "trainer.rollout_data_dir=$SESSION_DIR/rollouts"
 'trainer.logger=[console,file]' trainer.log_val_generations=0
 transfer_queue.backend.SimpleStorage.num_data_storage_units=1
 +ray_kwargs.ray_init.include_dashboard=False
 +ray_kwargs.ray_init.object_store_memory=2147483648
)
# Complete resolved config is preserved alongside metrics; does not allocate GPUs.
"$ROOT/.venv/bin/python" train_grpo.py "${args[@]}" "$@" --cfg job --resolve > "$SESSION_DIR/config.yaml"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "Config validated: $SESSION_DIR/config.yaml"
    exit 0
fi
printf 'Metrics and logs: %s\n' "$SESSION_DIR"
printf 'Checkpoint namespace: %s\n' "$ROOT/checkpoints/$RUN_NAME"
"$ROOT/.venv/bin/python" train_grpo.py "${args[@]}" "$@" 2>&1 | tee "$SESSION_DIR/train.log"
