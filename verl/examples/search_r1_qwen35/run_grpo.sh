#!/usr/bin/env bash
# Starting recipe, not a measured hardware configuration. Run from anywhere.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}
: "${MODEL_PATH:?Set the exact local Qwen3.5 checkpoint path}"
: "${TRAIN_FILE:?Set the processed training parquet path}"
: "${VAL_FILE:?Set an independent processed validation parquet path}"
NGPUS=${NGPUS:-8}
TP=${TP:-2}
BATCH=${BATCH:-32}
MINIBATCH=${MINIBATCH:-16}
GROUP_SIZE=${GROUP_SIZE:-4}
PROMPT_LEN=${PROMPT_LEN:-2048}
TRAJECTORY_LEN=${TRAJECTORY_LEN:-8192}
CONTEXT_LEN=$((PROMPT_LEN + TRAJECTORY_LEN))
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8000/retrieve}

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.return_raw_chat=True \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length="$PROMPT_LEN" \
  data.max_response_length="$TRAJECTORY_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINIBATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_model_len="$CONTEXT_LEN" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n="$GROUP_SIZE" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO_ROOT/examples/search_r1_qwen35/search_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
  actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.test_freq=50 \
  trainer.save_freq=100 \
  trainer.project_name=search_r1_qwen35 \
  trainer.experiment_name=qwen35_search_grpo \
  trainer.default_local_dir="$REPO_ROOT/checkpoints/search_r1_qwen35" \
  trainer.logger='["console"]' \
  "$@"
