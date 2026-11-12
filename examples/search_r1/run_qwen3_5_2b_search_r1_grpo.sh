#!/usr/bin/env bash
# GRPO | Qwen3.5-2B | FSDP2 training | Search-R1 with a local E5 + FAISS retriever
# Run from the repo root.
# Requires an activated environment (source .venv/bin/activate); do not use `uv run`,
# which would replace the environment's faiss-gpu without it being in uv.lock.

set -xeuo pipefail

########################### user-adjustable ###########################
PROJECT_NAME=${PROJECT_NAME:-search_r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5-2b-search-r1-grpo}
MODEL_PATH=${MODEL_PATH:-${PWD}/data/models/Qwen3.5-2B}
TRAIN_FILE=${TRAIN_FILE:-${PWD}/data/search_r1/train.parquet}
TEST_FILE=${TEST_FILE:-${PWD}/data/search_r1/test.parquet}
TOOL_CONFIG=${TOOL_CONFIG:-${PWD}/examples/search_r1/config/search_tool_config.yaml}
CKPTS_DIR=${CKPTS_DIR:-${PWD}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${PWD}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
########################### end user-adjustable ###########################

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p "${LOG_DIR}"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=512
    data.max_prompt_length=1024
    data.max_response_length=3072
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.param_offload=False
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
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=1
    trainer.total_training_steps=500
    trainer.total_epochs=15
    trainer.save_freq=100
    trainer.test_freq=50
    trainer.val_before_train=True
    trainer.resume_mode=auto
    trainer.logger=['console','file']
    trainer.log_val_generations=10
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/qwen3_5-2b-search-r1-${start_time}.log"
