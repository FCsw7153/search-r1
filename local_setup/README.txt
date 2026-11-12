Search-R1 local environment — updated 2026-11-12

ACTIVE DATA (keep for training/evaluation):
  data/nq_hotpotqa_train/train.parquet: 169615 rows
  data/nq_hotpotqa_train/test.parquet: 51713 rows
  data/retrieval/e5_Flat.index: 21015324 vectors, 768 dimensions
  data/retrieval/wiki-18.jsonl: all 21015324 rows checked for valid JSON,
    string contents and consecutive IDs matching index order.
  data/models/e5-base-v2: retrieval encoder
  data/models/Qwen3.5-2B and Qwen3.5-4B: official downloaded weights

CLEANUP:
  Removed original index parts, compressed corpus, intermediate TAR,
  old Qwen2.5-0.5B smoke model and generated smoke datasets.
  Exact paths and freed bytes are recorded in cleanup.json.
  validation.json is historical: some listed originals were deliberately removed.
  Old run_smoke.sh/smoke_train.py depend on deleted fixtures/model and are retired.
  download_data.py skips source downloads when the prepared resource exists.

QWEN3.5 ENVIRONMENT:
  source local_setup/env_qwen35.sh
  Inherits host torch 2.11.0+cu130, Transformers 5.12.1, flash-attn.
  Added tensordict 0.15.6, faiss-cpu 1.16.0, codetiming 1.4.0.
  Exact packages: qwen35-environment.freeze.txt.
  The old .venv/Transformers 4.47.1 is retained for the existing retriever.

LOCAL MODEL VERIFICATION:
  python local_setup/validate_qwen35.py --model Qwen3.5-2B
  python local_setup/validate_qwen35.py --model Qwen3.5-4B
  Uses actual weights: text generation, BF16 full-parameter loss/backward,
  gradient clipping, AdamW update; records loss, finite gradients and peak VRAM.
  This is a short numerical/compatibility check, not a quality benchmark.
  Uses BF16 optimizer state; it does not validate production FP32 optimizer state.

TARGET MACHINE: one node with four full B300 GPUs.
  bash local_setup/validate_4gpu.sh
  Runs the same check using torchrun/DDP/NCCL across four GPUs.
  Four-GPU behavior cannot be verified on the current one-device MIG slice.
  This launcher is an environment test, not a Search-R1 PPO/GRPO training job.
  The original 8-GPU vLLM launchers are not validated for Qwen3.5.
  End-to-end Search-R1 RL, FSDP, retrieval integration and production batch sizes
  require additional validation before long training. Trainer import alone is
  not proof of compatibility.

RETRIEVAL:
  bash local_setup/retriever.sh
  Complete index alone uses about 60 GiB host RAM; current limit is 128 GiB.
  Full retrieval service was not launched alongside model training validation.

MEASURED ON CURRENT B300 MIG 2g.67gb (2026-11-12):
  Qwen3.5-2B: generation/backward/AdamW passed; peak allocated 16.89 GiB.
    Result: Qwen3.5-2B-validation.json
  Qwen3.5-4B: generation/backward/AdamW passed; peak allocated 35.28 GiB.
    Result: Qwen3.5-4B-validation-1gpu.json
  Both losses and gradient norms were finite; checked actual parameter change.
  Test uses batch size 1 and a short prompt. These peaks are not full RL budgets.
  Data after cleanup: 87.2954 GiB; freed 79.2328 GiB (logical file sizes).
  Recreate environment on compatible B300 host: bash local_setup/bootstrap_qwen35.sh
