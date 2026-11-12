# Search-R1 with verl

GRPO training for Search-R1-style reasoning with a local E5 + FAISS Wikipedia retriever,
using verl's built-in reward (`searchR1_*` data sources) and multi-turn `tool_agent` loop.

## Data layout

```
data/nq_hotpotqa_train/{train,test}.parquet   # original Search-R1 parquet files
data/retrieval/{e5_Flat.index,wiki-18.jsonl}  # FAISS index and corpus
data/models/{Qwen3.5-2B,e5-base-v2}           # policy model and retriever encoder
```

## Setup

```bash
source .venv/bin/activate
pip install -e . --no-deps
```

## Preprocess

```bash
python3 examples/data_preprocess/search_r1.py \
    --input_dir data/nq_hotpotqa_train \
    --local_save_dir data/search_r1
```

This writes `train.parquet` and `test.parquet`. Add `--val_samples_per_source 200` to also
write a smaller, stratified `test_sub.parquet` for faster validation.

## Start the retriever

```bash
CUDA_VISIBLE_DEVICES=0 nohup python examples/search_r1/retrieval_server.py --faiss_gpu \
    --index_path data/retrieval/e5_Flat.index \
    --corpus_path data/retrieval/wiki-18.jsonl \
    --retriever_model data/models/e5-base-v2 &
```

Check it is up:

```bash
curl -s http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
    -d '{"queries": ["who wrote the wealth of nations"], "topk": 3, "return_scores": false}'
```

## Train

```bash
bash examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

The recipe targets 4×B300. GPU0 also hosts the retriever, so the vLLM rollout uses 0.6 of
the GPU memory. See the script's variable block to override paths or the project name.
