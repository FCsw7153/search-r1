#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$SEARCH_R1_ROOT"
exec python search_r1/search/retrieval_server.py \
 --index_path data/retrieval/e5_Flat.index \
 --corpus_path data/retrieval/wiki-18.jsonl \
 --retriever_name e5 --retriever_model data/models/e5-base-v2 --topk 3
