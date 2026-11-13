# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E5 + FAISS retrieval server for Search-R1 with async micro-batching."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# Time to wait for more queries before running a batch, in seconds.
BATCH_WAIT = 0.005


class Encoder:
    """E5 encoder: ``query: `` prefix, mean pooling and L2 normalization."""

    def __init__(self, model_path: str, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path).eval().cuda().half()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, queries: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            [f"query: {query}" for query in queries],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()


class Retriever:
    """Load the corpus and FAISS index; used only from the dedicated worker thread."""

    def __init__(self, args):
        self.encoder = Encoder(args.retriever_model)
        self.index = faiss.read_index(args.index_path)
        if args.faiss_gpu:
            resources = faiss.StandardGpuResources()
            options = faiss.GpuClonerOptions()
            options.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(resources, 0, self.index, options)
            self.gpu_resources = resources  # keep alive as long as the index
        self.corpus = datasets.load_dataset("json", data_files=args.corpus_path, split="train", num_proc=4)

    def search(self, queries: list[str], topk: int) -> tuple[list, list]:
        scores, indices = self.index.search(self.encoder.encode(queries), k=topk)
        documents = [[self.corpus[int(index)] for index in row] for row in indices]
        return documents, scores.tolist()


class Batcher:
    """Coalesce per-query futures into batches and run them on one worker thread."""

    def __init__(self, retriever: Retriever, executor: ThreadPoolExecutor, max_batch: int):
        self.retriever = retriever
        self.executor = executor
        self.max_batch = max_batch
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker = None

    def start(self):
        self.worker = asyncio.create_task(self._run())

    async def search(self, query: str, topk: int) -> tuple[list, list]:
        future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait((query, topk, future))
        return await future

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + BATCH_WAIT
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            try:
                queries = [query for query, _, _ in batch]
                documents, scores = await loop.run_in_executor(
                    self.executor, self.retriever.search, queries, max(topk for _, topk, _ in batch)
                )
                for (_, topk, future), row_documents, row_scores in zip(batch, documents, scores, strict=True):
                    if not future.done():
                        future.set_result((row_documents[:topk], row_scores[:topk]))
            except Exception as exc:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(exc)


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


args = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FAISS GPU and the encoder are initialized and used on this single thread.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-batch")
    retriever = await asyncio.get_running_loop().run_in_executor(executor, Retriever, args)
    app.state.batcher = Batcher(retriever, executor, args.max_batch)
    app.state.batcher.start()
    yield
    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    topk = request.topk or args.topk
    groups = await asyncio.gather(*(app.state.batcher.search(query, topk) for query in request.queries))
    if request.return_scores:
        result = [
            [{"document": document, "score": score} for document, score in zip(documents, scores, strict=True)]
            for documents, scores in groups
        ]
    else:
        result = [documents for documents, _ in groups]
    return {"result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Search-R1 FAISS retriever.")
    parser.add_argument("--index_path", required=True, help="FAISS index file.")
    parser.add_argument("--corpus_path", required=True, help="JSONL corpus file.")
    parser.add_argument("--retriever_model", required=True, help="E5 model path or HuggingFace id.")
    parser.add_argument("--topk", type=int, default=3, help="Default number of passages per query.")
    parser.add_argument("--max_batch", type=int, default=128, help="Maximum queries per encode+search call.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Run the FAISS index on GPU 0.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
