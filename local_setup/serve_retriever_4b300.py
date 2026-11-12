"""GPU E5 + FP32 GPU FAISS with 128-query dynamic batches and exact-result caching."""
import asyncio
import importlib.util
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import faiss
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from retrieval_batcher import RetrievalBatcher

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('search_r1_retrieval', ROOT/'search_r1/search/retrieval_server.py')
retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval)
threads = int(os.environ.get('RETRIEVER_THREADS', '4'))
max_batch = int(os.environ.get('RETRIEVER_BATCH_SIZE', '128'))
index_device = os.environ.get('RETRIEVER_INDEX_DEVICE', 'gpu')
if index_device not in ('gpu', 'cpu'):
    raise ValueError('RETRIEVER_INDEX_DEVICE must be gpu or cpu')
log_path = Path(os.environ.get('RETRIEVER_METRICS', str(ROOT/'logs/retriever/requests.jsonl')))


def log_batch(record):
    with log_path.with_name('batches.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(f"Retrieval batch: {record['batch_size']} queries, "
          f"{record['service_seconds']:.3f}s, success={record['success']}", flush=True)


def search_batch(keys):
    faiss.omp_set_num_threads(threads)
    topk = max(k for _, k in keys)
    documents, scores = retrieval.retriever.batch_search(
        [query for query, _ in keys], num=topk, return_score=True)
    return [[{'document': doc, 'score': score} for doc, score in zip(docs[:k], values[:k], strict=True)]
            for (_, k), docs, values in zip(keys, documents, scores, strict=True)]


def initialize_retriever():
    # Construct and use FAISS GPU resources on the same dedicated worker thread.
    torch.set_num_threads(threads)
    faiss.omp_set_num_threads(threads)
    faiss.cvar.distance_compute_blas_threshold = 4
    if index_device == 'gpu':
        required = ('StandardGpuResources', 'GpuClonerOptions', 'index_cpu_to_gpu')
        missing = [name for name in required if not hasattr(faiss, name)]
        if missing:
            raise RuntimeError('GPU FAISS bindings are missing from the existing environment: '
                               + ', '.join(missing) + '. No automatic CPU fallback.')
    print(f'Loading exact FAISS index on {index_device}; max batch={max_batch}', flush=True)
    retrieval.config = retrieval.Config(
        retrieval_method='e5', retrieval_topk=3, faiss_gpu=False,
        index_path=str(ROOT/'data/retrieval/e5_Flat.index'),
        corpus_path=str(ROOT/'data/retrieval/wiki-18.jsonl'),
        retrieval_model_path=str(ROOT/'data/models/e5-base-v2'),
        retrieval_batch_size=max_batch)
    retrieval.retriever = retrieval.get_retriever(retrieval.config)
    if index_device == 'gpu':
        torch.cuda.set_device(0)
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(512 * 1024 * 1024)
        options = faiss.GpuClonerOptions()
        options.useFloat16 = False
        # The launcher exposes only EMBEDDING_GPU, so logical device 0 is shared with E5.
        retrieval.retriever.index = faiss.index_cpu_to_gpu(resources, 0, retrieval.retriever.index, options)
        retrieval.retriever._gpu_resources = resources  # keep resource lifetime >= index lifetime
    # Exercise the configured batch size before publishing readiness; bypass result cache.
    start = time.monotonic()
    search_batch([(f'What is the capital of France? Reference {i}', 3) for i in range(max_batch)])
    duration = time.monotonic()-start
    print(f'Warmup: {max_batch} queries in {duration:.3f}s ({max_batch/duration:.2f} queries/s); '
          'warmup throughput is not an end-to-end training benchmark.', flush=True)


@asynccontextmanager
async def lifespan(app):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    batcher = RetrievalBatcher(search_batch, max_batch=max_batch,
        wait_ms=int(os.environ.get('RETRIEVER_BATCH_WAIT_MS', '20')),
        cache_size=int(os.environ.get('RETRIEVER_CACHE_SIZE', '10000')), on_batch=log_batch)
    app.state.batcher = batcher
    try:
        await asyncio.get_running_loop().run_in_executor(batcher.executor, initialize_retriever)
        batcher.start()
        yield
    finally:
        await batcher.close()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def health():
    return {'ready': True, 'embedding': 'e5-base-v2', 'index': 'e5_Flat', 'topk': 3,
            'mode': 'dynamic_batch', 'max_batch': max_batch, 'index_device': index_device,
            'index_precision': 'float32', **app.state.batcher.stats}


@app.post('/retrieve')
async def endpoint(request: retrieval.QueryRequest):
    topk = request.topk or 3
    if not 1 <= topk <= 100 or not 1 <= len(request.queries) <= 256:
        raise HTTPException(400, 'Expected 1..256 queries and topk between 1 and 100')
    start = time.monotonic()
    success = False
    try:
        groups = await asyncio.gather(*(app.state.batcher.query(query, topk) for query in request.queries))
        success = True
        if not request.return_scores:
            groups = [[item['document'] for item in group] for group in groups]
        return {'result': groups}
    finally:
        with log_path.open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), queries=len(request.queries), success=success,
                total_seconds=time.monotonic()-start, mode='dynamic_batch'))+'\n')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('RETRIEVER_PORT', '8000')))
