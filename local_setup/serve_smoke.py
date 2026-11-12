"""Local-only fixture retrieval service for the end-to-end training smoke test."""
import uvicorn
from search_r1.search import retrieval_server as server
server.config=server.Config(retrieval_method='e5',retrieval_topk=3,
 index_path='data/retrieval-smoke/index.faiss',corpus_path='data/retrieval-smoke/corpus.jsonl',
 faiss_gpu=False,retrieval_model_path='data/models/e5-base-v2',retrieval_use_fp16=True)
server.retriever=server.get_retriever(server.config)
uvicorn.run(server.app,host='127.0.0.1',port=18080)
