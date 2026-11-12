"""Exercise the real E5 encoder and FastAPI retrieval endpoint on a tiny fixture."""
import json
from pathlib import Path
import faiss
from fastapi.testclient import TestClient
from search_r1.search import retrieval_server as server
root=Path('data/retrieval-smoke'); root.mkdir(exist_ok=True)
docs=[{'id':'0','contents':'"Paris"\nParis is the capital of France.'},
      {'id':'1','contents':'"Berlin"\nBerlin is the capital of Germany.'},
      {'id':'2','contents':'"Tokyo"\nTokyo is the capital of Japan.'}]
corpus=root/'corpus.jsonl'
corpus.write_text(''.join(json.dumps(d)+'\n' for d in docs))
encoder=server.Encoder('e5','data/models/e5-base-v2','mean',256,True)
vectors=encoder.encode([d['contents'] for d in docs],is_query=False)
index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
faiss.write_index(index,str(root/'index.faiss'))
del encoder
server.config=server.Config(retrieval_method='e5',retrieval_topk=1,
 index_path=str(root/'index.faiss'),corpus_path=str(corpus),faiss_gpu=False,
 retrieval_model_path='data/models/e5-base-v2',retrieval_use_fp16=True)
server.retriever=server.get_retriever(server.config)
response=TestClient(server.app).post('/retrieve',json={'queries':['What is the capital of France?'],'topk':1,'return_scores':True})
response.raise_for_status()
result=response.json();print(result)
assert 'Paris' in result['result'][0][0]['document']['contents']
print('E5 GPU + CPU FAISS + RETRIEVAL API PASS')
