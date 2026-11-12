"""Resume downloads into this project; uses the host CA bundle."""
import os
from pathlib import Path
os.environ.setdefault('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import snapshot_download,hf_hub_download
root=Path(__file__).resolve().parents[1]/'data'
snapshot_download('PeterJinGo/nq_hotpotqa_train',repo_type='dataset',local_dir=root/'nq_hotpotqa_train',max_workers=2)
for repo,files in [('PeterJinGo/wiki-18-e5-index',['part_aa','part_ab']),('PeterJinGo/wiki-18-corpus',['wiki-18.jsonl.gz'])]:
    ready = root/'retrieval'/('e5_Flat.index' if 'index' in repo else 'wiki-18.jsonl')
    if ready.exists():
        print('SKIP prepared resource', ready, flush=True)
        continue
    for name in files:
        hf_hub_download(repo_id=repo,filename=name,repo_type='dataset',local_dir=root/'retrieval')
