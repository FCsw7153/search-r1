"""Merge downloaded index parts and decompress corpus without deleting originals."""
from pathlib import Path
import gzip, shutil, os
root=Path(__file__).resolve().parents[1]/'data/retrieval'
index=root/'e5_Flat.index'
parts=[root/'part_aa',root/'part_ab']
if not index.exists():
    assert parts[0].stat().st_size==42949672960
    assert parts[1].stat().st_size==21609402413
    temp=index.with_suffix('.index.tmp')
    with temp.open('wb') as out:
        for part in parts:
            with part.open('rb') as src: shutil.copyfileobj(src,out,16*1024*1024)
    assert temp.stat().st_size==sum(p.stat().st_size for p in parts)
    os.replace(temp,index)
    print('INDEX MERGED',index,flush=True)
corpus=root/'wiki-18.jsonl'
if not corpus.exists():
    temp=corpus.with_suffix('.jsonl.tmp')
    with gzip.open(root/'wiki-18.jsonl.gz','rb') as src,temp.open('wb') as out:
        shutil.copyfileobj(src,out,16*1024*1024)
    os.replace(temp,corpus)
    print('CORPUS DECOMPRESSED',corpus,flush=True)
# The published .jsonl.gz actually contains a tar archive, despite its name.
import tarfile
if tarfile.is_tarfile(corpus):
    archive=root/'wiki-18.corpus.tar'
    os.replace(corpus,archive)
    with tarfile.open(archive,'r:') as tar:
        members=[m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jsonl')]
        if len(members)!=1:
            raise RuntimeError(f'Expected one JSONL member, got {len(members)}')
        temp=corpus.with_suffix('.jsonl.tmp')
        with tar.extractfile(members[0]) as src,temp.open('wb') as out:
            shutil.copyfileobj(src,out,16*1024*1024)
        assert temp.stat().st_size==members[0].size
        os.replace(temp,corpus)
        print('CORPUS EXTRACTED',members[0].name,members[0].size,flush=True)
