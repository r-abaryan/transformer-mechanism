"""Acquire pinned public publisher files and verify bytes, without dataset execution."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVISION='b08601e04326c79dfdd32d625aee71d232d685c3'
EXPECTED={
    'train':(6068114,'dfc27e4360c639dc1fba1e403bfffd53af4a5c75d5363b5724d49bf12d07cce6',36718),
    'validation':(617738,'717de9a0c1c0b0b1dfdd8f1e6ad8a30ece618bbde81f5da8207277547d324215',3760),
    'test':(685430,'e6b3913da714b63a60a571698b20ff15441fb015783ea1b5285f707d4f2f00a9',4358),
}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Transformers_RL-research/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response: return response.read()


def main():
    folder=ROOT/'data/wikitext-2-v1'; folder.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'results/wikitext2_acquisition.json'; assert not manifest_path.exists()
    api=f'https://huggingface.co/api/datasets/Salesforce/wikitext/revision/{REVISION}'
    metadata=json.loads(get(api)); assert metadata['sha']==REVISION
    info=next(c for c in metadata['cardData']['dataset_info'] if c['config_name']=='wikitext-2-v1')
    assert {x['name']:x['num_examples'] for x in info['splits']}=={k:v[2] for k,v in EXPECTED.items()}
    files=json.loads(get(f'https://huggingface.co/api/datasets/Salesforce/wikitext/tree/{REVISION}/wikitext-2-v1'))
    entries=[]
    for split,(size,sha,rows) in EXPECTED.items():
        name=f'{split}-00000-of-00001.parquet'; remote=f'wikitext-2-v1/{name}'
        record=next(f for f in files if f['path']==remote)
        assert record['size']==size and record['lfs']['oid']==sha
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/{remote}'
        path=folder/name
        if path.exists(): assert path.stat().st_size==size and digest(path)==sha
        else:
            content=get(url); assert len(content)==size and hashlib.sha256(content).hexdigest()==sha
            assert content[:4]==b'PAR1' and content[-4:]==b'PAR1'; path.write_bytes(content)
        with path.open('rb') as f:
            assert f.read(4)==b'PAR1'; f.seek(-4,2); assert f.read()==b'PAR1'
        entries.append(dict(split=split,path=str(path.relative_to(ROOT)),url=url,size=size,sha256=sha,publisher_rows=rows))
        print(json.dumps(dict(split=split,bytes=size,sha256=sha)),flush=True)
    card_path=folder/'publisher-README.md'; url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/README.md'
    content=get(url)
    if card_path.exists(): assert card_path.read_bytes()==content
    else: card_path.write_bytes(content)
    meta_path=folder/'publisher-metadata.json'; assert not meta_path.exists()
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    manifest=dict(status='publisher_bytes_verified_not_parsed',dataset='Salesforce/wikitext',config='wikitext-2-v1',
                  revision=REVISION,licenses=metadata['cardData']['license'],files=entries,
                  total_bytes=sum(x['size'] for x in entries),card_path=str(card_path.relative_to(ROOT)),
                  card_sha256=digest(card_path),metadata_path=str(meta_path.relative_to(ROOT)),metadata_sha256=digest(meta_path),
                  source_sha256=digest(Path(__file__)),protocol_sha256=digest(ROOT/'docs/decisions/0003-wikitext2-matched-baseline.md'))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'],files=len(entries),total_bytes=manifest['total_bytes'])))


if __name__=='__main__': main()
