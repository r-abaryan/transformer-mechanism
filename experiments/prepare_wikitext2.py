"""Freeze row-preserving, training-vocabulary word streams and inspect boundaries."""
import array
import collections
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.parquet-reader'))
import pyarrow
import pyarrow.parquet as pq
from position_distractors import digest


def read_rows(path):
    table=pq.read_table(path,columns=['text'])
    assert table.column_names==['text'] and table.column('text').null_count==0
    return table.column('text').to_pylist()


def main():
    output=ROOT/'results/wikitext2_encoding.json';assert not output.exists()
    acquisition=json.loads((ROOT/'results/wikitext2_acquisition.json').read_text())
    assert digest(ROOT/'docs/decisions/0003-wikitext2-matched-baseline.md')==acquisition['protocol_sha256']
    rows={}
    for f in acquisition['files']:
        path=ROOT/f['path'];assert digest(path)==f['sha256']
        rows[f['split']]=read_rows(path);assert len(rows[f['split']])==f['publisher_rows']
        assert all(t=='' or (t.endswith('\n') and t.count('\n')==1) for t in rows[f['split']])
    vocabulary=['<unk>','<eos>']+sorted({w for t in rows['train'] for w in t.split()}-{'<unk>','<eos>'})
    lookup={w:i for i,w in enumerate(vocabulary)};assert len(vocabulary)==33278
    folder=ROOT/'data/wikitext-2-v1';vp=folder/'vocabulary.json';assert not vp.exists()
    vp.write_text(json.dumps(vocabulary,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    splits={};expected=dict(train=2088628,validation=217646,test=245569)
    for split,texts in rows.items():
        tokens=[];mapped=literal=words=0
        for text in texts:
            for w in text.split():
                mapped+=int(w not in lookup);literal+=int(w=='<unk>');words+=1;tokens.append(lookup.get(w,0))
            tokens.append(1)
        assert len(tokens)==expected[split]
        encoded=array.array('I',tokens);assert encoded.itemsize==4
        if sys.byteorder!='little':encoded.byteswap()
        path=folder/f'{split}.uint32';assert not path.exists();path.write_bytes(encoded.tobytes())
        complete,tail=divmod(len(tokens)-1,64)
        splits[split]=dict(rows=len(texts),empty_rows=sum(t=='' for t in texts),words=words,tokens=len(tokens),
                           mapped_unknown_words=mapped,literal_unknown_words=literal,unknown_word_rate=(mapped+literal)/words,
                           eos_tokens=len(texts),targets=len(tokens)-1,complete_blocks=complete,tail_targets=tail,
                           path=str(path.relative_to(ROOT)),sha256=digest(path))
    result=dict(passed=True,pyarrow=pyarrow.__version__,vocabulary=len(vocabulary),vocabulary_path=str(vp.relative_to(ROOT)),
                vocabulary_sha256=digest(vp),splits=splits,acquisition_sha256=digest(ROOT/'results/wikitext2_acquisition.json'),
                source_sha256=digest(Path(__file__)),execution_protocol_sha256=digest(ROOT/'docs/wikitext2-execution-protocol.md'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
