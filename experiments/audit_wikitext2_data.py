"""Independent regex encoding, corpus counts and exactly-once target coverage."""
import array
import collections
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.parquet-reader'))
import pyarrow.parquet as pq
from position_distractors import digest


def main():
    rp=ROOT/'results/wikitext2_encoding.json';r=json.loads(rp.read_text())
    assert r['source_sha256']==digest(ROOT/'experiments/prepare_wikitext2.py')
    assert r['execution_protocol_sha256']==digest(ROOT/'docs/wikitext2-execution-protocol.md')
    ap=ROOT/'results/wikitext2_acquisition.json';assert digest(ap)==r['acquisition_sha256'];a=json.loads(ap.read_text())
    rows={}
    for record in a['files']:
        p=ROOT/record['path'];assert digest(p)==record['sha256']
        rows[record['split']]=pq.ParquetFile(p).read(columns=['text']).to_pydict()['text']
    trainwords={w for text in rows['train'] for w in re.findall(r'\S+',text)}
    vocabulary=json.loads((ROOT/r['vocabulary_path']).read_text(encoding='utf-8'))
    assert vocabulary==['<unk>','<eos>']+sorted(trainwords-{'<unk>','<eos>'})
    assert digest(ROOT/r['vocabulary_path'])==r['vocabulary_sha256'];indices={w:i for i,w in enumerate(vocabulary)}
    checks={}
    for split,texts in rows.items():
        expected=[];mapped=literal=words=0
        for text in texts:
            assert not text or (text[-1]=='\n' and text[:-1].find('\n')==-1)
            for word in re.findall(r'\S+',text):
                expected.append(indices.get(word,0));mapped+=int(word not in indices);literal+=int(word=='<unk>');words+=1
            expected.append(indices['<eos>'])
        stored=r['splits'][split];p=ROOT/stored['path'];assert digest(p)==stored['sha256']
        actual=array.array('I');actual.frombytes(p.read_bytes())
        if sys.byteorder!='little':actual.byteswap()
        assert actual.tolist()==expected and len(actual)==stored['tokens']
        assert words==stored['words'] and mapped==stored['mapped_unknown_words'] and literal==stored['literal_unknown_words']
        assert actual.count(1)==len(texts)==stored['eos_tokens']
        coverage=bytearray(len(actual)-1);wrong=0
        for start in range(0,len(actual)-1,64):
            count=min(64,len(actual)-1-start)
            x=actual[start:start+count];y=actual[start+1:start+count+1]
            assert len(x)==len(y)==count
            for j in range(count):
                assert y[j]==actual[start+j+1];coverage[start+j]+=1;wrong+=int(x[j]!=y[j])
        assert set(coverage)=={1} and wrong>0
        checks[split]=dict(tokens=len(actual),targets=len(coverage),mapped_unknown_words=mapped,
                          unshifted_target_control_mismatches=wrong,tail_targets=(len(actual)-1)%64)
    result=dict(passed=True,splits=checks,encoding_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    out=ROOT/'results/wikitext2_data_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
