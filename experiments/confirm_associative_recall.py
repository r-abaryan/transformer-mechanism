"""Evaluate the untouched complement of the original canonical episode space."""
import hashlib
import itertools
import json
from pathlib import Path
import random

import torch
from train_associative_recall import datasets, tensors, evaluate
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    splits=datasets()
    seen={(tuple(sorted(zip(x[:-2:2],x[1:-2:2]))),x[-1]) for name in ('train','validation','test') for x,y in splits[name]}
    assert len(seen)==10240
    rows=[]
    identities=[]
    rng=random.Random(20261050)
    for keys in itertools.combinations(range(8),3):
        for values in itertools.product(range(8,12),repeat=3):
            mapping=tuple(zip(keys,values))
            for query in keys:
                identity=(mapping,query)
                if identity in seen:
                    continue
                pairs=list(mapping)
                rng.shuffle(pairs)
                rows.append(([v for pair in pairs for v in pair]+[12,query],dict(mapping)[query]))
                identities.append(identity)
    assert len(rows)==512 and len(set(identities))==512 and not set(identities)&seen
    frequency=sum(min(range(8,12),key=lambda v:(-x[1:-2:2].count(v),v))==y for x,y in rows)/len(rows)
    training=json.loads((ROOT/'results/associative_recall_training.json').read_text())
    report=dict(count=512,frequency_accuracy=frequency,oracle_accuracy=1.0,uniform_expected=.25,runs=[],
                torch=torch.__version__,device='cpu',dtype='float64',
                data_sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT/'docs/associative-confirmation-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('confirm_associative_recall.py','train_associative_recall.py','verify_decoder.py')})
    for run in training['runs']:
        checkpoint=ROOT/run['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==run['checkpoint_sha256']
        model=TinyDecoder(13,16,2,32).double()
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        result=evaluate(model,tensors(rows))
        report['runs'].append(dict(seed=run['seed'],checkpoint_sha256=run['checkpoint_sha256'],metrics=result,
                                  accuracy_minus_frequency=result['accuracy']-frequency))
    report['criteria']=dict(no_seed_at_least_90=all(r['metrics']['accuracy']<.9 for r in report['runs']),
                            no_seed_exceeds_frequency_by_5_points=all(r['accuracy_minus_frequency']<=.05 for r in report['runs']))
    (ROOT/'results/associative_recall_confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
