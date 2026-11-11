"""Pipeline and tiny-set fitting controls for the failed associative recipe."""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from train_associative_recall import datasets, tensors, evaluate
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits=datasets()
    prior=json.loads((ROOT/'results/associative_recall_training.json').read_text())
    for name,rows in splits.items():
        assert hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()==prior['split_hashes'][name]
    single=[([k,v,12,k],v) for k in range(8) for v in range(8,12)]
    small=[]
    for value in range(8,12):
        small.extend([r for r in splits['train'] if r[1]==value][:8])
    controls=dict(single_pair=single,small_three_pair=small)
    for rows in controls.values():
        assert len(rows)==32 and all(sum(y==v for x,y in rows)==8 for v in range(8,12))
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',steps=3000,lr=.1,batch=32,runs=[],
                datasets=controls,
                data_sha256={n:hashlib.sha256(json.dumps(r,separators=(',',':')).encode()).hexdigest() for n,r in controls.items()},
                protocol_sha256=hashlib.sha256((ROOT/'docs/positive-controls-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('positive_controls.py','train_associative_recall.py','verify_decoder.py')})
    directory=ROOT/'results/positive_control_checkpoints'
    directory.mkdir(exist_ok=True)
    for name,rows in controls.items():
        data=tensors(rows)
        for seed in (11,29,47,73,101):
            torch.manual_seed(seed)
            model=TinyDecoder(13,16,2,32).double()
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            curve=[dict(step=0,**evaluate(model,data))]
            start=time.perf_counter()
            x,y=data
            for step in range(1,3001):
                optimizer.zero_grad()
                loss=F.cross_entropy(model(x)[:,-1],y)
                assert torch.isfinite(loss),(name,seed,step)
                loss.backward()
                optimizer.step()
                if step%500==0:
                    curve.append(dict(step=step,**evaluate(model,data)))
            result=dict(control=name,seed=seed,fit=evaluate(model,data),curve=curve,
                        seconds_including_evaluation=time.perf_counter()-start,
                        parameters=sum(p.numel() for p in model.parameters()),examples_processed=96000,
                        tokens_processed=96000*len(rows[0][0]))
            if name=='small_three_pair':
                result['held_out_three_pair']=evaluate(model,tensors(splits['test']))
            checkpoint=directory/f'{name}_{seed}.pt'
            torch.save(model.state_dict(),checkpoint)
            result.update(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            report['runs'].append(result)
            (ROOT/'results/positive_controls.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('curve','checkpoint_sha256')}),flush=True)
    report['criteria']={n:all(r['fit']['accuracy']>=.99 for r in report['runs'] if r['control']==n) for n in controls}
    (ROOT/'results/positive_controls.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()
