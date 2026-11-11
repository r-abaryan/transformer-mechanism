"""Standard depth/width comparison, reusing prior Adam baseline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from train_associative_recall import datasets,tensors,evaluate
from verify_decoder import TinyDecoder

ROOT=Path(__file__).resolve().parents[1]


class StackedDecoder(nn.Module):
    def __init__(self,width,hidden,depth):
        super().__init__()
        self.embedding=nn.Embedding(13,width)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,2,hidden,dropout=0,batch_first=True,norm_first=True) for _ in range(depth)])
        self.final_norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,13,bias=False)

    def forward(self,ids):
        x=self.embedding(ids)
        n,d=x.shape[1:]
        position=torch.arange(n,dtype=x.dtype)[:,None]
        frequency=10000**(-torch.arange(0,d,2,dtype=x.dtype)/d)
        x=x+torch.stack(((position*frequency).sin(),(position*frequency).cos()),-1).flatten(1)
        mask=torch.ones(n,n,dtype=torch.bool).triu(1)
        for block in self.blocks:
            x=block(x,src_mask=mask)
        return self.output(self.final_norm(x))


def verify():
    torch.manual_seed(202)
    old=TinyDecoder(13,16,2,32).double()
    new=StackedDecoder(16,32,1).double()
    new.embedding=copy.deepcopy(old.embedding)
    new.blocks[0]=copy.deepcopy(old.block)
    new.final_norm=copy.deepcopy(old.final_norm)
    new.output=copy.deepcopy(old.output)
    ids=torch.tensor([[0,8,1,9,12,0],[2,10,3,11,12,3]])
    a,b=old(ids),new(ids)
    error=(a-b).abs().max().item()
    probe=torch.randn_like(a)
    (a*probe).sum().backward(); (b*probe).sum().backward()
    gradient=max((p.grad-q.grad).abs().max().item() for p,q in zip(old.parameters(),new.parameters(),strict=True))
    deep=StackedDecoder(16,32,2).double()
    changed=ids.clone(); changed[:,-1]=7
    causal=(deep(ids)[:,:-1]-deep(changed)[:,:-1]).abs().max().item()
    assert error<1e-10 and gradient<1e-9 and causal<1e-10
    return dict(one_block_output_error=error,one_block_gradient_error=gradient,two_block_causal_error=causal)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-verified-deep',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks=verify()
    previous=json.loads((ROOT/'results/optimizer_comparison.json').read_text())
    splits=datasets()
    data={n:tensors(splits[n]) for n in ('train','validation')}
    for n in data:
        assert hashlib.sha256(json.dumps(splits[n],separators=(',',':')).encode()).hexdigest()==previous['split_hashes'][n]
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',verification=checks,runs=[],
                baseline_report_sha256=hashlib.sha256((ROOT/'results/optimizer_comparison.json').read_bytes()).hexdigest(),
                baseline=[r for r in previous['runs'] if r['recipe']=='adam'],
                protocol_sha256=hashlib.sha256((ROOT/'docs/depth-protocol.md').read_bytes()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'experiments'/n).read_bytes()).hexdigest() for n in
                               ('compare_depth.py','train_associative_recall.py','verify_decoder.py')})
    if args.resume_verified_deep:
        partial=json.loads((ROOT/'results/depth_comparison.json').read_text())
        archive=ROOT/'results/depth_run_provenance'
        assert hashlib.sha256((archive/'compare_depth_initial.py').read_bytes()).hexdigest()==partial['source_sha256']['compare_depth.py']
        assert hashlib.sha256((archive/'depth-protocol-initial.md').read_bytes()).hexdigest()==partial['protocol_sha256']
        assert len(partial['runs'])==5 and all(r['model']=='deep' for r in partial['runs'])
        assert partial['baseline_report_sha256']==report['baseline_report_sha256']
        for n in ('train_associative_recall.py','verify_decoder.py'):
            assert partial['source_sha256'][n]==report['source_sha256'][n]
        for r in partial['runs']:
            assert hashlib.sha256((ROOT/r['checkpoint']).read_bytes()).hexdigest()==r['checkpoint_sha256']
            r['source_stage']='initial'
        report['runs']=partial['runs']
        report['initial_stage']=dict(source='results/depth_run_provenance/compare_depth_initial.py',
                                    source_sha256=partial['source_sha256']['compare_depth.py'],
                                    protocol='results/depth_run_provenance/depth-protocol-initial.md',
                                    protocol_sha256=partial['protocol_sha256'],
                                    correction='Wide parameter assertion corrected from 5024 to 5152; dimensions/training unchanged')
    directory=ROOT/'results/depth_checkpoints'; directory.mkdir(exist_ok=True)
    for name,width,hidden,depth,count in [('deep',16,32,2,4896),('wide',24,40,1,5152)]:
        for seed in (11,29,47,73,101):
            if any(r['model']==name and r['seed']==seed for r in report['runs']):
                continue
            torch.manual_seed(seed)
            model=StackedDecoder(width,hidden,depth).double()
            parameters=sum(p.numel() for p in model.parameters())
            assert parameters==count,(name,parameters)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
            generator=torch.Generator().manual_seed(seed+1000)
            batch_hash=hashlib.sha256()
            curve=[dict(step=0,**{n:evaluate(model,d) for n,d in data.items()})]
            start=time.perf_counter()
            x,y=data['train']
            for step in range(1,3001):
                indices=torch.randint(len(x),(64,),generator=generator)
                batch_hash.update(json.dumps(indices.tolist(),separators=(',',':')).encode())
                optimizer.zero_grad()
                loss=F.cross_entropy(model(x[indices])[:,-1],y[indices])
                assert torch.isfinite(loss),(name,seed,step)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                if step%500==0:
                    scores={n:evaluate(model,d) for n,d in data.items()}
                    curve.append(dict(step=step,**scores))
                    print(json.dumps(dict(model=name,seed=seed,step=step,validation=scores['validation'])),flush=True)
            baseline=next(r for r in report['baseline'] if r['seed']==seed)
            assert batch_hash.hexdigest()==baseline['minibatches_sha256']
            checkpoint=directory/f'{name}_{seed}.pt'; torch.save(model.state_dict(),checkpoint)
            result=dict(model=name,width=width,hidden=hidden,depth=depth,seed=seed,parameters=parameters,source_stage='current',
                        metrics={n:evaluate(model,d) for n,d in data.items()},curve=curve,
                        seconds_including_evaluation=time.perf_counter()-start,steps=3000,examples_processed=192000,tokens_processed=1536000,
                        minibatches_sha256=batch_hash.hexdigest(),checkpoint=str(checkpoint.relative_to(ROOT)),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            report['runs'].append(result)
            (ROOT/'results/depth_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    gains=[]
    for seed in (11,29,47,73,101):
        pair={r['model']:r for r in report['runs'] if r['seed']==seed}
        gains.append(pair['deep']['metrics']['validation']['accuracy']-pair['wide']['metrics']['validation']['accuracy'])
    report['criteria']=dict(deep_all_train_validation_90=all(r['metrics'][n]['accuracy']>=.9 for r in report['runs'] if r['model']=='deep' for n in data),
                            mean_deep_minus_wide=statistics.mean(gains),sd_deep_minus_wide=statistics.stdev(gains),
                            depth_gain_at_least_10_points=statistics.mean(gains)>=.1)
    (ROOT/'results/depth_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['criteria']))


if __name__=='__main__':
    main()
