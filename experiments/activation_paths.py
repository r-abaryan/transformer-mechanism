"""Explicit pre-norm branch replacement with library-forward controls."""
import itertools
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from compare_depth import StackedDecoder
from position_distractors import digest, logits, pe

ROOT=Path(__file__).resolve().parents[1]


def forward(model,ids,shift,donor=None,patch=()):
    x=model.embedding(ids)+pe(torch.arange(ids.shape[1])+shift,16)
    mask=torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool).triu(1)
    cache={}
    for i,block in enumerate(model.blocks):
        z=block.norm1(x)
        a=block.self_attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        cache[f'r{i}']=x
        cache[f'a{i}']=a
        residual=donor[f'r{i}'] if f'r{i}' in patch else x
        a=donor[f'a{i}'] if f'a{i}' in patch else a
        u=residual+a
        f=block.linear2(block.activation(block.linear1(block.norm2(u))))
        cache[f'f{i}']=f
        f=donor[f'f{i}'] if f'f{i}' in patch else f
        x=u+f
    return model.output(model.final_norm(x))[:,-1],cache


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/activation_paths.json'
    assert not output.exists()
    priorpath=ROOT/'results/position_shifts.json'
    prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'
    assert digest(splitpath)==prior['split_sha256']
    rows=json.loads(splitpath.read_text())['six_pairs']
    crops,labels=[],[]
    for ids,target in rows:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            kept=sorted((q,*others))
            crop=[ids[p] for i in kept for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    x=torch.tensor(crops); y=torch.tensor(labels)
    patches={key:(key,) for key in ('r0','a0','f0','r1','a1','f1')}
    patches.update(both_residual=('r0','r1'),both_attention=('a0','a1'),both_ffn=('f0','f1'))
    allkeys=('r0','a0','f0','r1','a1','f1')
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
                protocol_sha256=digest(ROOT/'docs/activation-protocol.md'),
                source_sha256={n:digest(ROOT/'experiments'/n) for n in ('activation_paths.py','position_distractors.py','compare_depth.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']
        assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(path,weights_only=True))
        assert all(b.dropout.p==0 and b.dropout1.p==0 and b.dropout2.p==0 for b in model.blocks)
        pieces={k:[] for k in ('compact','shifted',*patches)}
        errors=dict(compact=0.,shifted=0.,sham=0.,full_donor=0.)
        with torch.no_grad():
            for chunk in x.split(256):
                c,donor=forward(model,chunk,0)
                s,shifted=forward(model,chunk,4)
                checks=dict(compact=(c,logits(model,chunk)),shifted=(s,logits(model,chunk,(torch.arange(8)+4).expand_as(chunk))),
                            sham=(s,forward(model,chunk,4,shifted,allkeys)[0]),
                            full_donor=(c,forward(model,chunk,4,donor,allkeys)[0]))
                for name,(a,b) in checks.items():
                    error=(a-b).abs().max().item()
                    errors[name]=max(errors[name],error)
                    assert error<1e-10
                pieces['compact'].append(c); pieces['shifted'].append(s)
                for name,patch in patches.items(): pieces[name].append(forward(model,chunk,4,donor,patch)[0])
        scores={k:torch.cat(v) for k,v in pieces.items()}
        correct=scores['shifted'].argmax(-1)==y
        conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                                 rescued=int((now&~correct).sum()),harmed=int((~now&correct).sum()))
        for name,oldkey in [('compact','global_0'),('shifted','global_4')]:
            for metric in ('accuracy','loss'):
                assert abs(conditions[name][metric]-run['lengths']['six_pairs'][oldkey][metric])<1e-12
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),errors=errors,conditions=conditions))
    report['all_seeds_at_least_90']={name:all(r['conditions'][name]['accuracy']>=.9 for r in report['runs']) for name in patches}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['all_seeds_at_least_90']))


if __name__=='__main__':
    main()
