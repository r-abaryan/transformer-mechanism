"""Descriptive rank/value-pattern errors of the fixed intervention."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from confirm_pair_shifts import crop_rows
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=ROOT/'results/error_strata.json'; assert not output.exists()
    priorpath=ROOT/'results/pair_shift_confirmation.json'; prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/pair_shift_confirmation_parents.json'; assert digest(splitpath)==prior['split_sha256']
    crops=crop_rows(json.loads(splitpath.read_text()))
    x=torch.tensor([r[0] for r in crops]); y=torch.tensor([r[1] for r in crops])
    rank=(x[:,:6:2]==x[:,-1,None]).long().argmax(1)
    patterns=[]
    for ids,target in crops:
        vals=ids[1:6:2]; count=len(set(vals))
        patterns.append('distinct' if count==3 else 'all_equal' if count==1 else 'target_twice' if vals.count(target)==2 else 'target_once')
    pnames=('distinct','target_once','target_twice','all_equal')
    pmasks={n:torch.tensor([p==n for p in patterns]) for n in pnames}
    groups={'all':torch.ones(len(x),dtype=torch.bool),**{f'rank_{i}':rank==i for i in range(3)},**pmasks}
    groups.update({f'rank_{i}/{p}':(rank==i)&pmasks[p] for i in range(3) for p in pnames})
    assert torch.stack(list(pmasks.values())).sum(0).eq(1).all()
    assert torch.stack([groups[f'rank_{i}/{p}'] for i in range(3) for p in pnames]).sum(0).eq(1).all()
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        group_counts={n:int(m.sum()) for n,m in groups.items()},protocol_sha256=digest(ROOT/'docs/error-strata-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('error_strata.py','confirm_pair_shifts.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(path,weights_only=True))
        conditions={}
        with torch.no_grad():
            for shift in (0,4,6,8):
                originals=[]; patched=[]
                for chunk in x.split(256):
                    base,cache=forward(model,chunk,shift); originals.append(base)
                    layer=model.blocks[0].self_attn; z=model.blocks[0].norm1(cache['r0'])
                    q,k,v=[t.reshape(len(chunk),8,2,8).transpose(1,2) for t in F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,-1)]
                    mask=torch.ones(8,8,dtype=torch.bool).triu(1)
                    w=(q@k.transpose(-2,-1)/(8**.5)).masked_fill(mask,float('-inf')).softmax(-1)
                    for row in (3,5): w[:,:,row]=0; w[:,:,row,row-1]=1
                    assert (w.sum(-1)-1).abs().max()<1e-12
                    assert w.masked_select(mask[None,None].expand_as(w)).abs().max()==0
                    a=F.linear((w@v).transpose(1,2).reshape(len(chunk),8,16),layer.out_proj.weight,layer.out_proj.bias)
                    patched.append(forward(model,chunk,shift,{'a0':a},('a0',))[0])
                before=torch.cat(originals).argmax(-1)==y
                for mode,parts in [('original',originals),('pair',patched)]:
                    scores=torch.cat(parts); assert torch.isfinite(scores).all()
                    pred=scores.argmax(-1); correct=pred==y
                    losses=F.cross_entropy(scores,y,reduction='none')
                    old=run['conditions'][f'{shift}_{mode}']
                    assert abs(correct.double().mean().item()-old['accuracy'])<1e-10
                    assert abs(losses.mean().item()-old['loss'])<1e-10
                    supplied=(pred[:,None]==x[:,1:6:2]).any(1)
                    invalid=(pred<8)|(pred>11)
                    wrong=~correct
                    categories={'supplied_wrong_value':wrong&supplied,'absent_value':wrong&~supplied&~invalid,'invalid_token':wrong&invalid}
                    assert torch.stack(list(categories.values())).sum(0).eq(wrong.long()).all()
                    stats={}
                    for name,selection in groups.items():
                        count=int(selection.sum())
                        stats[name]=dict(count=count,correct=int((correct&selection).sum()),
                            accuracy=correct[selection].double().mean().item() if count else None,
                            loss=losses[selection].mean().item() if count else None,
                            rescued=int((correct&~before&selection).sum()),harmed=int((~correct&before&selection).sum()),
                            errors={n:int((m&selection).sum()) for n,m in categories.items()})
                    conditions[f'{shift}_{mode}']=dict(groups=stats,predictions=pred.tolist())
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),conditions=conditions))
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report['group_counts']))


if __name__=='__main__': main()
