"""Matched internal pre/post LayerNorm placement crossed with warmup."""
import hashlib
import itertools
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from reserved_cycle import model_new,evaluate
from position_training import shifted,weight_hash,metrics
from position_distractors import digest,pe,logits

ROOT=Path(__file__).resolve().parents[1]


def model_for(placement):
    model=model_new()
    for block in model.blocks: block.norm_first=(placement=='pre')
    return model


def explicit(model,ids):
    x=model.embedding(ids)+pe(torch.arange(ids.shape[1]),16); mask=torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool).triu(1)
    for b in model.blocks:
        if b.norm_first:
            z=b.norm1(x); x=x+b.self_attn(z,z,z,attn_mask=mask,need_weights=False)[0]
            x=x+b.linear2(b.activation(b.linear1(b.norm2(x))))
        else:
            x=b.norm1(x+b.self_attn(x,x,x,attn_mask=mask,need_weights=False)[0])
            x=b.norm2(x+b.linear2(b.activation(b.linear1(x))))
    return model.output(model.final_norm(x))[:,-1]


def probe(model,x,y):
    loss=F.cross_entropy(model(x)[:,-1],y); named=list(model.named_parameters()); grads=torch.autograd.grad(loss,[p for n,p in named]); groups={}
    for (name,p),g in zip(named,grads):
        group='.'.join(name.split('.')[:2]) if name.startswith('blocks.') else name.split('.')[0]
        groups.setdefault(group,[]).append(g.flatten())
    return dict(loss=loss.item(),parameter_gradient_rms={n:torch.cat(gs).square().mean().sqrt().item() for n,gs in groups.items()})


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'results/norm_placement.json'; sp=ROOT/'results/norm_placement_splits.json'; dp=ROOT/'results/norm_placement_test_scores.json'
    assert not any(p.exists() for p in (out,sp,dp))
    excluded=set(); exclusions={}
    def identity(ids): return tuple(sorted(zip(ids[:-2:2],ids[1:-2:2])))
    for name in ('reserved_cycle_splits.json','position_training_splits.json','joint_routing_confirmation_inputs.json','canonical_positions_inputs.json','final_length_inputs.json'):
        p=ROOT/'results'/name; exclusions[name]=digest(p); d=json.loads(p.read_text())
        groups=d.values() if isinstance(d,dict) else [d]
        for rows in groups: excluded.update(identity(x) for x,y in rows if len(x)==8)
    assert len(excluded)==6020
    rng=random.Random(20261228); maps={n:[] for n in ('train','validation','test')}
    for values in itertools.combinations(range(12,18),3):
        candidates=[tuple(zip(ks,vs)) for ks in itertools.combinations(range(12),3) for vs in itertools.permutations(values) if tuple(zip(ks,vs)) not in excluded]
        rng.shuffle(candidates)
        for n,a,b in [('train',0,80),('validation',80,100),('test',100,120)]: maps[n].extend(candidates[a:b])
    assert len(set.union(*(set(v) for v in maps.values())))==2400
    splits={n:[([t for pair in order for t in pair]+[18,q],v) for m in ms for order in itertools.permutations(m) for q,v in m] for n,ms in maps.items()}
    for rows in splits.values():
        assert all(dict(identity(x))[x[-1]]==y for x,y in rows)
        assert all(sum(y==v for x,y in rows)==len(rows)//6 for v in range(12,18))
        assert all(sum(x[:6:2].index(x[-1])==i for x,y in rows)==len(rows)//3 for i in range(3))
    sp.write_text(json.dumps(splits,separators=(',',':'))+'\n')
    data={n:(torch.tensor([x for x,y in rows]),torch.tensor([y for x,y in rows])) for n,rows in splits.items()}
    indices=torch.randperm(7200,generator=torch.Generator().manual_seed(20261229))[:128]
    px,py=(t[indices] for t in data['validation'])
    report=dict(phase='training',torch=torch.__version__,device='cpu',dtype='float64',split_sha256=digest(sp),exclusion_hashes=exclusions,probe_indices=indices.tolist(),
                protocol_sha256=digest(ROOT/'docs/norm-placement-protocol.md'),source_sha256={n:digest(ROOT/'experiments'/n) for n in ('norm_placement.py','reserved_cycle.py','compare_depth.py','position_training.py','position_distractors.py')},verification={},runs=[])
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for placement in ('pre','post'):
        torch.manual_seed(601); model=model_for(placement); assert sum(p.numel() for p in model.parameters())==5088
        a=model(px[:8])[:,-1]; b=explicit(model,px[:8]); ps=list(model.parameters())
        ga=torch.autograd.grad(F.cross_entropy(a,py[:8]),ps); gb=torch.autograd.grad(F.cross_entropy(b,py[:8]),ps)
        err=(a-b).abs().max().item(); ge=max((x-y).abs().max().item() for x,y in zip(ga,gb)); assert max(err,ge)<1e-10
        report['verification'][placement]=dict(forward_error=err,gradient_error=ge)
    directory=ROOT/'results/norm_placement_checkpoints'; directory.mkdir(exist_ok=True)
    for seed in (601,631,659,683,719):
        for placement in ('pre','post'):
            for schedule in ('constant','warmup'):
                torch.manual_seed(seed); model=model_for(placement); initial=weight_hash(model)
                opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
                batches=torch.Generator().manual_seed(seed+1000); offsets=random.Random(seed+2000); bh=hashlib.sha256(); oh=hashlib.sha256()
                run=dict(seed=seed,placement=placement,schedule=schedule,initial_sha256=initial,curve=[],probes=[],status='running'); report['runs'].append(run)
                run['curve'].append(dict(step=0,**evaluate(model,data['validation']))); run['probes'].append(dict(step=0,**probe(model,px,py)))
                x,y=data['train']; max_grad=0.; rate_sum=0.
                for step in range(1,3001):
                    idx=torch.randint(len(x),(64,),generator=batches); bh.update(json.dumps(idx.tolist()).encode()); shift=offsets.randrange(5); oh.update(bytes([shift]))
                    rate=.001*min(step/300,1) if schedule=='warmup' else .001; opt.param_groups[0]['lr']=rate; rate_sum+=rate
                    opt.zero_grad(); loss=F.cross_entropy(shifted(model,x[idx],shift),y[idx])
                    if not torch.isfinite(loss): run.update(status='failed',failure_step=step,failure='nonfinite loss'); break
                    loss.backward()
                    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): run.update(status='failed',failure_step=step,failure='nonfinite gradient'); break
                    max_grad=max(max_grad,max(p.grad.abs().max().item() for p in model.parameters())); opt.step()
                    if step%500==0:
                        val=evaluate(model,data['validation']); run['curve'].append(dict(step=step,**val))
                        if step in (500,1500,3000): run['probes'].append(dict(step=step,**probe(model,px,py)))
                        save(); print(json.dumps(dict(seed=seed,placement=placement,schedule=schedule,step=step,validation=val)),flush=True)
                run.update(minibatch_sha256=bh.hexdigest(),offset_sha256=oh.hexdigest(),max_training_gradient=max_grad,learning_rate_sum=rate_sum)
                if run['status']!='failed':
                    path=directory/f'{placement}_{schedule}_{seed}.pt'; assert not path.exists(); torch.save(model.state_dict(),path)
                    run.update(status='trained',checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path),steps=3000)
                run['first_observed_validation_99']=next((c['step'] for c in run['curve'] if c['accuracy']>=.99),None)
                siblings=[r for r in report['runs'] if r['seed']==seed]; assert len({r['initial_sha256'] for r in siblings})==1
                finished=[r for r in siblings if r['status']=='trained']; assert len({r['minibatch_sha256'] for r in finished})<=1 and len({r['offset_sha256'] for r in finished})<=1
                save()
    assert len(report['runs'])==20; report['phase']='all_training_attempts_finished_before_test'; save()
    details=[]; x,y=data['test']
    for run in report['runs']:
        if run['status']!='trained': continue
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=model_for(run['placement']).eval(); model.load_state_dict(torch.load(path,weights_only=True)); s=logits(model,x)
        run['test']=metrics(s,y); details.append(dict(seed=run['seed'],placement=run['placement'],schedule=run['schedule'],scores=s.tolist())); save()
    report['condition_all_99']={f'{p}_{w}':all(r['status']=='trained' and r['test']['accuracy']>=.99 for r in report['runs'] if r['placement']==p and r['schedule']==w) for p in ('pre','post') for w in ('constant','warmup')}
    dp.write_text(json.dumps(details,separators=(',',':'))+'\n'); report.update(scores_sha256=digest(dp),phase='complete'); save(); print(json.dumps(report['condition_all_99']))


if __name__=='__main__': main()
