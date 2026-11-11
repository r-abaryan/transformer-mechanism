"""Fixed final-query score interventions on archived six-pair diagnostics."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from position_distractors import digest, logits
from qkv_paths import project

ROOT=Path(__file__).resolve().parents[1]
NAMES=('ordinary','sharpen2','target_floor','wrong_slot_floor')


def intervene(scores,selected,name):
    a=(scores*2 if name=='sharpen2' else scores).softmax(-1)
    if name.endswith('floor'):
        index=selected[:,None,None].expand(-1,2,1)
        p=a.gather(-1,index); desired=p.clamp_min(.9)
        a=a*((1-desired)/(1-p))
        a.scatter_(-1,index,desired)
    assert torch.isfinite(a).all() and (a>=0).all() and (a.sum(-1)-1).abs().max()<1e-12
    return a


def metrics(scores,y,baseline):
    now=scores.argmax(-1)==y; old=baseline.argmax(-1)==y
    return dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(scores,y).item(),
                rescued=int((now&~old).sum()),harmed=int((old&~now).sum()))


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rp=ROOT/'results/trained_competition.json'; tp=ROOT/'results/trained_competition_tensors.pt'
    assert not rp.exists() and not tp.exists()
    pp=ROOT/'results/depth_confirmation.json'; prior=json.loads(pp.read_text())
    sp=ROOT/prior['split_file']; assert digest(sp)==prior['split_sha256']
    rows=json.loads(sp.read_text())['six_pairs']
    x=torch.tensor([a for a,b in rows]); y=torch.tensor([b for a,b in rows]); assert len(x)==1024
    rank=(x[:,:-2:2]==x[:,-1,None]).long().argmax(-1); target=2*rank+1; wrong=2*((rank+1)%6)+1
    assert torch.equal(x.gather(1,target[:,None]).squeeze(1),y)
    paths=['experiments/trained_competition.py','experiments/activation_paths.py','experiments/compare_depth.py','experiments/position_distractors.py','experiments/qkv_paths.py','docs/trained-competition-protocol.md']
    r=dict(hashes={p:digest(ROOT/p) for p in paths},prior_sha256=digest(pp),split_file=prior['split_file'],split_sha256=digest(sp),
           torch=torch.__version__,device='cpu',dtype='float64',control_same_answer=int((x.gather(1,wrong[:,None]).squeeze(1)==y).sum()),runs=[])
    saved=dict(ids=x,labels=y,target=target,wrong=wrong,runs=[])
    with torch.no_grad():
        for old in prior['runs']:
            if old['model']!='deep': continue
            cp=ROOT/old['checkpoint']; assert digest(cp)==old['checkpoint_sha256']
            model=StackedDecoder(old['width'],old['hidden'],old['depth']).double().eval()
            model.load_state_dict(torch.load(cp,weights_only=True))
            pieces={n:[] for n in NAMES}; residues=[]; ss=[]; vv=[]; control=0.
            for start in range(0,len(x),128):
                ids=x[start:start+128]; b,n=ids.shape
                base,cache=forward(model,ids,0)
                control=max(control,(base-logits(model,ids)).abs().max().item())
                q,k,v=project(model,cache['r1'])
                q,k,v=[a.reshape(b,n,2,8).transpose(1,2) for a in (q,k,v)]
                score=(q[:,:,-1,None]@k.transpose(-2,-1)).squeeze(-2)/math.sqrt(8)
                values=v.clone(); residues.append(cache['r1'][:,-1]); ss.append(score); vv.append(v)
                for name in NAMES:
                    selected=(wrong if name=='wrong_slot_floor' else target)[start:start+b]
                    a=intervene(score,selected,name)
                    heads=(a[:,:,None,:]@v).squeeze(-2).reshape(b,16)
                    out=model.blocks[1].self_attn.out_proj
                    patch=cache['a1'].clone(); patch[:,-1]=F.linear(heads,out.weight,out.bias)
                    scores=forward(model,ids,0,{'a1':patch},('a1',))[0]
                    if name=='ordinary': control=max(control,(scores-base).abs().max().item())
                    # Independently traverse the library model with only the final row replaced.
                    def hook(module,args,result):
                        output=result[0].clone(); output[:,-1]=patch[:,-1]
                        return output,result[1]
                    handle=model.blocks[1].self_attn.register_forward_hook(hook)
                    try: library=model(ids)[:,-1]
                    finally: handle.remove()
                    control=max(control,(scores-library).abs().max().item())
                    assert torch.equal(v,values) and torch.isfinite(scores).all()
                    pieces[name].append(scores)
            assert control<1e-10 and digest(cp)==old['checkpoint_sha256']
            outputs={k:torch.cat(v) for k,v in pieces.items()}; scores=torch.cat(ss)
            a=scores.softmax(-1); mass=a.gather(-1,target[:,None,None].expand(-1,2,1)).squeeze(-1)
            top=scores.argmax(-1)==target[:,None]; error=outputs['ordinary'].argmax(-1)!=y
            conditions={name:metrics(z,y,outputs['ordinary']) for name,z in outputs.items()}
            for key in ('accuracy','loss'): assert abs(conditions['ordinary'][key]-old['metrics']['six_pairs'][key])<1e-10
            strata=[]
            for label,mask in [('both_heads_target_top',top.all(-1)),('other',~top.all(-1))]:
                strata.append(dict(name=label,count=int(mask.sum()),original_errors=int((mask&error).sum()),
                    rescued={name:int((mask&error&(z.argmax(-1)==y)).sum()) for name,z in outputs.items()}))
            run=dict(seed=old['seed'],checkpoint=old['checkpoint'],checkpoint_sha256=digest(cp),max_control_error=control,
                     conditions=conditions,target_mass_by_head=mass.mean(0).tolist(),target_top_by_head=top.double().mean(0).tolist(),strata=strata)
            r['runs'].append(run); saved['runs'].append(dict(seed=old['seed'],residual=torch.cat(residues),scores=scores,values=torch.cat(vv),outputs=outputs))
            print(json.dumps(run),flush=True)
    assert len(r['runs'])==5
    gains=[run['conditions']['sharpen2']['accuracy']-run['conditions']['ordinary']['accuracy'] for run in r['runs']]
    r['sharpen2_keep']=sum(gains)/5>=.05 and min(gains)>0
    torch.save(saved,tp); r.update(complete=True,tensors_sha256=digest(tp))
    rp.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__': main()
