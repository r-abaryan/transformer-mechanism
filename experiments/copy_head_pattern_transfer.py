"""Paired head-0 attention transfer with native long values and a location control."""
import csv
import json
import uuid
import torch
from copy_intact_development import trace,SNAP
from copy_learning_trajectory import ROOT,PARENT,MODELS,summarize,audit_logits
from copy_position_sites import make_model,forward,digest,write
from audit_kv_cache_equivalence import norm

OUT=ROOT/'results/copy_head_pattern_transfer'
MODES=('native','mapped','wrong_position')


def mapped_weights(short,long,mode):
    result=long['attention'].clone()
    if mode=='native':return result
    assert mode in ('mapped','wrong_position')
    result[:,0]=0;result[:,0,:6]=short['attention'][:,0,:6];result[:,0,-1]=short['attention'][:,0,-1]
    if mode=='wrong_position':result[:,0,[0,-1]]=result[:,0,[-1,0]].clone()
    return result


def audit_weights(a,short,long,mode):
    assert torch.all(a>=0) and torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-12,rtol=0)
    assert torch.equal(a[:,1],long['attention'][:,1])
    expected=long['attention'].clone()
    if mode!='native':
        expected[:,0]=0
        for src,dst in enumerate(list(range(6))+[a.shape[-1]-1]):
            if mode=='wrong_position':dst=a.shape[-1]-1 if dst==0 else 0 if dst==a.shape[-1]-1 else dst
            expected[:,0,dst]=short['attention'][:,0,src]
    assert torch.equal(a,expected)


def intervene(model,rows,short,long,mode):
    a=mapped_weights(short,long,mode);audit_weights(a,short,long,mode)
    # Native branch difference, before norm2, restricted to the selected query row.
    change=((a[:,0]-long['attention'][:,0])[...,None]*long['projected_values'][:,0]).sum(1)
    def patch(module,args):
        x=args[0].clone();x[:,-1]+=change;return (x,)
    handle=model.base.decoder.norm2.register_forward_pre_hook(patch)
    try:
        with torch.no_grad():z=forward(model,torch.tensor(rows),torch.tensor([[1]+r[:4] for r in rows]))[0][:,-1]
    finally:handle.remove()
    with torch.no_grad():
        # Rebuild both heads directly from raw long values, including output bias once.
        d=model.base.decoder;h=d.multihead_attn.num_heads;w=d.multihead_attn.embed_dim//h
        heads=[sum((a[:,j,k,None]*long['values'][:,j,k] for k in range(a.shape[-1])),torch.zeros_like(long['values'][:,j,0])) for j in range(h)]
        cross=torch.cat(heads,-1)@d.multihead_attn.out_proj.weight.T+d.multihead_attn.out_proj.bias
        x=norm(long['pre_cross']+cross,d.norm2)
        ff=(x@d.linear1.weight.T+d.linear1.bias).clamp_min(0)@d.linear2.weight.T+d.linear2.bias
        assert model.base.output.bias is None
        x=norm(x+ff,d.norm3);rz=x@model.base.output.weight.T
        error=float((z-rz).abs().max());assert error<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1))
        metrics=summarize(z,long['labels'],rows,True);audit_logits(rz,long['labels'],rows,metrics,True)
    return z,a,metrics,error


def preflight():
    folder=ROOT/'results/test_runs'/('head_transfer_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    torch.manual_seed(73);m=make_model('rope').eval();torch.save(m.state_dict(),folder/'initialized.pt');m.load_state_dict(torch.load(folder/'initialized.pt',weights_only=True))
    rows=[[3]*10,list(range(3,9))+[3]*4];shortrows=[r[:6] for r in rows];s,_=trace(m,shortrows);l,_=trace(m,rows)
    for mode in MODES:intervene(m,rows,s,l,mode)
    z,_,_,_=intervene(m,shortrows,s,s,'mapped');assert torch.allclose(z,s['logits'],atol=1e-9,rtol=0) and torch.equal(z.argmax(-1),s['logits'].argmax(-1))
    a=mapped_weights(s,l,'mapped');bad=a.clone();bad[:,0,[0,-1]]=bad[:,0,[-1,0]].clone()
    errors=0
    try:audit_weights(bad,s,l,'mapped')
    except AssertionError:errors+=1
    zz=torch.zeros(2,16,dtype=torch.float64);zz[:,2]=2;y=torch.tensor([3,3]);met=summarize(zz,y,rows,True)
    for mode in ('label','logit','missing'):
        q=zz.clone();yy=y.clone();mm=dict(met)
        if mode=='label':yy[0]=2
        elif mode=='logit':q[0,3]=3
        else:mm.pop('distinct')
        try:audit_logits(q,yy,rows,mm,True)
        except AssertionError:errors+=1
    assert errors==4
    absent=summarize(zz[:1],y[:1],rows[:1],True);audit_logits(zz[:1],y[:1],rows[:1],absent,True);assert absent['distinct'] is None
    control=mapped_weights(s,l,'wrong_position');assert torch.equal(a[:,0].sort(-1).values,control[:,0].sort(-1).values)
    return dict(passed=True,saved_initialized_evaluator=True,reference_and_scalar_checks=True,short_self_identity=True,four_intentional_errors_rejected=True,absent_groups=True,control_weight_multiset=True)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists();write(OUT/'preflight.json',preflight())
    protected=json.loads((ROOT/'results/copy_intact_development/plan.json').read_text())['protected']
    for p in ('experiments/copy_head_pattern_transfer.py','docs/protocols/copy-head-pattern-transfer.md','results/copy_head_pattern_transfer/preflight.json','results/copy_intact_development/report.json'):
        protected[p]=digest(ROOT/p)
    for seed,arm in MODELS:
        p=f'results/copy_intact_development/{seed}_{arm}_1000.pt';protected[p]=digest(ROOT/p)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'plan.json',dict(models=MODELS,step=1000,modes=MODES,selected_long_predictions_per_implementation=1536,short_self_identity_examples=512,protected=protected))
    rows=json.loads((PARENT/'data.json').read_text())['pools']['test_10'];assert len(rows)==128
    runs=[];saved={};csvrows=[];maxerror=0.
    for seed,arm in MODELS:
        model=make_model('rope').eval();model.load_state_dict(torch.load(SNAP/f'{seed}_{arm}_1000.pt',weights_only=True)['model'])
        shortrows=[r[:6] for r in rows];s,e=trace(model,shortrows);l,f=trace(model,rows);maxerror=max(maxerror,e,f)
        old=torch.load(ROOT/f'results/copy_intact_development/{seed}_{arm}_1000.pt',weights_only=True)
        for name,ev in (('short',s),('long',l)):
            err=float((ev['logits']-old[name]['logits']).abs().max());assert err<=1e-9 and torch.equal(ev['logits'].argmax(-1),old[name]['logits'].argmax(-1));maxerror=max(maxerror,err)
        z,_,_,error=intervene(model,shortrows,s,s,'mapped');assert torch.allclose(z,s['logits'],atol=1e-9,rtol=0) and torch.equal(z.argmax(-1),s['logits'].argmax(-1));maxerror=max(maxerror,error)
        metrics={};evidence={}
        for mode in MODES:
            z,a,met,error=intervene(model,rows,s,l,mode);maxerror=max(maxerror,error)
            y=l['labels'];native=l['logits'].argmax(-1);pred=z.argmax(-1)
            for group,g in met.items():
                ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r[:6]))==6)==(group=='distinct')]
                if g is not None:
                    g.update(rescues=int(((native[ids]!=y[ids])&(pred[ids]==y[ids])).sum()),harms=int(((native[ids]==y[ids])&(pred[ids]!=y[ids])).sum()))
                    csvrows.append(dict(seed=seed,arm=arm,mode=mode,group=group,**g))
            metrics[mode]=met;evidence[mode]=dict(logits=z,weights=a,labels=y)
        runs.append(dict(seed=seed,arm=arm,metrics=metrics));saved[f'{seed}_{arm}']=evidence
        print(seed,arm,{mode:m['all'] for mode,m in metrics.items()},flush=True)
    recovery=runs[0]['metrics']['mapped']['all']['exact_accuracy']>=.95
    preservation=all(run['metrics']['native'][g]['exact_accuracy']-run['metrics']['mapped'][g]['exact_accuracy']<=.05 for run in runs[1:] for g in ('all','distinct','repeated'))
    specificity=runs[0]['metrics']['mapped']['all']['exact_accuracy']-runs[0]['metrics']['wrong_position']['all']['exact_accuracy']>=.5
    dest=OUT/'evidence.pt';torch.save(saved,dest)
    for run in runs:
        for mode,e in torch.load(dest,weights_only=True)[f"{run['seed']}_{run['arm']}"].items():audit_logits(e['logits'],e['labels'],rows,run['metrics'][mode],True)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'report.json',dict(runs=runs,recovery=recovery,preservation=preservation,specificity=specificity,supported=recovery and preservation and specificity,training_updates=0,plan_sha256=digest(OUT/'plan.json'),evidence_sha256=digest(dest),audit=dict(passed=True,max_error=maxerror,argmax_differences=0,protected_files=len(protected),long_predictions_per_implementation=1536,short_identity_examples=512)))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
