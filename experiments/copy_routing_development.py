"""Conditional effects of historical Q/K updates, without retraining."""
import csv
import json
import uuid

import torch
from copy_learning_trajectory import (ROOT, PARENT, MODELS, summarize, audit_logits, equal)
from copy_position_sites import make_model, forward, inputs, digest, write

OUT=ROOT/'results/copy_routing_development'
SNAP=ROOT/'results/copy_learning_trajectory'
PACKED=('base.decoder.multihead_attn.in_proj_weight','base.decoder.multihead_attn.in_proj_bias')
CORNERS=((0,0),(1,0),(0,1),(1,1))


def hybrid(qk,rest):
    assert set(qk)==set(rest)
    state={k:v.clone() for k,v in rest.items()}
    for k in PACKED:
        assert state[k].shape[0]==24
        state[k][:16]=qk[k][:16]
    return state


def audit_scope(state,qk,rest):
    # Independent chunk assembly checks every packed projection and all other keys.
    assert set(state)==set(qk)==set(rest)
    for k,v in state.items():
        expected=torch.cat((*qk[k].chunk(3)[:2],rest[k].chunk(3)[2])) if k in PACKED else rest[k]
        assert torch.equal(v,expected),k


def evaluate(state,data):
    model=make_model('rope').eval();model.load_state_dict(state);saved={};metrics={};error=0.
    rows=data['pools']['test_10'];prefix=torch.tensor([[1]+r[:4] for r in rows]);y=torch.tensor([r[4] for r in rows])
    cases={f'paired_{n}':(torch.tensor([r[:n] for r in rows]),prefix,y,rows,True) for n in (6,10)}
    for name,rr in data['pools'].items():
        if name.startswith(('train_','validation_','test_')) and len(rr[0])<=6:
            x=torch.tensor(rr);p,labels=inputs(x)
            assert torch.equal(labels,torch.tensor([r+[2] for r in rr])) and torch.equal(p,torch.tensor([[1]+r for r in rr]))
            cases[name]=(x,p,labels,rr,False)
    with torch.no_grad():
        for name,(x,p,y,rr,paired) in cases.items():
            z=forward(model,x,p)[0];rz=forward(model,x,p,reference=True)[0]
            if paired:z=z[:,-1];rz=rz[:,-1]
            error=max(error,float((z-rz).abs().max()));assert error<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1))
            metrics[name]=summarize(z,y,rr,paired);audit_logits(rz,y,rr,metrics[name],paired);saved[name]=dict(logits=z,labels=y)
    return metrics,saved,error


def differences(saved):
    ds={}
    for corner,e in saved.items():
        y=e['paired_6']['labels'];a=e['paired_6']['logits'];b=e['paired_10']['logits']
        ds[corner]=(b[:,2]-b.gather(1,y[:,None]).squeeze(1))-(a[:,2]-a.gather(1,y[:,None]).squeeze(1))
    vectors={'native_increase':ds['11']-ds['00'],'qk_early_rest':ds['10']-ds['00'],
        'qk_late_rest':ds['11']-ds['01'],'rest_early_qk':ds['01']-ds['00'],
        'rest_late_qk':ds['11']-ds['10'],'interaction':ds['11']-ds['10']-ds['01']+ds['00']}
    assert torch.allclose(vectors['native_increase'],vectors['qk_early_rest']+vectors['rest_late_qk'],atol=1e-12,rtol=1e-12)
    return {name:dict(mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()),positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum())) for name,v in vectors.items()}


def preflight():
    folder=ROOT/'results/test_runs'/('copy_qk_development_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    torch.manual_seed(89);a=make_model('rope').state_dict();b={k:v+torch.randn_like(v)*.01 for k,v in a.items()}
    fixture=folder/'donors.pt';torch.save([a,b],fixture);a,b=torch.load(fixture,weights_only=True)
    assert equal(hybrid(a,a),a) and equal(hybrid(b,b),b)
    state=hybrid(b,a);audit_scope(state,b,a);wrong={k:v.clone() for k,v in state.items()};wrong[PACKED[0]][16,0]+=1
    try:audit_scope(wrong,b,a)
    except AssertionError:pass
    else:raise AssertionError('V-block corruption accepted')
    data=dict(pools={'test_10':[[3]*10,list(range(3,9))+[3]*4],'train_3':[[3]*3,[3,4,5]]})
    metrics,evidence,_=evaluate(state,data);torch.save(evidence,folder/'logits.pt')
    for name,e in torch.load(folder/'logits.pt',weights_only=True).items():
        audit_logits(e['logits'],e['labels'],data['pools']['test_10' if name.startswith('paired_') else name],metrics[name],name.startswith('paired_'))
    z=torch.zeros(2,16,dtype=torch.float64);z[:,2]=1;y=torch.tensor([3,3]);rows=data['pools']['test_10'];mm=summarize(z,y,rows,True)
    errors=0
    for mode in ('logit','label','missing'):
        zz=z.clone();yy=y.clone();met=dict(mm)
        if mode=='logit':zz[0,3]=2
        elif mode=='label':yy[0]=2
        else:met.pop('distinct')
        try:audit_logits(zz,yy,rows,met,True)
        except AssertionError:errors+=1
    assert errors==3
    z[:,3]=1;mm=summarize(z,y,rows,True);audit_logits(z,y,rows,mm,True);assert mm['all']['eos']==2
    absent=summarize(z[:1],y[:1],rows[:1],True);audit_logits(z[:1],y[:1],rows[:1],absent,True);assert absent['distinct'] is None
    null={str(i)+str(j):evidence for i,j in CORNERS};assert all(v['mean']==0 for v in differences(null).values())
    return dict(passed=True,initialized_saved_evaluator_and_audit=True,scope_and_v_error_control=True,three_metric_errors_rejected=True,ties_absent_groups=True,identical_donor_null=True)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    write(OUT/'preflight.json',preflight());data=json.loads((PARENT/'data.json').read_text())
    protected=json.loads((SNAP/'plan.json').read_text())['protected']
    paths=['experiments/copy_routing_development.py','docs/protocols/copy-routing-development.md','results/copy_routing_development/preflight.json','results/copy_learning_trajectory/report.json','results/copy_learning_trajectory/persisted-audit.json']
    for seed,arm in MODELS:
        for step in (500,1000):
            paths += [f'results/copy_learning_trajectory/{seed}_{arm}_{step}.pt',f'results/copy_learning_trajectory/{seed}_{arm}_{step}_logits.pt']
    for p in paths:protected[p]=digest(ROOT/p)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    counts={name:len(rr)*(len(rr[0])+1) for name,rr in data['pools'].items() if name.startswith(('train_','validation_','test_')) and len(rr[0])<=6}
    write(OUT/'plan.json',dict(models=MODELS,corners=CORNERS,steps=[500,1000],paired_predictions_per_implementation=4096,short_targets_per_corner=counts,short_predictions_per_implementation=16*sum(counts.values()),protected=protected))
    runs=[];maxerror=0.;csvrows=[];artifact_hashes={}
    for seed,arm in MODELS:
        donors=[torch.load(SNAP/f'{seed}_{arm}_{s}.pt',weights_only=True)['model'] for s in (500,1000)];saved={};corners={}
        for q,r in CORNERS:
            corner=f'{q}{r}';state=hybrid(donors[q],donors[r]);audit_scope(state,donors[q],donors[r])
            if q==r:assert equal(state,donors[q])
            met,ev,error=evaluate(state,data);maxerror=max(maxerror,error)
            if q==r:
                old=torch.load(SNAP/f'{seed}_{arm}_{(500,1000)[q]}_logits.pt',weights_only=True);assert set(ev)==set(old)
                for name,e in ev.items():
                    assert torch.equal(e['labels'],old[name]['labels'])
                    err=float((e['logits']-old[name]['logits']).abs().max());assert err<=1e-9 and torch.equal(e['logits'].argmax(-1),old[name]['logits'].argmax(-1));maxerror=max(maxerror,err)
            dest=OUT/f'{seed}_{arm}_{corner}.pt';torch.save(ev,dest);loaded=torch.load(dest,weights_only=True)
            for name,e in loaded.items():
                rr=data['pools']['test_10' if name.startswith('paired_') else name];paired=name.startswith('paired_');audit_logits(e['logits'],e['labels'],rr,met[name],paired)
                for group,m in met[name].items():
                    csvrows.append(dict(seed=seed,arm=arm,corner=corner,pool=name,group=group,**{k:('' if m is None else m.get(k,'')) for k in ('count','token_loss','token_accuracy','exact_accuracy','correct','eos','margin')}))
            corners[corner]=met;saved[corner]=ev;artifact_hashes[dest.name]=digest(dest)
        effects=differences(saved);delta=effects['native_increase']['mean'];preservation={};answers={}
        for c,base in (('10','00'),('01','11')):
            preservation[c]=all(corners[base]['paired_6'][g]['exact_accuracy']-corners[c]['paired_6'][g]['exact_accuracy']<=.05 for g in ('all','distinct','repeated') if corners[base]['paired_6'][g])
            answers[c]={}
            for length in (6,10):
                name=f'paired_{length}';y=saved[c][name]['labels'];p=saved[base][name]['logits'].argmax(-1);h=saved[c][name]['logits'].argmax(-1)
                answers[c][name]=dict(rescues=int(((p!=y)&(h==y)).sum()),harms=int(((p==y)&(h!=y)).sum()))
        carrying=delta>0 and all(effects[k]['mean']>=.9*delta for k in ('qk_early_rest','qk_late_rest'))
        run=dict(seed=seed,arm=arm,corners=corners,effects=effects,short_preservation=preservation,answers=answers,carrying_most_in_both_contexts=carrying,supported=carrying and all(preservation.values()))
        runs.append(run);print(json.dumps(dict(seed=seed,arm=arm,effects=effects,short_preservation=preservation,supported=run['supported'])),flush=True)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'report.json',dict(runs=runs,weak_support=runs[0]['supported'],training_updates=0,plan_sha256=digest(OUT/'plan.json'),artifact_sha256=artifact_hashes,audit=dict(passed=True,max_logit_error=maxerror,producer_reference_argmax_differences=0,protected_files=len(protected),corners=16,csv_rows=len(csvrows))))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
