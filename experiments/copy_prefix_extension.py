"""Paired source extension with fixed prefix and label; no training."""
import json
import uuid
import torch
from copy_position_sites import ROOT,forward,make_model,digest,write
from copy_site_failure import MODELS,PARENT
from copy_end_route import reference

OUT=ROOT/'results/copy_prefix_extension'


def validate(rows,short,prefix,labels,t):
    assert short.tolist()==[r[:6] for r in rows]
    assert prefix.tolist()==[[1]+r[:t-1] for r in rows]
    assert labels.tolist()==[r[t-1] for r in rows]


def summarize(a,b,y,rows):
    out={}
    for g in ('all','distinct_prefix','repeated_prefix'):
        ids=[i for i,r in enumerate(rows) if g=='all' or (len(set(r[:6]))==6)==(g=='distinct_prefix')]
        if not ids:out[g]=None;continue
        x,z,labels=a[ids],b[ids],y[ids];pa,pb=x.argmax(-1),z.argmax(-1)
        ma=x[:,2]-x.gather(1,labels[:,None]).squeeze(1);mb=z[:,2]-z.gather(1,labels[:,None]).squeeze(1)
        met=dict(count=len(ids),short_correct=int((pa==labels).sum()),long_correct=int((pb==labels).sum()),short_eos=int((pa==2).sum()),long_eos=int((pb==2).sum()),correct_to_eos=int(((pa==labels)&(pb==2)).sum()),eos_to_correct=int(((pa==2)&(pb==labels)).sum()),positive_margin_changes=int((mb-ma>1e-10).sum()),short_margin=float(ma.mean()),long_margin=float(mb.mean()),mean_margin_change=float((mb-ma).mean()))
        aa=x.tolist();bb=z.tolist();yy=labels.tolist()
        sa=[max(range(len(v)),key=v.__getitem__) for v in aa];sb=[max(range(len(v)),key=v.__getitem__) for v in bb]
        assert met['correct_to_eos']==sum(p==v and q==2 for p,q,v in zip(sa,sb,yy))
        assert met['eos_to_correct']==sum(p==2 and q==v for p,q,v in zip(sa,sb,yy))
        deltas=[(q[2]-q[v])-(p[2]-p[v]) for p,q,v in zip(aa,bb,yy)]
        assert abs(sum(deltas)/len(ids)-met['mean_margin_change'])<1e-9
        out[g]=met
    return out


def evaluate(model,rows,t):
    x=torch.tensor(rows);short=x[:,:6];prefix=torch.tensor([[1]+r[:t-1] for r in rows]);y=x[:,t-1]
    validate(rows,short,prefix,y,t);zs=[];error=0.
    with torch.no_grad():
        for source in (short,x):
            z=forward(model,source,prefix)[0][:,-1];rz=reference(model,source,prefix,'native')[:,-1]
            err=float((z-rz).abs().max());assert err<=1e-9 and torch.equal(z.argmax(-1),rz.argmax(-1));error=max(error,err);zs.append(z)
    return dict(short=zs[0],long=zs[1],labels=y),summarize(*zs,y,rows),error


def preflight():
    rows=[[3,4,5,6,7,8,9,10,3,4],[4]*10]
    torch.manual_seed(83);model=make_model('rope').eval();folder=ROOT/'results/test_runs'/('prefix_extension_'+uuid.uuid4().hex);folder.mkdir(parents=True)
    p=folder/'initialized.pt';torch.save(model.state_dict(),p);model.load_state_dict(torch.load(p,weights_only=True))
    for t in (4,5):
        saved,m,err=evaluate(model,rows,t)
        null=summarize(saved['short'],saved['short'],saved['labels'],rows)
        assert null['all']['mean_margin_change']==0 and null['all']['correct_to_eos']==0
    bad=torch.tensor([1,1])
    try:validate(rows,torch.tensor(rows)[:,:6],torch.tensor([[1]+r[:3] for r in rows]),bad,4)
    except AssertionError:pass
    else:raise AssertionError('Label fault undetected')
    a=torch.zeros(2,16,dtype=torch.float64);a[:,3]=2;b=a.clone();b[0,2]=3;y=torch.tensor([3,3])
    m=summarize(a,b,y,rows);assert m['all']['correct_to_eos']==1
    b[0,2]=2;assert summarize(a,b,y,rows)['all']['long_eos']==1
    assert summarize(a[1:],b[1:],y[1:],rows[1:])['distinct_prefix'] is None
    return dict(passed=True,initialized_saved_evaluator=True,reference_cases=4,wrong_label_rejected=True,null_and_tie_and_absent_groups=True)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists();write(OUT/'preflight.json',preflight())
    parent=json.loads((PARENT/'report.json').read_text());data=json.loads((PARENT/'data.json').read_text())['pools'];rows=data['test_10'];assert len(rows)==128
    protected=json.loads((ROOT/'results/copy_end_route/plan.json').read_text())['protected']
    for p in (ROOT/'experiments/copy_end_route.py',ROOT/'docs/protocols/copy-prefix-extension.md',OUT/'preflight.json',ROOT/'experiments/copy_prefix_extension.py'):
        protected[str(p.relative_to(ROOT))]=digest(p)
    assert all(digest(ROOT/p)==v for p,v in protected.items())
    prefixes=[tuple(r[:6]) for r in rows]
    overlap={k:sum(p in {tuple(r) for r in pool} for p in prefixes) for k,pool in data.items() if k.endswith('_6')}
    write(OUT/'plan.json',dict(protected=protected,models=MODELS,positions=[4,5],unique_short_prefixes=len(set(prefixes)),short_pool_overlap=overlap))
    cells=[];saved={};maxerror=0.
    for seed,arm in MODELS:
        run=next(r for r in parent['confirmation'] if r['seed']==seed and r['arm']==arm)
        model=make_model('rope').eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model']);old=torch.load(ROOT/run['scores'],weights_only=True)
        for t in (4,5):
            evidence,groups,err=evaluate(model,rows,t);archived=old['test_10']['logits'][:,t-1]
            ae=float((archived-evidence['long']).abs().max());assert ae<=1e-9 and torch.equal(archived.argmax(-1),evidence['long'].argmax(-1));maxerror=max(maxerror,err,ae)
            cells.append(dict(seed=seed,arm=arm,position=t,groups=groups));saved[f'{seed}_{arm}_{t}']=evidence
            print(json.dumps(dict(seed=seed,arm=arm,position=t,**groups['all'])),flush=True)
    primary=cells[1]['groups']['all'];supported=primary['correct_to_eos']/128>=.9 and primary['positive_margin_changes']/128>=.9
    torch.save(saved,OUT/'evidence.pt');assert len(cells)==8 and all(digest(ROOT/p)==v for p,v in protected.items())
    write(OUT/'report.json',dict(cells=cells,supported=supported,training_updates=0,plan_sha256=digest(OUT/'plan.json'),evidence_sha256=digest(OUT/'evidence.pt'),audit=dict(passed=True,paired_cells=8,selected_predictions_per_implementation=2048,max_logit_error=maxerror,prediction_differences=0,protected_files=len(protected)),limitation='Opened selected long examples; source extension bundles suffix content, marker relocation and encoder context. Shared reference helpers.'))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
