"""Classifier control appended to the frozen exposure trace, no neural training."""
import copy
import json
import math
import uuid
from pathlib import Path

import torch

from copy_position_sites import ROOT,digest,write,make_model
from copy_exposure_stage_trace import STAGES,trace,audit as audit_native

PARENT=ROOT/'results/copy_exposure_stage_trace'
OUT=PARENT/'readout_control'
TERMS=('control_native','intervention_native_common','intervention_native_own',
       'fixed_control_difference','fixed_intervention_difference','representation',
       'classifier','rival_change','common_difference','native_difference',
       'classifier_at_control_state','classifier_at_intervention_state')


def score_pair(a,b):
    assert torch.equal(a['labels'],b['labels'])
    assert a['readout'].shape==b['readout'].shape
    competitor=a['competitor'];other=b['competitor'];n=competitor.shape[1]-1
    assert torch.equal(competitor[:,:n],a['labels'][:,:n])
    assert torch.equal(other[:,:n],b['labels'][:,:n])
    assert bool((competitor[:,-1]!=2).all()) and bool((other[:,-1]!=2).all())
    aa=a['readout'][2]-a['readout'][competitor]
    ab=b['readout'][2]-b['readout'][competitor]
    own=b['readout'][2]-b['readout'][other]
    terms={}
    for stage in STAGES:
        ha,hb=a['stages'][stage],b['stages'][stage]
        dh,da=hb-ha,ab-aa
        dot=lambda u,v:(u*v).sum(-1)
        ma,mb,mo=dot(aa,ha),dot(ab,hb),dot(own,hb)
        terms[stage]=dict(control_native=ma,intervention_native_common=mb,intervention_native_own=mo,
            fixed_control_difference=dot(aa,dh),fixed_intervention_difference=dot(ab,dh),
            representation=dot((aa+ab)/2,dh),classifier=dot(da,(ha+hb)/2),
            rival_change=mo-mb,common_difference=mb-ma,native_difference=mo-ma,
            classifier_at_control_state=dot(da,ha),classifier_at_intervention_state=dot(da,hb))
    return terms


def scalar_audit(a,b,terms):
    assert set(terms)==set(STAGES)
    maximum=0.
    for s in STAGES:
        assert set(terms[s])==set(TERMS)
        for i,row in enumerate(a['stages'][s].tolist()):
            for j,ha in enumerate(row):
                hb=b['stages'][s][i,j].tolist();ca=int(a['competitor'][i,j]);cb=int(b['competitor'][i,j])
                aa=[float(a['readout'][2,k]-a['readout'][ca,k]) for k in range(len(ha))]
                ab=[float(b['readout'][2,k]-b['readout'][ca,k]) for k in range(len(ha))]
                own=[float(b['readout'][2,k]-b['readout'][cb,k]) for k in range(len(ha))]
                dot=lambda x,y:math.fsum(u*v for u,v in zip(x,y))
                ma,mb,mo=dot(aa,ha),dot(ab,hb),dot(own,hb)
                dh=[v-u for u,v in zip(ha,hb)];da=[v-u for u,v in zip(aa,ab)]
                expected=dict(control_native=ma,intervention_native_common=mb,intervention_native_own=mo,
                    fixed_control_difference=dot(aa,dh),fixed_intervention_difference=dot(ab,dh),
                    representation=math.fsum((x+y)*(v-u)/2 for x,y,u,v in zip(aa,ab,ha,hb)),
                    classifier=math.fsum((y-x)*(u+v)/2 for x,y,u,v in zip(aa,ab,ha,hb)),
                    rival_change=mo-mb,common_difference=mb-ma,native_difference=mo-ma,
                    classifier_at_control_state=dot(da,ha),classifier_at_intervention_state=dot(da,hb))
                for k,v in expected.items():
                    e=abs(float(terms[s][k][i,j])-v);assert e<1e-9,(s,k,e);maximum=max(maximum,e)
                assert abs(expected['representation']+expected['classifier']-expected['common_difference'])<1e-9
                assert abs(expected['fixed_control_difference']+expected['classifier_at_intervention_state']-expected['common_difference'])<1e-9
                assert abs(expected['fixed_intervention_difference']+expected['classifier_at_control_state']-expected['common_difference'])<1e-9
                assert abs(expected['representation']+expected['classifier']+expected['rival_change']-expected['native_difference'])<1e-9
        torch.testing.assert_close(terms[s]['control_native'],a['margins'][s],atol=1e-9,rtol=0)
        torch.testing.assert_close(terms[s]['intervention_native_own'],b['margins'][s],atol=1e-9,rtol=0)
    return maximum


def cells(ids,kind,n,anchors):
    return [(i,j) for i in ids for j in (range(n) if kind=='content' else [n] if kind=='terminal' else [anchors[i]]) if j is not None]


def reduce_terms(terms,selected):
    if not selected:return None
    summary={s:{k:math.fsum(float(v[i,j]) for i,j in selected)/len(selected) for k,v in ts.items()} for s,ts in terms.items()}
    robustness={}
    for name in ('fixed_control_difference','fixed_intervention_difference'):
        cn,ff,final=[summary[s][name] for s in ('cross_norm','ffn_sum','final')]
        robustness[name]=dict(mean_grows_at_ffn=abs(ff)>abs(cn),mean_grows_at_final_norm=abs(final)>abs(ff),
            examples_grow_at_ffn=math.fsum(abs(float(terms['ffn_sum'][name][i,j]))>abs(float(terms['cross_norm'][name][i,j])) for i,j in selected)/len(selected),
            examples_grow_at_final_norm=math.fsum(abs(float(terms['final'][name][i,j]))>abs(float(terms['ffn_sum'][name][i,j])) for i,j in selected)/len(selected))
    return dict(positions=len(selected),examples=len({i for i,j in selected}),stages=summary,robustness=robustness)


def preflight():
    folder=ROOT/'results/test_runs'/('readout-control-'+uuid.uuid4().hex);folder.mkdir(parents=True)
    torch.manual_seed(139);m=make_model('rope').eval();torch.save(m.state_dict(),folder/'initialized.pt')
    m.load_state_dict(torch.load(folder/'initialized.pt',weights_only=True))
    rows=[[3,4,5],[3,3,4]];a=trace(m,rows);audit_native(m,rows,a)
    t=score_pair(a,a);scalar_audit(a,a,t)
    assert all(torch.count_nonzero(ts['representation'])==0 and torch.count_nonzero(ts['classifier'])==0 for ts in t.values())
    b=copy.deepcopy(a)
    # Supplied two-coordinate interacting example: total 10 = state 4 + classifier 6.
    for e in (a,b):
        e['readout'].zero_();e['competitor'].fill_(3);e['labels'].fill_(3)
        for s in STAGES:e['stages'][s].zero_()
    a['readout'][2,0]=1;b['readout'][2,0]=1;b['readout'][2,1]=2
    for s in STAGES:
        a['stages'][s][...,0]=1;a['stages'][s][...,1]=2
        b['stages'][s][...,0]=3;b['stages'][s][...,1]=4
        a['margins'][s]=torch.ones_like(a['labels'],dtype=torch.float64)
        b['margins'][s]=torch.full_like(b['margins'][s],11.)
    terms=score_pair(a,b);scalar_audit(a,b,terms)
    assert float(terms['final']['representation'][0,0])==4 and float(terms['final']['classifier'][0,0])==6
    assert reduce_terms(terms,[]) is None
    bad=copy.deepcopy(terms);bad['final']['representation'][0,0]+=.01
    try:scalar_audit(a,b,bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupted decomposition accepted')
    bad=copy.deepcopy(terms);bad['final'].pop('rival_change')
    try:scalar_audit(a,b,bad)
    except AssertionError:pass
    else:raise AssertionError('Missing term accepted')
    shift=torch.arange(8,dtype=torch.float64)[None,:]
    shifted=copy.deepcopy(b);shifted['readout']+=shift
    for s in STAGES:
        for k,v in score_pair(a,shifted)[s].items():torch.testing.assert_close(v,terms[s][k],atol=1e-9,rtol=0)
    return dict(passed=True,initialized_saved_trace=True,interacting_fixture=True,common_logit_shift_invariance=True,
                empty_group_explicit=True,corrupted_and_missing_terms_rejected=True)


def main():
    assert not OUT.exists(),'Preserve derived evidence'
    pre=preflight()
    original=json.loads((PARENT/'report.json').read_text());plan=json.loads((PARENT/'plan.json').read_text())
    assert digest(PARENT/'traces.pt')==original['traces_sha256'] and digest(PARENT/'plan.json')==original['plan_sha256']
    protected=dict(plan['protected'])
    for p in (PARENT/'report.json',PARENT/'plan.json',PARENT/'traces.pt',ROOT/'experiments/copy_exposure_readout_control.py',
              ROOT/'docs/protocols/copy-exposure-readout-control.md',ROOT/'tests/test_copy_exposure_readout_control.py'):
        protected[p.relative_to(ROOT).as_posix()]=digest(p)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    OUT.mkdir();write(OUT/'preflight.json',pre);write(OUT/'plan.json',dict(protected=protected,preflight_sha256=digest(OUT/'preflight.json')))
    report_path=ROOT/'docs/topics/benchmarks/copy-exposure-stage-trace-results.md'
    (OUT/'educational-report-before.md').write_bytes(report_path.read_bytes())
    traces=torch.load(PARENT/'traces.pt',weights_only=True);summary={};derived={};maximum=0.
    for step in (6500,7000,7500,8000):
        key=str(step);summary[key]={};derived[key]={}
        for pool in ('test_6','test_10'):
            a,b=[traces[f'{arm}_{step}'][pool] for arm in ('cross_only','both')]
            terms=score_pair(a,b);maximum=max(maximum,scalar_audit(a,b,terms));derived[key][pool]=terms
            groups=plan['cohorts'] if pool=='test_10' else {'short_control':list(range(len(a['labels'])))}
            kinds=('content','terminal','control_stop') if pool=='test_10' else ('content','terminal')
            summary[key][pool]={g:{kind:reduce_terms(terms,cells(ids,kind,a['labels'].shape[1]-1,plan['anchors'])) for kind in kinds} for g,ids in groups.items()}
            print(step,pool,flush=True)
    torch.save(derived,OUT/'terms.pt')
    # Reload the derived file, and recompute its scalar values from original traces.
    loaded=torch.load(OUT/'terms.pt',weights_only=True)
    for step in loaded:
        for pool,terms in loaded[step].items():
            a,b=[traces[f'{arm}_{step}'][pool] for arm in ('cross_only','both')]
            scalar_audit(a,b,terms)
            groups=plan['cohorts'] if pool=='test_10' else {'short_control':list(range(len(a['labels'])))}
            for g,ids in groups.items():
                for kind,val in summary[step][pool][g].items():assert reduce_terms(terms,cells(ids,kind,a['labels'].shape[1]-1,plan['anchors']))==val
                # Reconcile original mean native stage differences, including rival changes.
                for kind,val in summary[step][pool][g].items():
                    if val is None:continue
                    for s in STAGES:
                        old_a=original['summary'][f'cross_only_{step}'][pool][g][kind]['margin'][s]
                        old_b=original['summary'][f'both_{step}'][pool][g][kind]['margin'][s]
                        assert abs(val['stages'][s]['native_difference']-(old_b-old_a))<1e-9
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    write(OUT/'report.json',dict(status='complete',summary=summary,max_scalar_error=maximum,persisted_audit_passed=True,
        paired_model_pool_cells=8,stage_cells=48,paired_positions=sum(traces[f'cross_only_{step}'][pool]['labels'].numel() for step in (6500,7000,7500,8000) for pool in ('test_6','test_10')),
        protected_files=len(protected),terms_sha256=digest(OUT/'terms.pt'),plan_sha256=digest(OUT/'plan.json')))


if __name__=='__main__':
    torch.set_num_threads(1);main()
