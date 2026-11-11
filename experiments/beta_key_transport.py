"""Observed-stage beta/key factorial interventions with fixed value/head anchors."""
import argparse
import copy
import importlib.util
import itertools
import json
import time
import unittest
import torch
import write_interference_stages as prior

ROOT=prior.ROOT
OUT=ROOT/'results/beta_key_transport'
PROTOCOL='docs/protocols/beta-key-transport.md'
SOURCES=('experiments/beta_key_transport.py','experiments/audit_beta_key_transport.py','tests/test_beta_key_transport.py')
STEPS=(3000,12000)


def identity(case):return tuple(case[k] for k in ('seed','anchor','keys','beta'))


def cases():
    checkpoints={(c['seed'],c['step']):c['checkpoint'] for c in prior.stages.cases()}
    return [dict(seed=s,anchor=a,keys=k,beta=b,checkpoints={str(t):checkpoints[s,t] for t in STEPS},
                 data='results/delta_query_coverage/development_data.json')
            for s,a,k,b in itertools.product((3461,3463),STEPS,STEPS,STEPS)]


def load_models(case):
    with torch.random.fork_rng(devices=[]):
        return {int(t):prior.prior.historical.parent.previous.load_model(ROOT/p).eval() for t,p in case['checkpoints'].items()}


@torch.no_grad()
def hybrid(models,case):
    result=copy.deepcopy(models[case['anchor']]);result.key.weight.copy_(models[case['keys']].key.weight)
    result.write_logit.copy_(models[case['beta']].write_logit)
    for name,value in result.state_dict().items():
        donor=case['keys'] if name=='key.weight' else case['beta'] if name=='write_logit' else case['anchor']
        assert torch.equal(value,models[donor].state_dict()[name]),'Intervention scope'
    return result


def contrasts(bundles):
    result=[]
    for seed,anchor in sorted({(b['case']['seed'],b['case']['anchor']) for b in bundles}):
        cells={(b['case']['keys'],b['case']['beta']):b for b in bundles if b['case']['seed']==seed and b['case']['anchor']==anchor}
        assert set(cells)==set(itertools.product(STEPS,STEPS))
        for index in range(len(cells[STEPS[0],STEPS[0]]['summary'])):
            summaries={key:b['summary'][index] for key,b in cells.items()};base=summaries[STEPS[0],STEPS[0]]
            r=dict(seed=seed,anchor=anchor,group=base['group'],position=base['position'],count=base['count'])
            for field,name in (('target_coefficient_ratio_mean','ratio'),('final_accuracy','accuracy'),('final_ce_mean','ce')):
                values={k:v[field] for k,v in summaries.items()}
                a,b,c,d=[values[k] for k in itertools.product(STEPS,STEPS)]
                terms=dict(beta_early_keys=None,beta_late_keys=None,keys_early_beta=None,keys_late_beta=None,interaction=None) if a is None else dict(
                    beta_early_keys=b-a,beta_late_keys=d-c,keys_early_beta=c-a,keys_late_beta=d-b,interaction=d-c-b+a)
                r.update({name+'_'+k:v for k,v in terms.items()})
            result.append(r)
    return result


@torch.no_grad()
def execute(out,selected=None,fixture=False):
    assert not out.exists(),'Preserve evidence';selected=cases() if selected is None else selected
    files={p:prior.prior.digest(ROOT/p) for p in (*SOURCES,PROTOCOL,*prior.SOURCES,'experiments/shared_state_preservation.py',
        'experiments/association_load.py','experiments/query_gradient_variability.py','experiments/audit_shared_readout_feasibility.py')}
    for case in selected:
        for p in (*case['checkpoints'].values(),case['data']):files[p]=prior.prior.digest(ROOT/p)
    cached=None
    if not fixture:
        for name in ('lock.json','report.json','audit.json','traces.pt'):
            p=prior.OUT/name;files[p.relative_to(ROOT).as_posix()]=prior.prior.digest(p)
        old=json.loads((prior.OUT/'audit.json').read_text());assert old['passed'] and old['report_sha256']==prior.prior.digest(prior.OUT/'report.json')
        cached=torch.load(prior.OUT/'traces.pt',weights_only=True)
        p=ROOT/'results/beta_key_transport_preflight.json';pre=json.loads(p.read_text());assert pre['passed']
        assert pre['sources']=={s:prior.prior.digest(ROOT/s) for s in SOURCES};files[p.relative_to(ROOT).as_posix()]=prior.prior.digest(p)
    prior.prior.check_files(files);out.mkdir(parents=True)
    prior.prior.write(out/'lock.json',dict(files=files,cases=selected,fixture=fixture,steps=list(STEPS),guard=1e-9,versions=prior.prior.versions()))
    start=time.perf_counter();bundles=[];models_cache={};rows_cache={};references={}
    for case in selected:
        seed=case['seed']
        if seed not in models_cache:
            models_cache[seed]=load_models(case)
            for t,p in case['checkpoints'].items():assert torch.load(ROOT/p,weights_only=True)['step']==int(t)
            rows_cache[seed]=json.loads((ROOT/case['data']).read_text(encoding='utf-8-sig'))['pools']['train_32']
        models=models_cache[seed];rows=rows_cache[seed];snap={t:{k:v.clone() for k,v in m.state_dict().items()} for t,m in models.items()};rng=torch.get_rng_state().clone()
        if fixture:
            q,y,_=prior.prior.labels(rows);endpoint=prior.stages.original.features(models[STEPS[-1]],rows,'train_32')['native_logits'].argmax(-1)==y
        else:endpoint=next(b for b in cached if b['seed']==seed and b['step']==STEPS[-1])['endpoint_correct']
        refkey=(seed,case['anchor'])
        if refkey not in references:
            references[refkey]=prior.analyze(models[case['anchor']],rows,endpoint,seed,case['anchor']) if fixture else next(b for b in cached if b['seed']==seed and b['step']==case['anchor'])
        ref=references[refkey];model=hybrid(models,case);b=prior.analyze(model,rows,endpoint,seed,case['anchor'])
        native=ref['fields']['final_correct'];good=b['fields']['final_correct']
        b['fields'].update(final_ce=-b['logits'][-1].log_softmax(-1).gather(-1,b['labels'][...,None]).squeeze(-1),
            rescued=~native&good,harmed=native&~good,ratio_change=b['fields']['target_coefficient_ratio']-ref['fields']['target_coefficient_ratio'])
        b['summary']=prior.summaries(b['fields'],prior.cohorts(endpoint),seed,case['anchor'])
        for r in b['summary']:r['final_accuracy']=r['final_correct_count']/r['count'] if r['count'] else None
        b['case']=case
        if case['keys']==case['beta']==case['anchor']:
            torch.testing.assert_close(b['logits'],ref['logits'],atol=1e-9,rtol=1e-9)
            assert torch.equal(b['logits'].argmax(-1),ref['logits'].argmax(-1))
        assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,models[t].state_dict()[k]) for t,s in snap.items() for k,v in s.items())
        bundles.append(b);print(json.dumps(dict(cell=identity(case),accuracy=b['metrics']['all']['accuracy'],ce=b['metrics']['all']['ce'])),flush=True)
    torch.save(bundles,out/'traces.pt')
    report=dict(status='complete',cells=[dict(case=b['case'],beta=b['beta'],metrics=b['metrics'],summary=b['summary']) for b in bundles],contrasts=contrasts(bundles),
        query_rows=sum(b['labels'].numel() for b in bundles),answer_outputs=sum(len(b['logits'])*b['labels'].numel() for b in bundles),
        replay_writes=sum((len(b['logits'])-1)*len(b['labels']) for b in bundles),training_updates=0,optimizer_calls=0,autograd_calls=0,solver_calls=0,
        seconds=time.perf_counter()-start,lock_sha256=prior.prior.digest(out/'lock.json'),traces_sha256=prior.prior.digest(out/'traces.pt'))
    prior.prior.check_files(files);prior.prior.write(out/'report.json',report)


def preflight():
    spec=importlib.util.spec_from_file_location('checks',ROOT/SOURCES[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    p=ROOT/('results/beta_key_transport_preflight.json' if result.wasSuccessful() else f'results/beta_key_transport_preflight_failed_{time.time_ns()}.json')
    assert not p.exists();prior.prior.write(p,dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={s:prior.prior.digest(ROOT/s) for s in SOURCES}));assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight:preflight()
    else:execute(OUT)
