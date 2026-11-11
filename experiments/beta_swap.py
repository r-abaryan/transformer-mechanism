"""Frozen-checkpoint write-strength intervention, without training or a beta grid."""
import argparse
import copy
import csv
import importlib.util
import itertools
import json
import math
import time
import unittest

import torch
import delta_query_coverage as prior
import query_write_order as order

study=prior.study
ROOT=prior.ROOT
OUT=ROOT/'results/beta_swap'
VARIANTS=('native','donor_beta')
POOLS=tuple(f'{split}_{m}' for split in ('train','validation') for m in study.LOADS)


def dependencies():
    paths=('experiments/beta_swap.py','tests/test_beta_swap.py','docs/protocols/beta-swap.md')
    return {**prior.dependencies(),**{p:study.parent.digest(ROOT/p) for p in paths}}


def parent_evidence():
    r=json.loads((prior.OUT/'report.json').read_text());a=json.loads((prior.OUT/'independent_audit.json').read_text())
    assert r['status']=='complete' and a['passed'] and a['report_sha256']==study.parent.digest(prior.OUT/'report.json')
    assert r['source_dependencies']==prior.dependencies() and r['scores_sha256']==study.parent.digest(prior.OUT/'scores.pt')
    assert len(r['runs'])==4 and {(x['seed'],x['arm']) for x in r['runs']}==set(itertools.product(prior.DEV_SEEDS,prior.ARMS))
    data=json.loads((prior.OUT/'development_data.json').read_text());assert study.parent.digest(prior.OUT/'development_data.json')==r['data_sha256']['development']
    assert set(data['pools'])==set(POOLS);study.audit_data(data)
    for x in r['runs']:assert study.parent.digest(ROOT/x['checkpoint'])==x['checkpoint_sha256']
    return r,data


def beta_only(base,changed):
    a,b=base.state_dict(),changed.state_dict();assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a if k!='write_logit'),'Non-beta weight changed'


def variant_model(base,donor,variant):
    assert variant in VARIANTS
    result=copy.deepcopy(base)
    if variant=='donor_beta':
        with torch.no_grad():result.write_logit.copy_(donor.write_logit)
    beta_only(base,result)
    assert torch.equal(result.write_logit,donor.write_logit if variant=='donor_beta' else base.write_logit)
    return result


def positions(rows):
    queries,labels=prior.query_tables(rows);result=[]
    for i,row in enumerate(rows):
        assert len(row['events'])==32
        result.append([next(t for t,(k,_) in enumerate(row['events']) if k==q) for q in queries[i].tolist()])
    return queries,labels,torch.tensor(result)


def evaluate(model,rows):
    q,_,first=positions(rows);events,_,_=study.parent.tensors(rows)
    before=study.parent.state_hash(model.state_dict())
    with torch.no_grad():
        # Group requested first-write prefixes, using the original Reader forward.
        immediate=torch.empty(len(rows),q.shape[1],6,dtype=torch.float64)
        for t in range(32):
            i,j=(first==t).nonzero(as_tuple=True)
            if len(i):
                immediate[i,j]=torch.cat([model(events[a,:t+1],q[a,b]) for a,b in zip(i.split(128),j.split(128))])
        finals=[]
        for e,queries in zip(events.split(128),q.split(128)):
            state=model.prefill(e)
            finals.append(torch.stack([model.read(state,queries[:,j]) for j in range(queries.shape[1])],1))
        final=torch.cat(finals)
    assert before==study.parent.state_hash(model.state_dict())
    return dict(immediate=immediate,final=final)


def summarize(scores,rows):
    flat=prior.expanded_rows(rows);_,labels,_=positions(rows)
    complete=(scores['final'].argmax(-1)==labels).all(1)
    return dict(**{s:study.parent.metrics(z.reshape(-1,6),flat)['all'] for s,z in scores.items()},
                complete_maps=dict(count=len(rows),correct=int(complete.sum()),accuracy=float(complete.double().mean())),
                immediate_to_final=prior.previous.geometry.prior.paired_counts(scores['immediate'].reshape(-1,6),scores['final'].reshape(-1,6),flat))


def reference(model,rows):
    # Independent dictionary labels/first positions and matrix transition recurrence.
    memory=[dict(row['events']) for row in rows];queries=[sorted(m) for m in memory]
    first=[[next(t for t,e in enumerate(row['events']) if e[0]==q) for q in qs] for row,qs in zip(rows,queries)]
    labels=[[m[q] for q in qs] for m,qs in zip(memory,queries)]
    for row,m in zip(rows,memory):
        assert len(m)==row['load'] and m[row['query']]==row['label']
        assert all(m[k]==v for k,v in row['events'])
    events=torch.tensor([row['events'] for row in rows]);qids=torch.tensor(queries);first=torch.tensor(first)
    raw=model.key.weight[events[...,0]];k=raw/raw.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
    qr=model.key.weight[qids];q=qr/qr.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
    v=model.value.weight[events[...,1]];state=torch.zeros(len(rows),16,8,dtype=torch.float64)
    eye=torch.eye(16,dtype=torch.float64);beta=model.write_logit.sigmoid()
    immediate=torch.empty(len(rows),q.shape[1],6,dtype=torch.float64)
    with torch.no_grad():
        for t in range(32):
            kt=k[:,t]
            state=(eye-beta*kt[:,:,None]*kt[:,None,:])@state+beta*kt[:,:,None]*v[:,t,None,:]
            logits=(q@state)@model.output.weight.T+model.output.bias
            mask=first==t;immediate[mask]=logits[mask]
    flat=[dict(row,query=q,label=m[q]) for row,m,qs in zip(rows,memory,queries) for q in qs]
    return dict(immediate=immediate,final=logits),flat,labels


def audit_cell(base_path,donor_path,variant,rows,scores,metrics):
    # Do not reuse variant_model when applying the audit intervention.
    model=prior.previous.geometry.prior.load(base_path,'delta')
    original={k:v.clone() for k,v in model.state_dict().items()}
    if variant=='donor_beta':
        other=torch.load(donor_path,weights_only=True)['model']
        with torch.no_grad():model.write_logit.copy_(other['write_logit'])
    else:assert variant=='native'
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items() if k!='write_logit')
    ref,flat,labels=reference(model,rows);errors=[]
    for stage in ('immediate','final'):
        errors.append(prior.previous.geometry.prior.check_scores(scores[stage].reshape(-1,6),ref[stage].reshape(-1,6)))
        reduced=prior.previous.geometry.prior.reduce_scores(ref[stage].reshape(-1,6),flat)['all']
        prior.previous.geometry.prior.check_metrics({'all':metrics[stage]},{'all':reduced})
    correct=sum(all(max(range(6),key=lambda c:float(ref['final'][i,j,c]))==labels[i][j] for j in range(len(labels[i]))) for i in range(len(rows)))
    assert metrics['complete_maps']==dict(count=len(rows),correct=correct,accuracy=correct/len(rows))
    assert metrics['immediate_to_final']==order.independent_pairs(ref['immediate'].reshape(-1,6),ref['final'].reshape(-1,6),flat)
    return max(errors),ref,flat


def decision(cells):
    expected=set(itertools.product(prior.DEV_SEEDS,prior.ARMS,POOLS,VARIANTS))
    lookup={(x['seed'],x['arm'],x['pool'],x['variant']):x for x in cells}
    assert len(cells)==len(lookup)==len(expected) and set(lookup)==expected,'Missing or duplicate beta-swap cells'
    gains={}
    for arm in prior.ARMS:
        gains[arm]=[lookup[s,arm,'validation_32','donor_beta']['metrics']['final']['accuracy']-lookup[s,arm,'validation_32','native']['metrics']['final']['accuracy'] for s in prior.DEV_SEEDS]
    lower=all(g>=.01 for g in gains['fixed']);higher=all(g<=-.01 for g in gains['resampled'])
    return dict(validation_32_gains=gains,lower_beta_improves_fixed=lower,higher_beta_harms_resampled=higher,
                bidirectional_write_strength_evidence=lower and higher,
                swapped_complete_map_95={arm:all(lookup[s,arm,'validation_32','donor_beta']['metrics']['complete_maps']['accuracy']>=.95 for s in prior.DEV_SEEDS) for arm in prior.ARMS})


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old,data=parent_evidence();source=dependencies()
    assert not OUT.exists();OUT.mkdir()
    protected=json.loads((prior.OUT/'preserved.json').read_text());protected.update(old['source_dependencies'])
    for p in prior.OUT.rglob('*'):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=study.parent.digest(p)
    for p in ('docs/topics/attention/delta-query-coverage-results.md','docs/topics/attention/delta-query-coverage-guide.md'):protected[p]=study.parent.digest(ROOT/p)
    assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items());study.parent.write(OUT/'preserved.json',protected)
    spec=importlib.util.spec_from_file_location('test_beta_swap',ROOT/'tests/test_beta_swap.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    study.parent.write(OUT/'preflight.json',dict(passed=result.wasSuccessful(),methods=result.testsRun,source_dependencies=source));assert result.wasSuccessful()
    expected=len(prior.DEV_SEEDS)*len(prior.ARMS)*len(POOLS)*len(VARIANTS)
    predictions=len(prior.DEV_SEEDS)*len(prior.ARMS)*len(VARIANTS)*sum(len(rows)*rows[0]['load'] for rows in data['pools'].values())
    study.parent.write(OUT/'lock.json',dict(source_dependencies=source,expected_cells=expected,predictions_per_stage=predictions,
        parent_report_sha256=study.parent.digest(prior.OUT/'report.json'),data_sha256=study.parent.digest(prior.OUT/'development_data.json'),preflight_sha256=study.parent.digest(OUT/'preflight.json')))
    lookup={(x['seed'],x['arm']):x for x in old['runs']};original_scores=torch.load(prior.OUT/'scores.pt',weights_only=True)
    cells,saved=[],{};start=time.perf_counter()
    for seed,arm in itertools.product(prior.DEV_SEEDS,prior.ARMS):
        item=lookup[seed,arm];donor=lookup[seed,'resampled' if arm=='fixed' else 'fixed']
        base=prior.previous.geometry.prior.load(ROOT/item['checkpoint'],'delta');other=prior.previous.geometry.prior.load(ROOT/donor['checkpoint'],'delta')
        for pool in POOLS:
            rows=data['pools'][pool];native=None
            for variant in VARIANTS:
                model=variant_model(base,other,variant);scores=evaluate(model,rows);metrics=summarize(scores,rows)
                if variant=='native':
                    native=scores
                    prior.previous.geometry.prior.check_scores(scores['final'].reshape(-1,6),original_scores[f'development_{seed}_{arm}_all_{pool}'].reshape(-1,6))
                    prior.previous.geometry.prior.check_metrics({'all':metrics['final']},{'all':item['metrics']['all_'+pool]['all_queries']})
                    assert metrics['complete_maps']==item['metrics']['all_'+pool]['complete_maps']
                flat=prior.expanded_rows(rows)
                pairs={s:prior.previous.geometry.prior.paired_counts(native[s].reshape(-1,6),scores[s].reshape(-1,6),flat) for s in ('immediate','final')}
                key=f'{seed}_{arm}_{pool}_{variant}';saved[key]=scores
                cells.append(dict(seed=seed,arm=arm,pool=pool,variant=variant,beta=float(model.write_logit.sigmoid().detach()),
                    checkpoint=item['checkpoint'],checkpoint_sha256=item['checkpoint_sha256'],donor_checkpoint=donor['checkpoint'],donor_sha256=donor['checkpoint_sha256'],
                    metrics=metrics,vs_native=pairs,non_beta_weights_unchanged=True))
        print(f'Evaluated seed {seed}, {arm}, native and donor beta.',flush=True)
    verdict=decision(cells);assert len(cells)==expected and sum(x['immediate'].numel()//6 for x in saved.values())==predictions
    assert source==dependencies() and all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    torch.save(saved,OUT/'scores.pt')
    study.parent.write(OUT/'report.json',dict(status='complete',training_updates=0,device='cpu',dtype='float64',threads=1,
        source_dependencies=source,cells=cells,decision=verdict,predictions_per_stage=predictions,total_predictions=2*predictions,
        scores_sha256=study.parent.digest(OUT/'scores.pt'),protected_files_unchanged=len(protected),elapsed_seconds=time.perf_counter()-start,
        scope='Opened development training/validation histories, all present queries, first-write immediate versus common final state. Isolated inference beta effect within each checkpoint; no fresh confirmation or training repair.'))
    print('Beta intervention complete. Run audit next.',flush=True)


def audit():
    torch.set_num_threads(1);old,data=parent_evidence();r=json.loads((OUT/'report.json').read_text());lock=json.loads((OUT/'lock.json').read_text())
    assert r['status']=='complete' and r['training_updates']==0 and r['source_dependencies']==lock['source_dependencies']==dependencies()
    assert lock['parent_report_sha256']==study.parent.digest(prior.OUT/'report.json') and lock['data_sha256']==study.parent.digest(prior.OUT/'development_data.json')
    assert lock['preflight_sha256']==study.parent.digest(OUT/'preflight.json') and json.loads((OUT/'preflight.json').read_text())['passed']
    assert r['scores_sha256']==study.parent.digest(OUT/'scores.pt');saved=torch.load(OUT/'scores.pt',weights_only=True)
    expected=len(prior.DEV_SEEDS)*len(prior.ARMS)*len(POOLS)*len(VARIANTS);per_stage=len(prior.DEV_SEEDS)*len(prior.ARMS)*len(VARIANTS)*sum(len(rows)*rows[0]['load'] for rows in data['pools'].values())
    assert lock['expected_cells']==expected==len(r['cells'])==len(saved)==64 and lock['predictions_per_stage']==r['predictions_per_stage']==per_stage==445440
    assert r['total_predictions']==2*per_stage==890880 and decision(r['cells'])==r['decision']
    originals={(x['seed'],x['arm']):x for x in old['runs']};old_scores=torch.load(prior.OUT/'scores.pt',weights_only=True);errors=[];csvrows=[];used=set()
    for c in r['cells']:
        item=originals[c['seed'],c['arm']];donor=originals[c['seed'],'resampled' if c['arm']=='fixed' else 'fixed']
        assert c['checkpoint']==item['checkpoint'] and c['donor_checkpoint']==donor['checkpoint']
        assert c['checkpoint_sha256']==study.parent.digest(ROOT/c['checkpoint']) and c['donor_sha256']==study.parent.digest(ROOT/c['donor_checkpoint'])
        assert c['beta']==(donor if c['variant']=='donor_beta' else item)['beta'] and c['non_beta_weights_unchanged']
        rows=data['pools'][c['pool']];key=f"{c['seed']}_{c['arm']}_{c['pool']}_{c['variant']}";used.add(key)
        error,ref,flat=audit_cell(ROOT/c['checkpoint'],ROOT/c['donor_checkpoint'],c['variant'],rows,saved[key],c['metrics']);errors.append(error)
        native=saved[f"{c['seed']}_{c['arm']}_{c['pool']}_native"]
        for stage in ('immediate','final'):
            assert order.independent_pairs(native[stage].reshape(-1,6),ref[stage].reshape(-1,6),flat)==c['vs_native'][stage]
            csvrows.append(dict(seed=c['seed'],arm=c['arm'],pool=c['pool'],variant=c['variant'],stage=stage,beta=c['beta'],**c['metrics'][stage],
                complete_map_accuracy=c['metrics']['complete_maps']['accuracy'] if stage=='final' else '',rescued=c['vs_native'][stage]['rescued'],harmed=c['vs_native'][stage]['harmed']))
        if c['variant']=='native':
            prior.previous.geometry.prior.check_scores(ref['final'].reshape(-1,6),old_scores[f"development_{c['seed']}_{c['arm']}_all_{c['pool']}"].reshape(-1,6))
            prior.previous.geometry.prior.check_metrics({'all':c['metrics']['final']},{'all':item['metrics']['all_'+c['pool']]['all_queries']})
            assert c['metrics']['complete_maps']==item['metrics']['all_'+c['pool']]['complete_maps']
    assert used==set(saved) and len(csvrows)==expected*2
    protected=json.loads((OUT/'preserved.json').read_text());assert all(study.parent.digest(ROOT/p)==h for p,h in protected.items())
    assert not (OUT/'independent_audit.json').exists()
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    study.parent.write(OUT/'independent_audit.json',dict(passed=True,report_sha256=study.parent.digest(OUT/'report.json'),source_dependencies=dependencies(),
        cells=expected,stage_metric_cells=len(csvrows),predictions=2*per_stage,max_logit_error=max(errors),argmax_disagreements=0,protected_files_unchanged=len(protected),
        scope='Independent first-position/label construction, matrix transitions, scalar CE/map and paired-count reductions. Native archived scores reproduced. Data/weights/PyTorch shared; policy replay shares tested functions.'))
    print('Beta-swap audit passes.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','audit'));args=p.parse_args();run() if args.phase=='run' else audit()
