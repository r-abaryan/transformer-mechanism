"""Independent matrix gradients, Adam arithmetic, cap and complete evaluation."""
import copy
import json
import math
import torch
import readout_norm_cap_training as study
import audit_query_feedback_training as reference
import audit_shared_state_feasibility as comparisons


def norm(state):
    rows=[w+[b] for w,b in zip(state['output.weight'].tolist(),state['output.bias'].tolist())]
    mean=[math.fsum(r[j] for r in rows)/6 for j in range(9)]
    return math.sqrt(math.fsum((r[j]-mean[j])**2 for r in rows for j in range(9)))


def projection(raw,radius,arm):
    length=norm(raw);factor=min(1.,radius/length) if arm=='cap' else 1.;expected=copy.deepcopy(raw)
    if factor<1:
        for name in ('output.weight','output.bias'):
            values=raw[name];mean=values.mean(0,keepdim=True)
            expected[name]=mean+(values-mean)*factor
    return expected,dict(raw_norm=length,factor=factor,after_norm=norm(expected))


def independent_decision(cells,seeds):
    rows=[]
    lookup={(c['seed'],c['arm'],c['pool']):c['metrics'] for c in cells}
    for seed in seeds:
        def metric(arm,pool):return lookup[seed,arm,pool]
        old=metric('control','train_32');new=metric('cap','train_32');a=old['groups']['all'];b=new['groups']['all']
        tmean={arm:math.fsum(metric(arm,f'train_{m}')['groups']['all']['ce'] for m in (2,8,16,32))/4 for arm in ('control','cap')}
        vmean={arm:math.fsum(metric(arm,f'validation_{m}')['groups']['all']['ce'] for m in (2,8,16,32))/4 for arm in ('control','cap')}
        lower=all(metric('cap',f'train_{m}')['groups']['all']['accuracy']>=metric('control',f'train_{m}')['groups']['all']['accuracy'] for m in (2,8,16))
        vacc=all(metric('cap',f'validation_{m}')['groups']['all']['accuracy']>=metric('control',f'validation_{m}')['groups']['all']['accuracy'] for m in (2,8,16,32))
        fit={arm:all(metric(arm,f'train_{m}')['groups']['all']['accuracy']>=.99 for m in (2,8,16,32)) for arm in ('control','cap')}
        benefit=(b['accuracy']-a['accuracy']>=.005 and b['ce']<=.95*a['ce']+1e-10 and new['complete_maps']['correct']>=old['complete_maps']['correct'] and lower and tmean['cap']-tmean['control']<=1e-10)
        rows.append(dict(seed=seed,train32_accuracy_gain=b['accuracy']-a['accuracy'],train32_ce_delta=b['ce']-a['ce'],train_equal_load_ce_delta=tmean['cap']-tmean['control'],
            lower_train_preserved=lower,validation_accuracy_preserved=vacc,validation_equal_load_ce_delta=vmean['cap']-vmean['control'],training_benefit=benefit,
            validation_preserved=vacc and vmean['cap']-vmean['control']<=1e-10,training_fit=fit))
    return dict(seeds=rows,training_benefit_all=all(x['training_benefit'] for x in rows),validation_preserved_all=all(x['validation_preserved'] for x in rows))


def audit_run(case,run,pools,updates):
    root=study.ROOT;path=root/case['checkpoint'];source=torch.load(path,weights_only=True)
    assert run['start']==source['step'] and run['updates']==updates and run['stop']==source['step']+updates
    assert run['checkpoint_sha256']==study.training.replay.digest(root/run['checkpoint']) and run['traces_sha256']==study.training.replay.digest(root/run['traces'])
    traces=torch.load(root/run['traces'],weights_only=True);assert len(traces)==updates
    with torch.random.fork_rng(devices=[]):model=study.training.previous.load_model(path)
    state=copy.deepcopy(source['model']);optimizer=copy.deepcopy(source['optimizer']);group=optimizer['param_groups'][0];names=list(dict(model.named_parameters()))
    assert (group['lr'],group['betas'],group['eps'],group['weight_decay'],group['foreach'])==(.003,(.9,.999),1e-8,0,False)
    assert not group['amsgrad'] and not group['maximize'];b1,b2=group['betas'];radius=norm(state)
    assert abs(radius-run['radius'])<1e-11
    hr=torch.Generator().manual_seed(case['seed']+10000);qr=torch.Generator().manual_seed(case['seed']+20000)
    for step in range(1,source['step']+1):
        load=(2,8,16,32)[(step-1)%4];torch.randint(len(pools[f'train_{load}']),(64,),generator=hr);torch.randint(load,(64,load),generator=qr)
    tables={m:reference.independent.inputs(pools[f'train_{m}'],'all_queries') for m in (2,8,16,32)}
    maximum_gradient=maximum_adam=0.;targets=0;projections=0
    for index,t in enumerate(traces):
        step=source['step']+index+1;load=(2,8,16,32)[(step-1)%4]
        assert (t['seed'],t['arm'],t['step'],t['load'])==(case['seed'],run['arm'],step,load)
        ids=torch.randint(len(pools[f'train_{load}']),(64,),generator=hr);draws=torch.randint(load,(64,load),generator=qr)
        assert torch.equal(ids,t['indices']) and torch.equal(draws,t['draws']),'Stream mismatch'
        e,q,y,_=tables[load];model.load_state_dict(state)
        z,_=reference.independent.forward_tangent(model,e[ids],q[ids]);loss=(z.logsumexp(-1)-z.gather(-1,y[ids,...,None]).squeeze(-1)).mean()
        assert abs(float(loss.detach())-t['loss'])<1e-9,'Training CE'
        gradients=torch.autograd.grad(loss,tuple(model.parameters()))
        assert set(t['gradients'])==set(state)==set(t['raw_model'])==set(t['after_model'])
        for name,g in zip(names,gradients):maximum_gradient=max(maximum_gradient,reference.independent.close(t['gradients'][name],g,gradient=True))
        raw={}
        for name,pid in zip(names,group['params']):
            g=t['gradients'][name];previous=optimizer['state'].get(pid,{})
            count=int(previous['step'])+1 if previous else 1
            m=previous['exp_avg'] if previous else torch.zeros_like(g);v=previous['exp_avg_sq'] if previous else torch.zeros_like(g)
            first=b1*m+(1-b1)*g;second=b2*v+(1-b2)*g.square()
            expected=state[name]-group['lr']*(first/(1-b1**count))/((second/(1-b2**count)).sqrt()+group['eps'])
            err=float((expected-t['raw_model'][name]).abs().max());assert err<=1e-11,'Adam parameter update';maximum_adam=max(maximum_adam,err)
            optimizer['state'][pid]=dict(step=torch.tensor(float(count)),exp_avg=first,exp_avg_sq=second)
            raw[name]=t['raw_model'][name]
        expected,statistics=projection(raw,radius,run['arm']);comparisons.compare(t['projection'],statistics)
        for name,value in expected.items():
            err=float((value-t['after_model'][name]).abs().max());assert err<=1e-11,'Cap projection'
            if not name.startswith('output.'):assert torch.equal(raw[name],t['after_model'][name]),'Projection changed memory'
        if run['arm']=='cap':assert norm(t['after_model'])<=radius+1e-11
        projections+=statistics['factor']<1;state=t['after_model'];targets+=y[ids].numel()
        if (index+1)%250==0:print(json.dumps(dict(audit_seed=case['seed'],arm=run['arm'],step=step)),flush=True)
    final=torch.load(root/run['checkpoint'],weights_only=True)
    assert study.training.replay.equal(state,final['model']) and final['step']==run['stop']
    assert torch.equal(hr.get_state(),final['history_rng']) and torch.equal(qr.get_state(),final['query_rng'])
    assert group==final['optimizer']['param_groups'][0]
    for pid,wanted in optimizer['state'].items():
        actual=final['optimizer']['state'][pid];assert int(actual['step'])==int(wanted['step'])
        for name in ('exp_avg','exp_avg_sq'):assert float((actual[name]-wanted[name]).abs().max())<=1e-11,'Final moments'
    assert run['answer_targets']==targets and run['histories']==64*updates and run['input_events']==64*updates*32 and run['projected_updates']==projections
    assert abs(run['final_head_norm']-norm(final['model']))<=1e-11
    assert abs(run['value_norm']-float(final['model']['value.weight'].norm()))<=1e-11 and run['beta']==float(final['model']['write_logit'].sigmoid())
    if run['arm']=='control' and 'reference' in case:
        old=torch.load(root/case['reference'],weights_only=True)
        for name in ('model','optimizer','step','history_rng','query_rng'):assert study.training.replay.equal(final[name],old[name]),'Original control fidelity'
        assert run['historical_exact'] is True
    else:assert run['historical_exact'] is None
    return dict(max_gradient_error=maximum_gradient,max_adam_error=maximum_adam,updates=updates),traces


def audit(out,persist=True):
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text());replay=study.training.replay
    replay.check_files(lock['files']);assert report['status']=='complete' and lock['arms']==['control','cap'] and lock['loads']==[2,8,16,32]
    assert lock['decision_guard']==1e-10 and lock['versions']==dict(torch=torch.__version__)
    for name in ('lock','scores','pretest'):
        path=out/({'lock':'lock.json','scores':'scores.pt','pretest':'pretest_decision.json'}[name]);assert report[name+'_sha256']==replay.digest(path)
    assert lock['pool_sha256']==replay.digest(out/'pools.json');pools=json.loads((out/'pools.json').read_text())['pools']
    if not lock['fixture']:
        _,original=study.inputs();assert pools==original
        assert lock['updates']==1000 and [c['seed'] for c in lock['cases']]==[3461,3463]
        assert all(len(pools[f'{split}_{m}'])==count for split,count in (('train',768),('validation',192),('test',384)) for m in (2,8,16,32))
    for rows in pools.values():reference.independent.check_rows(rows)
    expected_runs=[(c['seed'],arm) for c in lock['cases'] for arm in ('control','cap')]
    assert [(r['seed'],r['arm']) for r in report['runs']]==expected_runs
    checks=[]
    for case in lock['cases']:
        paired=[]
        for arm in ('control','cap'):
            run=next(r for r in report['runs'] if (r['seed'],r['arm'])==(case['seed'],arm));check,trace=audit_run(case,run,pools,lock['updates']);checks.append(check);paired.append(trace)
        for a,b in zip(*paired):
            assert torch.equal(a['indices'],b['indices']) and torch.equal(a['draws'],b['draws'])
        for name in paired[0][0]['gradients']:assert torch.equal(paired[0][0]['gradients'][name],paired[1][0]['gradients'][name]),'Initial gradients differ'
        assert study.training.replay.equal(paired[0][0]['raw_model'],paired[1][0]['raw_model']),'Initial raw step differs'
    scores=torch.load(out/'scores.pt',weights_only=True);expected={(c['seed'],a,f'{split}_{m}') for c in lock['cases'] for a in ('baseline','control','cap') for split in ('train','validation','test') for m in (2,8,16,32)}
    ids=[(c['seed'],c['arm'],c['pool']) for c in scores];assert len(ids)==len(expected) and set(ids)==expected and len(report['cells'])==len(scores)
    errors=[];targets=0
    for cell,record in zip(scores,report['cells']):
        assert (cell['seed'],cell['arm'],cell['pool'],cell['checkpoint'])==(record['seed'],record['arm'],record['pool'],record['checkpoint'])
        case=next(c for c in lock['cases'] if c['seed']==cell['seed'])
        expected_path=case['checkpoint'] if cell['arm']=='baseline' else next(r['checkpoint'] for r in report['runs'] if (r['seed'],r['arm'])==(cell['seed'],cell['arm']))
        assert cell['checkpoint']==expected_path
        path=study.ROOT/expected_path;rows=pools[cell['pool']]
        with torch.random.fork_rng(devices=[]):result=reference.audit_cell(path,rows,cell)
        reference.check_metrics(record['metrics'],reference.metrics(cell['logits'],cell['labels'],cell['repeat']));errors.append(result['max_logit_error']);targets+=cell['labels'].numel()
        state=torch.load(path,weights_only=True)['model'];center=cell['logits']-cell['logits'].mean(-1,keepdim=True)
        diagnostics=dict(head_norm=norm(state),value_norm=float(state['value.weight'].norm()),beta=float(state['write_logit'].sigmoid()),centered_logit_rms=float(center.square().mean().sqrt()))
        comparisons.compare(cell['diagnostics'],diagnostics);comparisons.compare(record['diagnostics'],diagnostics)
        if cell['pool']=='train_32':
            baseline=next(c for c in scores if c['seed']==cell['seed'] and c['arm']=='baseline' and c['pool']=='train_32');mask=baseline['logits'].argmax(-1)!=baseline['labels']
            z=cell['logits'];y=cell['labels'];good=z.argmax(-1)==y;n=int(mask.sum());ce=z.logsumexp(-1)-z.gather(-1,y[...,None]).squeeze(-1)
            stats=dict(error_count=n,error_ce=float(ce[mask].sum()/n) if n else None,rescued=int((mask&good).sum()),harmed=int((~mask&~good).sum()))
            comparisons.compare(cell['fixed'],stats);comparisons.compare(record['fixed'],stats)
        else:assert 'fixed' not in cell and 'fixed' not in record
    decision=independent_decision([c for c in report['cells'] if not c['pool'].startswith('test_')],[c['seed'] for c in lock['cases']])
    comparisons.compare(report['decision'],decision);comparisons.compare(json.loads((out/'pretest_decision.json').read_text()),decision)
    assert report['evaluation_targets']==targets
    updates=len(expected_runs)*lock['updates'];count=sum(64*(2,8,16,32)[(step-1)%4] for case in lock['cases'] for _ in ('control','cap') for step in range(torch.load(study.ROOT/case['checkpoint'],weights_only=True)['step']+1,torch.load(study.ROOT/case['checkpoint'],weights_only=True)['step']+lock['updates']+1))
    assert report['training_updates']==report['optimizer_calls']==updates and report['histories']==updates*64 and report['input_events']==updates*64*32 and report['answer_targets']==count
    assert (report['device'],report['dtype'],report['batch'],report['threads'])==('cpu','float64',64,1) and report['test_is_exploratory'] is True
    assert report['peak_allocated_bytes'] is None and report['peak_reserved_bytes'] is None
    replay.check_files(lock['files']);result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),runs=len(expected_runs),cells=len(scores),evaluation_targets=targets,
        training_updates=updates,answer_targets=count,max_gradient_error=max(c['max_gradient_error'] for c in checks),max_adam_error=max(c['max_adam_error'] for c in checks),
        max_evaluation_logit_error=max(errors),protected_files=len(lock['files']))
    if persist:assert not (out/'audit.json').exists();replay.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);audit(study.OUT)
