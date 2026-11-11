"""Independent reset scope, scalar Adam, streams and complete evaluation."""
import copy
import json

import torch
import memory_first_moment_reset as study
import audit_readout_adam_transport as adam
import audit_query_feedback_training as reference
import audit_feedback_loss_decomposition as prior
import audit_adam_memory_error_transport as transport_audit


def reset_scope(source,trace,arm):
    assert study.training.replay.equal(source['model'],trace['before_model']), 'Starting weights changed'
    expected=copy.deepcopy(source['optimizer'])
    if arm=='reset':
        for name,pid in zip(trace['names'],expected['param_groups'][0]['params']):
            if name in study.transport.MEMORY:expected['state'][pid]['exp_avg']=torch.zeros_like(expected['state'][pid]['exp_avg'])
    assert study.training.replay.equal(expected,trace['before_optimizer']), 'Reset scope differs'


def audit_run(path,pools,traces,arm):
    source=torch.load(path,weights_only=True);start=source['step'];assert len(traces)==4
    assert [t['step'] for t in traces]==list(range(start+1,start+5))
    reset_scope(source,traces[0],arm)
    hr=torch.Generator();hr.set_state(source['history_rng']);qr=torch.Generator();qr.set_state(source['query_rng'])
    errors=[]
    for i,t in enumerate(traces):
        assert t['arm']==arm and t['start']==start and t['load']==(2,8,16,32)[(t['step']-1)%4]
        rows=pools[f"train_{t['load']}"]
        assert torch.equal(t['history_before'],hr.get_state()) and torch.equal(t['query_before'],qr.get_state())
        ids=torch.randint(len(rows),(64,),generator=hr);draws=torch.randint(t['load'],(64,t['load']),generator=qr)
        assert torch.equal(ids,t['indices']) and torch.equal(draws,t['draws']), 'Sampled stream differs'
        assert torch.equal(t['history_after'],hr.get_state()) and torch.equal(t['query_after'],qr.get_state())
        if i:
            assert study.training.replay.equal(t['before_model'],traces[i-1]['after_model'])
            assert study.training.replay.equal(t['before_optimizer'],traces[i-1]['after_optimizer'])
        selected=[rows[int(j)] for j in ids];gradient_error,logit_error=adam.matrix_gradients(t,selected)
        _,adam_error=adam.scalar_adam(t)
        errors.append(dict(max_gradient_error=gradient_error,max_training_logit_error=logit_error,max_adam_error=adam_error))
    return errors


def paired_scope(control,reset):
    assert len(control)==len(reset)==4
    for a,b in zip(control,reset):
        for key in ('indices','draws','queries','labels','history_before','history_after','query_before','query_after'):
            assert torch.equal(a[key],b[key]), 'Paired stream differs'
    for n,g in control[0]['gradients'].items():assert torch.equal(g,reset[0]['gradients'][n]), 'First gradient changed'
    for n in ('output.weight','output.bias'):
        assert torch.equal(control[0]['after_model'][n],reset[0]['after_model'][n]), 'First classifier update changed'
    parts,_=transport_audit.scalar_parts(control[0])
    for n in study.transport.MEMORY:
        actual=reset[0]['after_model'][n]-reset[0]['before_model'][n]
        assert float((actual-parts['current'][n]).abs().max())<=1e-11, 'Reset first update differs from current numerator'


def audit_evaluation(path,rows,cell,record,mask=None):
    error=reference.audit_cell(path,rows,cell)['max_logit_error'];reference.check_metrics(record['metrics'],reference.metrics(cell['logits'],cell['labels'],cell['repeat']))
    if mask is not None:
        expected=transport_audit.fixed(cell['logits'],cell['labels'],mask)
        prior.compare(cell['fixed'],expected);prior.compare(record['fixed'],expected)
    else:assert 'fixed' not in cell and 'fixed' not in record
    return error


def decision(immediate,final):
    records=[]
    for seed in (3461,3463):
        def get(cells,arm,pool):return next(c for c in cells if (c['seed'],c['arm'],c['pool'])==(seed,arm,pool))
        c,r=(get(immediate,a,'train_32') for a in ('control','reset'))
        immediate_delta=r['fixed']['error_ce']-c['fixed']['error_ce']
        c,r=(get(final,a,'train_32') for a in ('control','reset'))
        error_delta=r['fixed']['error_ce']-c['fixed']['error_ce'];accuracy_delta=r['metrics']['groups']['all']['accuracy']-c['metrics']['groups']['all']['accuracy']
        delta=sum(get(final,'reset',f'train_{m}')['metrics']['groups']['all']['ce']-get(final,'control',f'train_{m}')['metrics']['groups']['all']['ce'] for m in (2,8,16,32))/4
        records.append(dict(seed=seed,immediate_error_ce_delta=immediate_delta,final_error_ce_delta=error_delta,
            final_train32_accuracy_delta=accuracy_delta,final_equal_load_ce_delta=delta,immediate_benefit=immediate_delta < -1e-10,
            four_update_benefit=error_delta < -1e-10 and delta < -1e-10 and accuracy_delta>=0))
    return dict(seeds=records,immediate_benefit_all=all(r['immediate_benefit'] for r in records),four_update_benefit_all=all(r['four_update_benefit'] for r in records))


def main():
    torch.set_num_threads(1);out=study.OUT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text());replay.check_files(lock['files'])
    assert report['status']=='complete' and report['lock_sha256']==replay.digest(out/'lock.json')
    for name in ('scores','traces'):assert report[f'{name}_sha256']==replay.digest(out/f'{name}.pt')
    for path,h in report['checkpoints'].items():assert replay.digest(study.ROOT/path)==h
    scores=torch.load(out/'scores.pt',weights_only=True);traces=torch.load(out/'traces.pt',weights_only=True);starts,pools=study.inputs()
    training_errors=[];evaluation_errors=[]
    expected_trace_ids=[(s,a,t) for s in (3461,3463) for a in ('control','reset') for t in range(11004,11008)]
    assert [(t['seed'],t['arm'],t['step']) for t in traces]==expected_trace_ids
    expected_cells=[(s,'baseline',11003,f'{split}_{m}') for s in (3461,3463) for split in ('train','validation','test') for m in (2,8,16,32)]
    expected_cells += [(s,a,11004,'train_32') for s in (3461,3463) for a in ('control','reset')]
    expected_cells += [(s,a,11007,f'{split}_{m}') for s in (3461,3463) for a in ('control','reset') for split in ('train','validation','test') for m in (2,8,16,32)]
    assert [(c['seed'],c['arm'],c['step'],c['pool']) for c in scores]==expected_cells
    records=report['baseline']+report['immediate']+report['final']
    assert [(c['seed'],c['arm'],c['step'],c['pool']) for c in records]==expected_cells
    original=json.loads((study.transport.replay_source.continuation.OUT/'report.json').read_text())
    for binding,source in zip(report['starts'],starts):
        seed=binding['seed'];assert seed==source['seed'];path=study.ROOT/binding['path']
        assert replay.digest(path)==binding['sha256'];saved_start=torch.load(path,weights_only=True)
        assert replay.equal(saved_start,study.start_payload(source))
        runs={a:[t for t in traces if t['seed']==seed and t['arm']==a] for a in ('control','reset')}
        for arm,run in runs.items():
            training_errors.extend(audit_run(path,pools,run,arm))
            for t in run:
                cp=torch.load(out/f"{seed}_{arm}_{t['step']}.pt",weights_only=True)
                assert cp['step']==t['step'] and replay.equal(cp['model'],t['after_model']) and replay.equal(cp['optimizer'],t['after_optimizer'])
                assert torch.equal(cp['history_rng'],t['history_after']) and torch.equal(cp['query_rng'],t['query_after'])
        paired_scope(runs['control'],runs['reset'])
        assert replay.equal(runs['control'][0]['after_model'],source['after_model']) and replay.equal(runs['control'][0]['after_optimizer'],source['after_optimizer'])
        historical=next(r for r in original['runs'] if r['seed']==seed)
        assert [study.transport.replay_source.beta_record(t) for t in runs['control']]==[next(r for r in historical['steps'] if r['step']==t) for t in range(11004,11008)]
        base=next(c for c in scores if (c['seed'],c['arm'],c['pool'])==(seed,'baseline','train_32'))
        mask=base['logits'].argmax(-1)!=base['labels']
        for cell,record in zip(scores,records):
            if cell['seed']!=seed:continue
            for k in ('seed','arm','step','pool','checkpoint'):assert cell[k]==record[k]
            evaluation_errors.append(audit_evaluation(study.ROOT/cell['checkpoint'],pools[cell['pool']],cell,record,mask if cell['pool']=='train_32' else None))
    prior.compare(report['criteria'],decision(report['immediate'],report['final']))
    assert len(evaluation_errors)==76 and sum(c['labels'].numel() for c in scores)==566016
    assert report['optimizer_calls']==len(traces)==16 and report['intervention_updates']==report['control_reproduction_updates']==8
    assert report['new_post12000_updates']==0 and report['updates_per_arm']==4 and report['batch']==64
    assert report['histories']==sum(len(t['indices']) for t in traces)==1024 and report['input_events']==32768
    assert report['answer_targets']==sum(t['labels'].numel() for t in traces)==14848 and report['mean_targets_per_update']==928
    assert report['gpu_peak_allocated_bytes']==report['gpu_peak_reserved_bytes']==0 and report['device']=='cpu' and report['dtype']=='float64'
    replay.check_files(lock['files']);result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),optimizer_calls=16,evaluation_cells=76,evaluated_answers=566016,
        max_evaluation_logit_error=max(evaluation_errors),**{k:max(e[k] for e in training_errors) for k in training_errors[0]},protected_files=len(lock['files']))
    p=out/'audit.json';assert not p.exists();replay.write(p,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
