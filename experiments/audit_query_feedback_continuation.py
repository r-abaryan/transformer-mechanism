"""Independent continuation stopping, stream, checkpoint and metric checks."""
import json

import torch
import query_feedback_continuation as study
import audit_query_feedback_training as reference


def fit(training,threshold):
    assert set(training)=={'2','8','16','32'}
    for m,record in training.items():
        all_group=record['groups']['all'];expected='unique' if int(m)==32 else 'repeat'
        assert all_group['count']>0
        assert record['groups'][expected]==all_group,'Missing relevant group'
        assert record['groups']['repeat' if expected=='unique' else 'unique']==dict(count=0,ce=None,accuracy=None)
    return min(record['groups']['all']['accuracy'] for record in training.values())>=threshold


def audit_training(item,data,cells,prior_steps,initial_alpha):
    seed=item['seed'];start=item['start'];stop=item['stop'];interval=item['interval']
    assert start<stop<=item['cap'] and len(item['steps'])==stop-start
    assert [s['step'] for s in item['snapshots']]==list(range(start,stop+1,interval))
    assert item['added_updates']==stop-start and item['added_histories']==64*(stop-start)
    assert item['added_input_events']==2048*(stop-start) and item['parameters']==615
    targets=sum(64*(2,8,16,32)[(t-1)%4] for t in range(start+1,stop+1))
    assert item['added_answer_targets']==targets
    checkpoint=study.ROOT/item['parent_checkpoint']
    assert study.parent.replay.digest(checkpoint)==item['parent_sha256']
    original=torch.load(checkpoint,weights_only=True);assert original['step']==start
    resumed=torch.load(study.ROOT/item['snapshots'][0]['checkpoint'],weights_only=True)
    for name in ('model','optimizer'):assert study.parent.replay.equal(original[name],resumed[name]),'Changed resume state'
    assert len(prior_steps)==start
    records=prior_steps+item['steps']
    adam=reference.adam_audit.check_steps(records,initial_alpha)
    fits=[];checks=[]
    by={(c['step'],c['pool']):c for c in cells if c['seed']==seed}
    hr=torch.Generator().manual_seed(seed+10000);qr=torch.Generator().manual_seed(seed+20000)
    snapshots={s['step']:s for s in item['snapshots']}
    for t in range(stop+1):
        if t:
            m=(2,8,16,32)[(t-1)%4]
            torch.randint(len(data['pools'][f'train_{m}']),(64,),generator=hr)
            torch.randint(m,(64,m),generator=qr)
        if t not in snapshots:continue
        snap=snapshots[t];path=study.ROOT/snap['checkpoint']
        assert study.parent.replay.digest(path)==snap['sha256']
        cp=torch.load(path,weights_only=True);assert cp['step']==t
        assert torch.equal(cp['history_rng'],hr.get_state()) and torch.equal(cp['query_rng'],qr.get_state()),'Wrong stream state'
        assert all(int(s['step'])==t for s in cp['optimizer']['state'].values())
        assert snap['beta']==float(cp['model']['write_logit'].sigmoid())
        if t:
            state=cp['optimizer']['state'][cp['optimizer']['param_groups'][0]['params'][0]]
            row=records[t-1]
            assert float(cp['model']['write_logit'])==row['alpha_after']
            assert float(state['exp_avg'])==row['exp_avg'] and float(state['exp_avg_sq'])==row['exp_avg_sq']
        for m in (2,8,16,32):
            cell=by[t,f'train_{m}'];assert cell['checkpoint']==snap['checkpoint']
            assert cell['metrics']==snap['training'][str(m)]
            checks.append(reference.audit_cell(path,data['pools'][f'train_{m}'],cell))
        passed=fit(snap['training'],item['fit_threshold']);assert snap['fit']==passed;fits.append(passed)
    assert not any(fits[:-1]),'Continued beyond first eligible training check'
    assert item['training_fit']==fits[-1]
    assert item['stop_reason']==('training_fit' if fits[-1] else 'budget_cap')
    if not fits[-1]:assert stop==item['cap'],'Stopped before budget cap without fit'
    assert item['final_checkpoint']==item['snapshots'][-1]['checkpoint'] and item['beta']==item['snapshots'][-1]['beta']
    exposure_path=study.ROOT/item['exposure_path']
    assert study.parent.replay.digest(exposure_path)==item['exposure_sha256']
    exposure=torch.load(exposure_path,weights_only=True)
    stream=reference.audit_stream(seed,'exhaustive',{m:data['pools'][f'train_{m}'] for m in (2,8,16,32)},stop,exposure)
    for name in ('history_stream_sha256','potential_draws_sha256','actual_queries_labels_sha256'):assert item[name]==stream[name]
    assert stream['answer_targets']==sum(int(x.sum()) for x in exposure.values())
    for m,x in exposure.items():assert item['exposure'][str(m)]==dict(targets=int(x.sum()),covered=int((x>0).sum()),
        total=x.numel(),min_targets=int(x.min()),max_targets=int(x.max()))
    return dict(seed=seed,added_scalar_adam_steps=stop-start,scalar_steps_including_parent=stop,
        max_adam_error=adam['max_adam_error'],cells=len(checks),targets=sum(c['targets'] for c in checks),
        max_logit_error=max(c['max_logit_error'] for c in checks))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=study.OUT;report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    assert report['status']=='complete' and report['lock_sha256']==study.parent.replay.digest(out/'lock.json')
    study.parent.replay.check_files(lock['files'])
    assert study.parent.replay.digest(out/'scores.pt')==report['scores_sha256']
    assert study.parent.replay.digest(out/'training_decision.json')==report['training_decision_sha256']
    decision=json.loads((out/'training_decision.json').read_text())
    assert decision['runs']==report['runs'] and all(c['pool'].startswith('train_') for c in decision['cells'])
    assert report['heldout_previously_opened'] and not report['validation_test_for_selection']
    assert report['all_seed_training_fit']==decision['all_seed_training_fit']==all(r['training_fit'] for r in report['runs'])
    assert [r['seed'] for r in report['runs']]==[3461,3463]
    old=json.loads((study.parent.OUT/'report.json').read_text())
    data=json.loads((study.parent.parent.OUT/'development_data.json').read_text())
    fresh=json.loads((study.parent.OUT/'test_data.json').read_text());pools={**data['pools'],**fresh['pools']}
    saved=torch.load(out/'scores.pt',weights_only=True);expected=set()
    for item in report['runs']:
        expected|={(item['seed'],'exhaustive',s['step'],f'train_{m}') for s in item['snapshots'] for m in (2,8,16,32)}
        expected|={(item['seed'],'exhaustive',item['stop'],f'{split}_{m}') for split in ('validation','test') for m in (2,8,16,32)}
    assert len(saved)==len(report['cells'])==len(expected)==sum(4*len(r['snapshots'])+8 for r in report['runs'])
    assert {(c['seed'],c['arm'],c['step'],c['pool']) for c in saved}==expected
    for cell,record in zip(saved,report['cells']):
        assert record=={name:cell[name] for name in ('seed','arm','step','pool','checkpoint','metrics')}
    assert [c for c in report['cells'] if c['pool'].startswith('train_')]==decision['cells']
    runs=[];heldout=[]
    for item in report['runs']:
        assert (item['start'],item['cap'],item['interval'],item['fit_threshold'])==(3000,12000,1000,.99)
        prior=next(r for r in old['runs'] if r['seed']==item['seed'] and r['arm']=='exhaustive')
        assert item['parent_checkpoint']==prior['final_checkpoint']
        for new,previous in [('prefix_history_sha256','history_stream_sha256'),('prefix_draws_sha256','potential_draws_sha256'),
                             ('prefix_targets_sha256','actual_queries_labels_sha256')]:assert item[new]==prior[previous]
        initial=torch.load(study.ROOT/prior['snapshots'][0]['checkpoint'],weights_only=True)
        runs.append(audit_training(item,data,saved,prior['steps'],float(initial['model']['write_logit'])))
        for cell in saved:
            if cell['seed']==item['seed'] and not cell['pool'].startswith('train_'):
                assert cell['checkpoint']==item['final_checkpoint']
                heldout.append(reference.audit_cell(study.ROOT/cell['checkpoint'],pools[cell['pool']],cell))
        print(json.dumps(dict(seed=item['seed'],audited_stop=item['stop'])),flush=True)
    targets=sum(c['targets'] for c in runs)+sum(c['targets'] for c in heldout)
    assert targets==report['targets']==sum((768*len(r['snapshots'])+192+384)*58 for r in report['runs'])
    assert report['cells_count']==len(saved)
    for field in ('updates','histories','input_events','answer_targets'):
        assert report['total_added_'+field]==sum(r['added_'+field] for r in report['runs'])
    study.parent.replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.parent.replay.digest(out/'report.json'),runs=runs,cells=len(saved),targets=targets,
        added_scalar_adam_steps=sum(r['added_scalar_adam_steps'] for r in runs),
        scalar_steps_including_parent=sum(r['scalar_steps_including_parent'] for r in runs),
        max_logit_error=max([r['max_logit_error'] for r in runs]+[c['max_logit_error'] for c in heldout]),
        protected_files=len(lock['files']),training_only_stopping=True)
    assert not (out/'audit.json').exists();study.parent.replay.write(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__=='__main__':main()
