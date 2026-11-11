"""Matrix replay and independent map, transition and local-loss reductions."""
import csv
import json

import torch
import shared_state_preservation as study
import audit_shared_state_feasibility as reference


def reduced(z,y,repeat):
    metric=reference.existing.metrics(z,y,repeat)
    return {**metric['groups'],'complete_maps':metric['complete_maps']['correct']}


def audit(out,persist=True):
    prior=study.prior
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text())
    prior.check_files(lock['files'])
    assert report['lock_sha256']==prior.digest(out/'lock.json')
    assert report['traces_sha256']==prior.digest(out/'traces.pt')
    assert report['metrics_sha256']==prior.digest(out/'metrics.csv')
    assert report['training_updates']==report['optimizer_calls']==0 and report['status']=='complete'
    assert lock['passes']==1 and lock['arms']==['native','oracle'] and lock['guard']==1e-9
    assert lock['versions']==prior.versions()
    source=study.ROOT/lock['source'];saved=torch.load(source/'witnesses.pt',weights_only=True)
    traces=torch.load(out/'traces.pt',weights_only=True)
    expected_ids=[(c['seed'],a) for c in lock['cases'] for a in ('native','oracle')]
    assert [(b['seed'],b['arm']) for b in traces]==expected_ids, 'Missing/reordered arm'
    summaries=[];values=[];all_records=[];examples=[];max_error=0.;outputs=0;writes=0;cold_writes=0
    with torch.no_grad():
        for case,original in zip(lock['cases'],saved):
            rows=json.loads((study.ROOT/case['data']).read_text())['pools'][case['pool']]
            queries=[];targets=[];repeats=[]
            for row in rows:
                mapping={};counts={}
                for key,value in row['events']:
                    assert key not in mapping or mapping[key]==value, 'Inconsistent values'
                    mapping[key]=value;counts[key]=counts.get(key,0)+1
                assert mapping[row['query']]==row['label'], 'Assigned label'
                keys=sorted(counts);queries.append(keys);targets.append([mapping[k] for k in keys]);repeats.append([counts[k]>1 for k in keys])
            ids=torch.tensor(queries);y=torch.tensor(targets);repeat=torch.tensor(repeats)
            model=prior.historical.parent.previous.load_model(study.ROOT/case['checkpoint'])
            raw=model.key.weight;norm=raw.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt();keys=raw/norm
            w=model.output.weight;b=model.output.bias;beta=model.write_logit.sigmoid()
            e=torch.tensor([r['events'] for r in rows]);k=keys[e[...,0]];v=model.value.weight[e[...,1]];q=keys[ids]
            value_scores=model.value.weight@w.T+b
            values.append(dict(seed=case['seed'],logits=value_scores.tolist()))
            cold=torch.zeros_like(original['native_states']);eye=torch.eye(16,dtype=torch.float64)
            for t in range(e.shape[1]):
                cold=(eye-beta*k[:,t,:,None]*k[:,t,None,:])@cold+beta*k[:,t,:,None]*v[:,t,None,:]
            assert torch.allclose(cold,original['native_states'],atol=1e-9,rtol=1e-9), 'Cold source memory'
            cold_writes+=len(rows)*e.shape[1]
            for arm in ('native','oracle'):
                trace=next(x for x in traces if x['seed']==case['seed'] and x['arm']==arm)
                assert torch.equal(ids,trace['queries']) and torch.equal(y,trace['labels']), 'Targets/order'
                assert torch.equal(repeat,trace['repeat']), 'Groups'
                initial=original['native_states'] if arm=='native' else original['states']
                assert torch.equal(initial,trace['initial_state']), 'Initial state'
                s=initial.clone();zs=[];metrics=[];first_example=[]
                assert len(trace['records'])==len(trace['logits'])==e.shape[1]+1, 'Frame coverage'
                for step in range(e.shape[1]+1):
                    if step:
                        key=k[:,step-1];value=v[:,step-1];before_s=s.clone()
                        residual=value-(key[:,None,:]@s).squeeze(1)
                        loss_before=.5*(residual*residual).sum(-1)
                        s=(eye-beta*key[:,:,None]*key[:,None,:])@s+beta*key[:,:,None]*value[:,None,:]
                        residual_after=value-(key[:,None,:]@s).squeeze(1);loss_after=.5*(residual_after*residual_after).sum(-1)
                        expected_loss=loss_before*(1-beta*(key*key).sum(-1)).square()
                        assert torch.allclose(loss_after,expected_loss,atol=1e-9,rtol=1e-9), 'Residual contraction'
                        assert torch.allclose(loss_before,trace['write_loss_before'][step-1],atol=1e-9,rtol=1e-9), 'Before residual'
                        assert torch.allclose(loss_after,trace['write_loss_after'][step-1],atol=1e-9,rtol=1e-9), 'After residual'
                    z=(q@s)@w.T+b
                    error=float((z-trace['logits'][step]).abs().max());max_error=max(max_error,error)
                    assert torch.allclose(z,trace['logits'][step],atol=1e-9,rtol=1e-9), 'Replay logits'
                    assert torch.equal(z.argmax(-1),trace['logits'][step].argmax(-1)), 'Replay winners'
                    correct=z.argmax(-1)==y
                    if not step:initial_correct=correct.clone()
                    record=dict(seed=case['seed'],arm=arm,step=step,metrics=reduced(z,y,repeat),
                        initial_complete_retained=int((correct.all(-1)&initial_correct.all(-1)).sum()),
                        initially_incomplete_now_complete=int((correct.all(-1)&~initial_correct.all(-1)).sum()))
                    if step:
                        harms=(zs[-1].argmax(-1)==y)&~correct;rescues=(zs[-1].argmax(-1)!=y)&correct
                        counts=dict(written_harms=0,other_harms=0,written_rescues=0,other_rescues=0)
                        for i in range(len(rows)):
                            slot=queries[i].index(rows[i]['events'][step-1][0])
                            counts['written_harms']+=int(harms[i,slot]);counts['other_harms']+=int(harms[i].sum())-int(harms[i,slot])
                            counts['written_rescues']+=int(rescues[i,slot]);counts['other_rescues']+=int(rescues[i].sum())-int(rescues[i,slot])
                        predicted=beta*(q@key[:,:,None]).squeeze(-1)[...,None]*(residual@w.T)[:,None,:]
                        err=float((z-zs[-1]-predicted).abs().max());max_error=max(max_error,err)
                        assert err<=1e-9, 'Local score change'
                        record.update(transitions=counts,write_loss_before_mean=float(loss_before.mean()),write_loss_after_mean=float(loss_after.mean()),
                            write_loss_decreased=int((loss_after<loss_before-1e-9).sum()),write_loss_increased=int((loss_after>loss_before+1e-9).sum()),
                            write_loss_unresolved=int(((loss_after-loss_before).abs()<=1e-9).sum()),delta_identity_max_error=err)
                        if arm=='oracle' and not first_example and harms.any():
                            i,j=next((i,j) for i in range(len(rows)) for j in range(len(queries[i])) if harms[i,j])
                            target=int(y[i,j]);competitor=int(z[i,j].argmax());old_gap=float(zs[-1][i,j,target]-zs[-1][i,j,competitor]);new_gap=float(z[i,j,target]-z[i,j,competitor])
                            first_example.append(dict(seed=case['seed'],step=step,history=i,query=int(ids[i,j]),written_key=int(e[i,step-1,0]),
                                target=target,new_winner=competitor,beta=float(beta),key_dot=float(q[i,j]@key[i]),gap_before=old_gap,gap_after=new_gap,
                                gap_change=new_gap-old_gap,predicted_gap_change=float(predicted[i,j,target]-predicted[i,j,competitor]),
                                write_loss_before=float(loss_before[i]),write_loss_after=float(loss_after[i])))
                    reference.compare(trace['records'][step],record);metrics.append(record);zs.append(z)
                assert torch.allclose(s,trace['final_state'],atol=1e-9,rtol=1e-9), 'Final state'
                reference.compare(trace['examples'],first_example);examples.extend(first_example)
                good=torch.stack([z.argmax(-1)==y for z in zs]);maps=good.all(-1)
                first=[]
                for h in range(len(rows)):
                    missing=[t for t in range(len(zs)) if not maps[t,h]]
                    first.append(missing[0] if missing else None)
                assert trace['first_incomplete_step']==first, 'First damage'
                old=good[0];last=good[-1]
                summaries.append(dict(seed=case['seed'],arm=arm,histories=len(rows),start_correct=int(old.sum()),final_correct=int(last.sum()),
                    start_maps=int(maps[0].sum()),final_maps=int(maps[-1].sum()),initially_complete_ever_broken=int((maps[0]&(~maps).any(0)).sum()),
                    never_incomplete=int(maps.all(0).sum()),final_rescues_from_start=int((~old&last).sum()),final_harms_from_start=int((old&~last).sum()),
                    first_step_broken=int((maps[0]&~maps[1]).sum()),written_harms=sum(m.get('transitions',{}).get('written_harms',0) for m in metrics),
                    other_harms=sum(m.get('transitions',{}).get('other_harms',0) for m in metrics),
                    write_loss_decreases=sum(m.get('write_loss_decreased',0) for m in metrics),write_loss_increases=sum(m.get('write_loss_increased',0) for m in metrics),
                    write_loss_unresolved=sum(m.get('write_loss_unresolved',0) for m in metrics)))
                all_records.extend(trace['records']);outputs+=y.numel()*len(zs);writes+=y.shape[0]*e.shape[1]
    reference.compare(report['summary'],summaries);reference.compare(report['examples'],examples);reference.compare(report['value_logits'],values)
    assert report['cells']==len(all_records) and report['answer_outputs']==outputs and report['replay_writes']==writes and report['cold_fidelity_writes']==cold_writes
    if not lock['fixture']:
        assert len(all_records)==2*2*33 and outputs==2*2*33*768*32 and writes==2*2*768*32 and cold_writes==2*768*32
    with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==len(all_records), 'CSV coverage'
    for row,record in zip(csv_rows,all_records):
        flat={k:v for k,v in record.items() if k not in ('metrics','transitions')}
        for group,stats in record['metrics'].items():
            if group=='complete_maps':flat[group]=stats
            else:
                for key,value in stats.items():flat[group+'_'+key]=value
        flat.update(record.get('transitions',{}))
        for key,expected in flat.items():
            if expected is None:assert row[key]=='', key
            elif isinstance(expected,float):reference.close(float(row[key]),expected)
            else:assert row[key]==str(expected),key
        assert all(row[key]=='' for key in row if key not in flat)
    prior.check_files(lock['files'])
    result=dict(passed=True,report_sha256=prior.digest(out/'report.json'),cells=len(all_records),answer_outputs=outputs,
        replay_writes=writes,cold_fidelity_writes=cold_writes,max_equation_error=max_error,protected_files=len(lock['files']),training_updates=0)
    if persist:
        assert not (out/'audit.json').exists();prior.write(out/'audit.json',result)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    torch.set_num_threads(1);audit(study.OUT)
