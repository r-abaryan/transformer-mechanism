"""One-pass learned writes from supplied native/oracle shared memories."""
import argparse
import csv
import importlib.util
import json
import time
import unittest

import torch
import shared_state_feasibility as prior

ROOT=prior.ROOT
OUT=ROOT/'results/shared_state_preservation'
SOURCES=('experiments/shared_state_preservation.py','experiments/audit_shared_state_preservation.py',
         'tests/test_shared_state_preservation.py')
PROTOCOL='docs/protocols/shared-state-preservation.md'
ARMS=('native','oracle')


def read(model,s,queries):
    h,m=queries.shape
    return model.read((s[:,None].expand(-1,m,-1,-1).reshape(h*m,16,8),),queries.flatten()).reshape(h,m,6)


def frame(z,y,repeat,initial_correct,seed,arm,step,previous=None,written=None):
    metric=prior.metrics(z,y,repeat); correct=z.argmax(-1)==y
    record=dict(seed=seed,arm=arm,step=step,metrics=metric,
        initial_complete_retained=int((correct.all(-1)&initial_correct.all(-1)).sum()),
        initially_incomplete_now_complete=int((correct.all(-1)&~initial_correct.all(-1)).sum()))
    if previous is not None:
        old=previous.argmax(-1)==y; harms=old&~correct; rescues=~old&correct
        mask=torch.zeros_like(correct); mask.scatter_(1,written[:,None],True)
        record['transitions']=dict(written_harms=int((harms&mask).sum()),other_harms=int((harms&~mask).sum()),
            written_rescues=int((rescues&mask).sum()),other_rescues=int((rescues&~mask).sum()))
    return record


@torch.no_grad()
def replay(model,rows,queries,y,repeat,start,seed,arm,beta_override=None):
    before={k:v.clone() for k,v in model.state_dict().items()}; rng=torch.get_rng_state().clone()
    e=torch.tensor([r['events'] for r in rows]); k=model.keys(e[...,0]); v=model.value(e[...,1]); q=model.keys(queries)
    beta=model.write_logit.sigmoid() if beta_override is None else torch.tensor(beta_override,dtype=torch.float64)
    s=start.clone(); logits=[read(model,s,queries)]; initial=logits[0].argmax(-1)==y
    records=[frame(logits[0],y,repeat,initial,seed,arm,0)]
    losses_before=[]; losses_after=[]; identity=[]; examples=[]
    with torch.no_grad():
        for t in range(e.shape[1]):
            key=k[:,t]; value=v[:,t]; residual=value-(key[:,:,None]*s).sum(1)
            loss_before=.5*residual.square().sum(-1)
            s=s+beta*key[:,:,None]*residual[:,None,:]
            loss_after=.5*(value-(key[:,:,None]*s).sum(1)).square().sum(-1)
            z=read(model,s,queries); delta=beta*(q*key[:,None]).sum(-1)[...,None]*model.output(residual).sub(model.output.bias)[:,None]
            error=float((z-logits[-1]-delta).abs().max()); identity.append(error)
            slot=(queries==e[:,t,0,None]).long().argmax(-1)
            record=frame(z,y,repeat,initial,seed,arm,t+1,logits[-1],slot)
            record.update(write_loss_before_mean=float(loss_before.mean()),write_loss_after_mean=float(loss_after.mean()),
                write_loss_decreased=int((loss_after<loss_before-1e-9).sum()),
                write_loss_increased=int((loss_after>loss_before+1e-9).sum()),
                write_loss_unresolved=int(((loss_after-loss_before).abs()<=1e-9).sum()),delta_identity_max_error=error)
            records.append(record)
            if arm=='oracle' and not examples:
                harms=(logits[-1].argmax(-1)==y)&(z.argmax(-1)!=y)
                if harms.any():
                    h,j=torch.nonzero(harms,as_tuple=False)[0].tolist(); target=int(y[h,j]); competitor=int(z[h,j].argmax())
                    old_gap=float(logits[-1][h,j,target]-logits[-1][h,j,competitor])
                    new_gap=float(z[h,j,target]-z[h,j,competitor])
                    examples.append(dict(seed=seed,step=t+1,history=h,query=int(queries[h,j]),
                        written_key=int(e[h,t,0]),target=target,new_winner=competitor,beta=float(beta),
                        key_dot=float((q[h,j]*key[h]).sum()),gap_before=old_gap,gap_after=new_gap,
                        gap_change=new_gap-old_gap,predicted_gap_change=float(delta[h,j,target]-delta[h,j,competitor]),
                        write_loss_before=float(loss_before[h]),write_loss_after=float(loss_after[h])))
            logits.append(z);losses_before.append(loss_before);losses_after.append(loss_after)
    z=torch.stack(logits).detach(); complete=(z.argmax(-1)==y).all(-1)
    first=[]
    for h in range(len(rows)):
        bad=torch.where(~complete[:,h])[0]
        first.append(int(bad[0]) if len(bad) else None)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(seed=seed,arm=arm,queries=queries,labels=y,repeat=repeat,initial_state=start,final_state=s.detach(),
        logits=z,write_loss_before=torch.stack(losses_before),write_loss_after=torch.stack(losses_after),
        records=records,first_incomplete_step=first,examples=examples)


def summarize(bundles):
    result=[]
    for b in bundles:
        z=b['logits']; y=b['labels']; good=z.argmax(-1)==y; maps=good.all(-1); old=good[0]; last=good[-1]
        result.append(dict(seed=b['seed'],arm=b['arm'],histories=len(y),start_correct=int(old.sum()),
            final_correct=int(last.sum()),start_maps=int(maps[0].sum()),final_maps=int(maps[-1].sum()),
            initially_complete_ever_broken=int((maps[0]&(~maps).any(0)).sum()),
            never_incomplete=int(maps.all(0).sum()),final_rescues_from_start=int((~old&last).sum()),
            final_harms_from_start=int((old&~last).sum()),
            first_step_broken=int((maps[0]&~maps[1]).sum()),
            written_harms=sum(r.get('transitions',{}).get('written_harms',0) for r in b['records']),
            other_harms=sum(r.get('transitions',{}).get('other_harms',0) for r in b['records']),
            write_loss_decreases=sum(r.get('write_loss_decreased',0) for r in b['records']),
            write_loss_increases=sum(r.get('write_loss_increased',0) for r in b['records']),
            write_loss_unresolved=sum(r.get('write_loss_unresolved',0) for r in b['records'])))
    return result


def execute(out,source,fixture=False):
    assert not out.exists(), 'Preserve evidence'
    source_report=json.loads((source/'report.json').read_text()); source_lock=json.loads((source/'lock.json').read_text())
    audit=json.loads((source/'audit.json').read_text());assert audit['passed'] and audit['report_sha256']==prior.digest(source/'report.json')
    assert source_report['witnesses_sha256']==prior.digest(source/'witnesses.pt')
    files=dict(source_lock['files'])
    for p in (*SOURCES,PROTOCOL):files[p]=prior.digest(ROOT/p)
    for name in ('report.json','audit.json','lock.json','witnesses.pt'):
        p=source/name;files[p.relative_to(ROOT).as_posix()]=prior.digest(p)
    if not fixture:
        path=ROOT/'results/shared_state_preservation_preflight.json'; ready=json.loads(path.read_text())
        assert ready['passed'] and ready['sources']=={p:files[p] for p in SOURCES}
        files[path.relative_to(ROOT).as_posix()]=prior.digest(path)
    prior.check_files(files);out.mkdir()
    prior.write(out/'lock.json',dict(files=files,source=source.relative_to(ROOT).as_posix(),fixture=fixture,arms=list(ARMS),
        passes=1,guard=1e-9,versions=prior.versions(),cases=source_lock['cases']))
    source_bundles=torch.load(source/'witnesses.pt',weights_only=True)
    bundles=[];values=[]; start_time=time.perf_counter();cold_writes=0
    with torch.no_grad():
        for case,saved in zip(source_lock['cases'],source_bundles):
            rows=json.loads((ROOT/case['data']).read_text())['pools'][case['pool']]
            ids,y,rs=prior.labels(rows);assert torch.equal(ids,saved['queries']) and torch.equal(y,saved['labels'])
            assert all(r['outcome']=='positive' for r in saved['records'])
            model=prior.historical.parent.previous.load_model(ROOT/case['checkpoint'])
            cold=model.prefill(torch.tensor([r['events'] for r in rows]))[0]
            assert torch.allclose(cold,saved['native_states'],atol=1e-9,rtol=1e-9), 'Cold native state'
            values.append(dict(seed=case['seed'],logits=model.output(model.value.weight).detach().tolist()))
            cold_writes+=sum(len(r['events']) for r in rows)
            for arm in ARMS:
                state=cold if arm=='native' else saved['states']
                bundle=replay(model,rows,ids,y,rs,state,case['seed'],arm)
                initial=saved['native_logits'] if arm=='native' else saved['oracle_logits']
                assert torch.allclose(bundle['logits'][0],initial,atol=1e-9,rtol=1e-9), 'Initial scores'
                bundles.append(bundle);print(f"seed {case['seed']} {arm}: replay complete",flush=True)
    torch.save(bundles,out/'traces.pt')
    records=[r for b in bundles for r in b['records']]
    flat=[]
    for r in records:
        row={k:v for k,v in r.items() if k not in ('metrics','transitions')}
        for group,stats in r['metrics'].items():
            if group=='complete_maps':row[group]=stats
            else:
                for key,value in stats.items():row[group+'_'+key]=value
        row.update(r.get('transitions',{}));flat.append(row)
    fields=list(dict.fromkeys(k for row in flat for k in row))
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    report=dict(status='complete',fixture=fixture,device='cpu',dtype='float64',training_updates=0,optimizer_calls=0,
        replay_writes=sum(b['write_loss_before'].numel() for b in bundles),cold_fidelity_writes=cold_writes,
        answer_outputs=sum(b['labels'].numel()*len(b['logits']) for b in bundles),cells=len(records),
        summary=summarize(bundles),value_logits=values,examples=[x for b in bundles for x in b['examples']],
        seconds=time.perf_counter()-start_time,lock_sha256=prior.digest(out/'lock.json'),
        traces_sha256=prior.digest(out/'traces.pt'),metrics_sha256=prior.digest(out/'metrics.csv'))
    prior.check_files(files);prior.write(out/'report.json',report);print(json.dumps(report['summary']),flush=True)


def preflight():
    spec=importlib.util.spec_from_file_location('test_shared_state_preservation',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    path=ROOT/'results/shared_state_preservation_preflight.json';assert not path.exists()
    prior.write(path,dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={p:prior.digest(ROOT/p) for p in SOURCES}))
    assert result.wasSuccessful()


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:execute(OUT,prior.OUT)
