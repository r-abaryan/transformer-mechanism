"""Independent query policies, stream reconstruction and full-pool evaluation."""
import hashlib
import json
import math
import struct

import torch
import query_feedback_training as study
import audit_objective_beta_pressure as independent
import audit_retention_learning_replay as adam_audit


def metrics(logits,labels,repeat):
    groups=independent.metrics(logits,labels,repeat)
    scores=logits.tolist();ys=labels.tolist()
    correct=sum(all(max(range(6),key=lambda k:scores[i][j][k])==y for j,y in enumerate(row))
                for i,row in enumerate(ys))
    return dict(groups=groups,complete_maps=dict(count=len(ys),correct=correct,accuracy=correct/len(ys)))


def check_metrics(actual,expected):
    assert actual.keys()==expected.keys()
    for group,record in expected['groups'].items():
        for name,value in record.items():
            if value is None:assert actual['groups'][group][name] is None
            else:independent.close(actual['groups'][group][name],value)
    assert actual['complete_maps']==expected['complete_maps']


def audit_cell(checkpoint,rows,cell,batch=64):
    e,q,y,repeat=independent.inputs(rows,'all_queries')
    assert torch.equal(y,cell['labels']) and torch.equal(repeat,cell['repeat'])
    model=study.previous.load_model(checkpoint)
    with torch.no_grad():
        logits=torch.cat([independent.forward_tangent(model,e[i:i+batch],q[i:i+batch])[0]
                          for i in range(0,len(rows),batch)])
    error=independent.close(cell['logits'],logits)
    assert torch.equal(cell['logits'].argmax(-1),logits.argmax(-1))
    check_metrics(cell['metrics'],metrics(logits,y,repeat))
    return dict(max_logit_error=error,targets=y.numel())


def audit_stream(seed,arm,rows_by_load,steps,exposure):
    g=torch.Generator().manual_seed(seed+10000);qg=torch.Generator().manual_seed(seed+20000)
    maps={m:[dict(r['events']) for r in rows] for m,rows in rows_by_load.items()}
    counts={m:[[0]*m for _ in rows] for m,rows in rows_by_load.items()}
    history,draw_hash,target_hash=(hashlib.sha256() for _ in range(3))
    total=0
    for step in range(steps):
        m=(2,8,16,32)[step%4]
        indices=torch.randint(len(rows_by_load[m]),(64,),generator=g).tolist()
        draws=torch.randint(m,(64,m),generator=qg).tolist()
        history.update(struct.pack('<I',m));history.update(struct.pack('<64q',*indices))
        potential=[j for row in draws for j in row]
        draw_hash.update(struct.pack(f'<{len(potential)}q',*potential))
        queries,labels=[],[]
        for i,sample in zip(indices,draws):
            keys=sorted(maps[m][i])
            slots=list(range(m)) if arm=='exhaustive' else sample
            for slot in slots:
                key=keys[slot];queries.append(key);labels.append(maps[m][i][key]);counts[m][i][slot]+=1
        for values in (queries,labels):target_hash.update(struct.pack(f'<{len(values)}q',*values))
        total+=len(labels)
    for m,table in counts.items():assert torch.equal(exposure[m],torch.tensor(table))
    return dict(history_stream_sha256=history.hexdigest(),potential_draws_sha256=draw_hash.hexdigest(),
                actual_queries_labels_sha256=target_hash.hexdigest(),answer_targets=total)


def check_gradients(model,events,queries,labels):
    parameters=tuple(model.parameters())
    native=study.forward_queries(model,events,queries)
    actual=torch.autograd.grad(torch.nn.functional.cross_entropy(native.reshape(-1,6),labels.flatten()),parameters)
    # Every target independently processes its own history through matrix equations.
    repeated=events[:,None].expand(-1,queries.shape[1],-1,-1).reshape(-1,events.shape[1],2)
    flatq=queries.reshape(-1,1)
    ref,_=independent.forward_tangent(model,repeated,flatq)
    logp=ref-ref.logsumexp(-1,keepdim=True)
    expected=torch.autograd.grad(-logp.gather(-1,labels.flatten()[:,None,None]).mean(),parameters)
    return max(independent.close(a,b,True) for a,b in zip(actual,expected))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=study.OUT;report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    assert report['status']=='complete' and report['lock_sha256']==study.replay.digest(out/'lock.json')
    study.replay.check_files(lock['files'])
    assert study.replay.digest(out/'test_data.json')==lock['test_data_sha256']
    assert study.replay.digest(out/'pretest_report.json')==report['pretest_sha256']
    assert study.replay.digest(out/'scores.pt')==report['scores_sha256']
    data=json.loads((study.parent.OUT/'development_data.json').read_text())
    fresh=json.loads((out/'test_data.json').read_text());study.model_source.audit_data(fresh)
    assert all(len(rows)==384 for rows in fresh['pools'].values())
    families={study.model_source.family(r['events']) for rows in data['pools'].values() for r in rows}
    assert all(study.model_source.family(r['events']) not in families for rows in fresh['pools'].values() for r in rows)
    pools={**data['pools'],**fresh['pools']};saved=torch.load(out/'scores.pt',weights_only=True)
    expected={(s,a,t,f'train_{m}') for s in study.SEEDS for a in study.ARMS for t in study.PROBES for m in study.LOADS}
    expected|={(s,a,3000,f'{split}_{m}') for s in study.SEEDS for a in study.ARMS for split in ('validation','test') for m in study.LOADS}
    assert len(saved)==len(report['cells'])==len(expected)==96
    assert {(c['seed'],c['arm'],c['step'],c['pool']) for c in saved}==expected
    checks=[]
    for cell,record in zip(saved,report['cells']):
        assert record=={n:cell[n] for n in ('seed','arm','step','pool','checkpoint','metrics')}
        check=audit_cell(study.ROOT/cell['checkpoint'],pools[cell['pool']],cell)
        checks.append(dict(seed=cell['seed'],arm=cell['arm'],step=cell['step'],pool=cell['pool'],**check))
        if len(checks)%24==0:print(json.dumps(dict(cells_checked=len(checks))),flush=True)
    pretest=json.loads((out/'pretest_report.json').read_text())
    assert all(not c['pool'].startswith('test_') for c in pretest['cells'])
    assert pretest['decision']==report['decision']==study.decision(report['cells'])
    old=json.loads((study.parent.OUT/'report.json').read_text())
    stream_checks=[]
    for item in report['runs']:
        seed,arm=item['seed'],item['arm']
        assert item['updates']==3000 and item['histories']==192000 and item['input_events']==6144000
        assert item['answer_targets']==2784000 and item['parameters']==615
        exposure_path=study.ROOT/item['exposure_path']
        assert study.replay.digest(exposure_path)==item['exposure_sha256']
        exposure=torch.load(exposure_path,weights_only=True)
        stream=audit_stream(seed,arm,{m:data['pools'][f'train_{m}'] for m in study.LOADS},3000,exposure)
        for name,value in stream.items():assert item[name]==value
        historical=next(r for r in old['runs'] if r['seed']==seed)
        assert item['history_stream_sha256']==historical['history_stream_sha256']
        assert item['initial_sha256']==historical['initial_sha256']
        for m,x in exposure.items():
            assert item['exposure'][str(m)]==dict(targets=int(x.sum()),covered=int((x>0).sum()),total=x.numel(),
                                                  min_targets=int(x.min()),max_targets=int(x.max()))
        for snapshot in item['snapshots']:
            cp=study.ROOT/snapshot['checkpoint'];assert study.replay.digest(cp)==snapshot['sha256']
            checkpoint=torch.load(cp,weights_only=True);assert checkpoint['step']==snapshot['step']
            assert all(int(s['step'])==snapshot['step'] for s in checkpoint['optimizer']['state'].values())
            assert snapshot['beta']==float(checkpoint['model']['write_logit'].sigmoid())
            for m,metrics_record in snapshot['training'].items():
                cell=next(c for c in report['cells'] if (c['seed'],c['arm'],c['step'],c['pool'])==(seed,arm,snapshot['step'],f'train_{m}'))
                assert cell['metrics']==metrics_record
        initial=torch.load(study.ROOT/item['snapshots'][0]['checkpoint'],weights_only=True)
        step_check=adam_audit.check_steps(item['steps'],float(initial['model']['write_logit']))
        stream_checks.append(dict(seed=seed,arm=arm,**step_check))
    for seed in study.SEEDS:
        pair=[r for r in report['runs'] if r['seed']==seed]
        for name in ('initial_sha256','history_stream_sha256','potential_draws_sha256','answer_targets'):
            assert pair[0][name]==pair[1][name]
    assert sum(c['targets'] for c in checks)==846336
    study.replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.replay.digest(out/'report.json'),cells=96,targets=846336,
                scalar_adam_steps=12000,max_logit_error=max(c['max_logit_error'] for c in checks),
                protected_files=len(lock['files']),checks=checks,stream_checks=stream_checks)
    study.replay.write(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','stream_checks')}),flush=True)


if __name__=='__main__':main()
