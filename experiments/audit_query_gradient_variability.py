"""Native finite differences per target and independent scalar variance sums."""
import json
import math

import torch
import query_gradient_variability as study


def reduce_gradients(values,slots):
    histories=len(values);width=len(values[0])
    means=[math.fsum(row)/width for row in values]
    mean=math.fsum(means)/histories
    total=math.fsum((g-mean)**2 for row in values for g in row)/(histories*width)
    between=math.fsum((g-mean)**2 for g in means)/histories
    within=math.fsum((g-center)**2 for row,center in zip(values,means) for g in row)/(histories*width)
    one_se=math.sqrt(total/64);all_se=math.sqrt(between/64)
    assigned=[row[slot] for row,slot in zip(values,slots)]
    assigned_mean=math.fsum(assigned)/histories
    return dict(mean=mean,population_variance=total,history_variance=between,query_variance=within,
                query_variance_fraction=within/total if total else None,
                one_query_batch_se=one_se,all_query_batch_se=all_se,
                all_to_one_se_ratio=all_se/one_se if one_se else None,
                one_query_snr=abs(mean)/one_se if one_se else None,
                all_query_snr=abs(mean)/all_se if all_se else None,
                mean_pressure='lower' if mean>1e-7 else 'higher' if mean< -1e-7 else 'unresolved',
                assigned_mean=assigned_mean,assigned_variance=math.fsum((g-assigned_mean)**2 for g in assigned)/histories,
                targets=histories*width,histories=histories,queries_per_history=width)


def compare_statistics(actual,expected):
    assert actual.keys()==expected.keys()
    for name,value in expected.items():
        if isinstance(value,str) or value is None:assert actual[name]==value
        else:
            assert abs(actual[name]-value)<=1e-11+1e-9*abs(value),f'Variance reduction disagreement: {name}'


def audit_cell(checkpoint,rows,cell,batch=64):
    study.tangent.check_rows(rows)
    events,queries,labels,repeat=study.replay.pressure.prepare(rows,'all_queries')
    assert torch.equal(cell['labels'],labels) and torch.equal(cell['repeat'],repeat)
    slots=torch.tensor([list(query).index(r['query']) for query,r in zip(queries.tolist(),rows)])
    assert torch.equal(cell['assigned_slots'],slots)
    model=study.load_model(checkpoint)
    alpha=model.write_logit.detach().clone()
    before=study.replay.parent.study.parent.state_hash(model.state_dict())
    logit_errors=[];fd_errors=[];native_logits=[]
    with torch.no_grad():
        for first in range(0,len(rows),batch):
            e,q,y=events[first:first+batch],queries[first:first+batch],labels[first:first+batch]
            def forward(offset):
                model.write_logit.copy_(alpha+offset)
                state=model.prefill(e)
                return torch.stack([model.read(state,q[:,j]) for j in range(q.shape[1])],1)
            native=forward(0.)
            error=study.tangent.close(cell['logits'][first:first+batch],native)
            logit_errors.append(error);native_logits.append(native)
            assert torch.equal(native.argmax(-1),cell['logits'][first:first+batch].argmax(-1))
            for h in (1e-4,5e-5):
                plus=forward(h);minus=forward(-h)
                loss=lambda z:z.logsumexp(-1)-z.gather(-1,y[...,None]).squeeze(-1)
                fd=(loss(plus)-loss(minus))/(2*h)
                gradient=cell['gradients'][first:first+batch]
                difference=(fd-gradient).abs()
                assert bool((difference<=1e-8+1e-5*gradient.abs()).all()),'Target finite-difference disagreement'
                fd_errors.append(float(difference.max()))
        model.write_logit.copy_(alpha)
    assert before==study.replay.parent.study.parent.state_hash(model.state_dict())
    scalar=study.tangent.metrics(torch.cat(native_logits),labels,repeat)
    for group,record in scalar.items():
        for name,value in record.items():
            if value is None:assert cell['metrics'][group][name] is None
            else:study.tangent.close(cell['metrics'][group][name],value)
    independent=reduce_gradients(cell['gradients'].tolist(),slots.tolist())
    compare_statistics(cell['statistics'],independent)
    # Autograd of native mean CE, independent of the primary tangent.
    full_mean=0.;assigned_mean=0.
    for first in range(0,len(rows),batch):
        e,q,y=events[first:first+batch],queries[first:first+batch],labels[first:first+batch]
        state=model.prefill(e)
        logits=torch.stack([model.read(state,q[:,j]) for j in range(q.shape[1])],1)
        loss=(logits.logsumexp(-1)-logits.gather(-1,y[...,None]).squeeze(-1))
        full_mean+=float(torch.autograd.grad(loss.sum()/labels.numel(),model.write_logit,retain_graph=True)[0])
        chosen=loss.gather(1,slots[first:first+batch,None])
        assigned_mean+=float(torch.autograd.grad(chosen.sum()/len(rows),model.write_logit)[0])
    study.tangent.close(independent['mean'],full_mean,True)
    study.tangent.close(independent['assigned_mean'],assigned_mean,True)
    return dict(targets=labels.numel(),max_logit_error=max(logit_errors),max_target_fd_error=max(fd_errors),
                full_mean_gradient_error=abs(independent['mean']-full_mean),
                assigned_mean_gradient_error=abs(independent['assigned_mean']-assigned_mean))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=study.OUT;report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    assert report['status']=='complete' and report['lock_sha256']==study.replay.digest(out/'lock.json')
    study.replay.check_files(lock['files'])
    assert study.replay.digest(out/'targets.pt')==report['targets_sha256']
    assert study.replay.digest(out/'metrics.csv')==report['csv_sha256']
    saved=torch.load(out/'targets.pt',weights_only=True)
    data=json.loads((study.replay.parent.OUT/'development_data.json').read_text())
    expected={(s,a,t,m) for s in (3461,3463) for a in ('fixed','resampled') for t in study.replay.PROBES for m in (2,8,16,32)}
    assert len(saved)==len(report['cells'])==len(expected)==112
    assert {(c['seed'],c['arm'],c['step'],c['load']) for c in saved}==expected
    checks=[]
    for cell,record in zip(saved,report['cells']):
        assert record=={n:cell[n] for n in ('seed','arm','step','load','checkpoint','statistics','metrics')}
        result=audit_cell(study.ROOT/cell['checkpoint'],data['pools'][f"train_{cell['load']}"],cell)
        checks.append(dict(seed=cell['seed'],arm=cell['arm'],step=cell['step'],load=cell['load'],**result))
        if len(checks)%28==0:print(json.dumps(dict(cells_checked=len(checks))),flush=True)
    assert sum(c['targets'] for c in checks)==1247232
    study.replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=study.replay.digest(out/'report.json'),cells=112,targets=1247232,
                max_logit_error=max(c['max_logit_error'] for c in checks),
                max_target_fd_error=max(c['max_target_fd_error'] for c in checks),
                max_mean_gradient_error=max(c['full_mean_gradient_error'] for c in checks),
                protected_files=len(lock['files']),checks=checks)
    study.replay.write(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
