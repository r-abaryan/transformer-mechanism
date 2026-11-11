"""Independent matrix recurrence, state tangent and central differences."""
import json
import math

import torch
import objective_beta_pressure as study


def check_rows(rows):
    assert rows
    for row in rows:
        memory = {}
        for key,value in row['events']:
            assert key not in memory or memory[key] == value
            memory[key] = value
        assert len(memory) == row['load'] and memory[row['query']] == row['label'], 'Wrong independent label'
        repeats = sum(key == row['query'] for key,value in row['events']) > 1
        assert row['group'] == ('repeat' if repeats else 'unique')


def inputs(rows, objective):
    check_rows(rows)
    tables = [dict(row['events']) for row in rows]
    qs = [[row['query']] if objective == 'assigned' else sorted(table) for row,table in zip(rows,tables)]
    ys = [[table[q] for q in query] for table,query in zip(tables,qs)]
    e = torch.tensor([row['events'] for row in rows]); q = torch.tensor(qs); y = torch.tensor(ys)
    repeats = torch.tensor([[sum(k == query for k,v in row['events']) > 1 for query in queries]
                           for row,queries in zip(rows,qs)])
    return e,q,y,repeats


def forward_tangent(model, events, queries, alpha_offset=0.):
    raw = model.key.weight[events[...,0]]
    keys = raw/raw.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
    raw_q = model.key.weight[queries]
    query = raw_q/raw_q.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
    values = model.value.weight[events[...,1]]
    beta = (model.write_logit+alpha_offset).sigmoid()
    s = values.new_zeros(len(events),16,8); d = torch.zeros_like(s)
    identity = torch.eye(16,dtype=values.dtype)
    for t in range(events.shape[1]):
        key = keys[:,t]; v = values[:,t]
        transition = identity-beta*key[:,:,None]*key[:,None,:]
        correction = key[:,:,None]*(v-(key[:,None,:]@s).squeeze(1))[:,None,:]
        d = transition@d+correction
        s = transition@s+beta*key[:,:,None]*v[:,None,:]
    reads = torch.einsum('bmk,bkv->bmv',query,s)
    derivative = torch.einsum('bmk,bkv->bmv',query,d)@model.output.weight.T
    return reads@model.output.weight.T+model.output.bias, derivative


def metrics(logits, labels, repeat):
    values = logits.detach().reshape(-1,6).tolist(); ys = labels.flatten().tolist(); rs = repeat.flatten().tolist()
    losses, correct = [], []
    for score,y in zip(values,ys):
        high = max(score)
        losses.append(high+math.log(math.fsum(math.exp(z-high) for z in score))-score[y])
        correct.append(max(range(6),key=lambda j:score[j]) == y)
    answer = {}
    for group in ('all','unique','repeat'):
        indices = [i for i,r in enumerate(rs) if group == 'all' or r == (group == 'repeat')]
        n = len(indices)
        answer[group] = dict(count=n,ce=math.fsum(losses[i] for i in indices)/n if n else None,
                            accuracy=sum(correct[i] for i in indices)/n if n else None)
    return answer


def close(actual, reference, gradient=False):
    a,b = torch.as_tensor(actual,dtype=torch.float64),torch.as_tensor(reference,dtype=torch.float64)
    tolerance = 1e-9+1e-7*b.abs() if gradient else 1e-9
    assert bool(((a-b).abs() <= tolerance).all()), 'Independent numerical disagreement'
    return float((a-b).abs().max())


def audit_checkpoint(checkpoint, rows, objective, cell, batch=64):
    model = study.model_source.LoadReader('delta').eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    events, queries, labels, repeat = inputs(rows, objective)
    assert torch.equal(cell['labels'],labels) and torch.equal(cell['repeat'],repeat)
    parameters = dict(model.named_parameters())
    gradients = {n:torch.zeros_like(p) for n,p in parameters.items()}
    logits_list = []; analytic_alpha = 0.
    shifted_losses = {h:{sign:0. for sign in (-1,1)} for h in (1e-4,5e-5)}
    for first in range(0,len(rows),batch):
        e,q,y = events[first:first+batch],queries[first:first+batch],labels[first:first+batch]
        logits,tangent = forward_tangent(model,e,q)
        log_prob = logits-logits.logsumexp(-1,keepdim=True)
        loss = -log_prob.gather(-1,y[...,None]).sum()/labels.numel()
        gs = torch.autograd.grad(loss,tuple(parameters.values()))
        for (name,_),g in zip(parameters.items(),gs):gradients[name] += g.detach()
        with torch.no_grad():
            probabilities = log_prob.exp(); probabilities.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=torch.float64))
            beta = model.write_logit.sigmoid()
            analytic_alpha += float((probabilities*tangent).sum()/labels.numel()*beta*(1-beta))
            for h,values in shifted_losses.items():
                for sign in values:
                    shifted,_ = forward_tangent(model,e,q,sign*h)
                    values[sign] += float((shifted.logsumexp(-1)-shifted.gather(-1,y[...,None]).squeeze(-1)).sum()/labels.numel())
        logits_list.append(logits.detach())
    logits = torch.cat(logits_list)
    logit_error = close(cell['logits'],logits)
    assert torch.equal(cell['logits'].argmax(-1),logits.argmax(-1)), 'Argmax disagreement'
    scalar = metrics(logits,labels,repeat)
    assert scalar.keys() == cell['metrics'].keys()
    for group,row in scalar.items():
        for key,value in row.items():
            if value is None:assert cell['metrics'][group][key] is None
            else:close(cell['metrics'][group][key],value)
    errors = [close(cell['gradients'][n],g,True) for n,g in gradients.items()]
    alpha = float(cell['gradients']['write_logit'])
    close(alpha,analytic_alpha,True)
    fd = {str(h):(values[1]-values[-1])/(2*h) for h,values in shifted_losses.items()}
    for value in fd.values():assert abs(alpha-value) <= 1e-8+1e-5*abs(alpha), 'Beta finite-difference disagreement'
    return dict(max_logit_error=logit_error,max_parameter_gradient_error=max(errors),
                analytic_alpha_gradient=analytic_alpha,finite_differences=fd,
                max_fd_error=max(abs(alpha-v) for v in fd.values()),predictions=labels.numel())


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=study.OUT; digest=study.parent.study.parent.digest; write=study.parent.study.parent.write
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    assert report['status']=='complete' and digest(out/'lock.json')==report['lock_sha256']
    for name,sha in lock['files'].items():assert digest(study.ROOT/name)==sha
    assert digest(out/'scores.pt')==report['scores_sha256']
    saved=torch.load(out/'scores.pt',weights_only=True)
    parent=json.loads((study.parent.OUT/'report.json').read_text())
    data=json.loads((study.parent.OUT/'development_data.json').read_text())
    assert len(saved)==len(report['cells'])==32
    expected={(s,a,m,o) for s in (3461,3463) for a in ('fixed','resampled') for m in study.LOADS for o in study.OBJECTIVES}
    assert {(c['seed'],c['arm'],c['load'],c['objective']) for c in saved}==expected
    checks=[]
    for cell in saved:
        run=next(r for r in parent['runs'] if (r['seed'],r['arm'])==(cell['seed'],cell['arm']))
        checked=audit_checkpoint(study.ROOT/run['checkpoint'],data['pools'][f"train_{cell['load']}"],cell['objective'],cell)
        checks.append(dict(seed=cell['seed'],arm=cell['arm'],load=cell['load'],objective=cell['objective'],**checked))
        if len(checks)%8==0:print(json.dumps(dict(cells_checked=len(checks))),flush=True)
    # Independently reduce pooled vectors and all reported summary fields.
    for pooled in report['pooled']:
        pair={}
        for objective in study.OBJECTIVES:
            cells=[c for c in saved if (c['seed'],c['arm'],c['objective'])==(pooled['seed'],pooled['arm'],objective)]
            grad={n:torch.stack([c['gradients'][n] for c in cells]).mean(0) for n in cells[0]['gradients']}
            record=pooled['objectives'][objective];alpha=float(grad['write_logit'])
            close(record['alpha_gradient'],alpha,True)
            close(record['ce'],math.fsum(c['metrics']['all']['ce'] for c in cells)/4)
            close(record['beta_gradient'],alpha/(pooled['beta']*(1-pooled['beta'])),True)
            assert record['pressure']==('lower' if alpha>1e-7 else 'higher' if alpha< -1e-7 else 'unresolved')
            pair[objective]=grad
        actual_comparison=study.compare(*pair.values())
        for key in ('assigned_pressure','all_query_pressure','opposite_resolved_signs'):
            assert pooled['comparison'][key]==actual_comparison[key]
        for name,value in actual_comparison['gradient_cosines'].items():
            if value is None:assert pooled['comparison']['gradient_cosines'][name] is None
            else:close(pooled['comparison']['gradient_cosines'][name],value)
    for record,cell in zip(report['cells'],saved):assert record==dict(seed=cell['seed'],arm=cell['arm'],load=cell['load'],objective=cell['objective'],**study.describe(cell,float(next(p for p in report['pooled'] if (p['seed'],p['arm'])==(cell['seed'],cell['arm']))['beta'])))
    assert len(report['comparisons'])==16 and len(report['pooled'])==4
    for record in report['comparisons']:
        cells={c['objective']:c for c in saved if (c['seed'],c['arm'],c['load'])==(record['seed'],record['arm'],record['load'])}
        comparison=study.compare(*(cells[o]['gradients'] for o in study.OBJECTIVES))
        assert record==dict(seed=record['seed'],arm=record['arm'],load=record['load'],**comparison)
    for name,sha in lock['files'].items():assert digest(study.ROOT/name)==sha
    result=dict(passed=True,report_sha256=digest(out/'report.json'),cells=32,predictions=sum(c['predictions'] for c in checks),
                max_logit_error=max(c['max_logit_error'] for c in checks),max_parameter_gradient_error=max(c['max_parameter_gradient_error'] for c in checks),
                max_fd_error=max(c['max_fd_error'] for c in checks),protected_files=len(lock['files']),checks=checks)
    assert result['predictions']==190464
    write(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__ == '__main__':main()
