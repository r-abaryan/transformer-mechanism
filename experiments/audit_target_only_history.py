"""Independent query enumeration, closed-form clean reads and matrix recurrence."""
import json
import math

import torch
import target_only_history as study
import audit_query_feedback_training as reference
import audit_feedback_loss_decomposition as prior


def closed_logits(model, key_id, value_id, n):
    # Use actual squared norm: eps-clamped normalization need not produce a unit key.
    raw = model.key.weight[key_id].detach().tolist()
    den = max(math.sqrt(math.fsum(x*x for x in raw)), 1e-12)
    key = [x/den for x in raw]; h = math.fsum(x*x for x in key)
    beta = 1/(1+math.exp(-float(model.write_logit.detach())))
    factor = 1-(1-beta*h)**n
    read = [factor*v for v in model.value.weight[value_id].detach().tolist()]
    return [math.fsum(w*x for w,x in zip(row,read))+bias
            for row,bias in zip(model.output.weight.detach().tolist(),model.output.bias.detach().tolist())]


def audit_cell(path, rows, source, saved, record):
    _, queries, labels, repeat = reference.independent.inputs(rows,'all_queries')
    assert torch.equal(saved['labels'],labels) and torch.equal(saved['repeat'],repeat), 'Labels/order disagree'
    model = study.training.previous.load_model(path)
    state = {k:v.clone() for k,v in model.state_dict().items()}; rng = torch.get_rng_state().clone()
    expected = torch.empty(*labels.shape,6,dtype=torch.float64); counts = torch.zeros_like(labels)
    cases = {}
    for i, row in enumerate(rows):
        for j, q in enumerate(queries[i].tolist()):
            positions = [t for t,(key,value) in enumerate(row['events']) if key == q]
            values = [row['events'][t][1] for t in positions]
            assert values and all(v == int(labels[i,j]) for v in values), 'Unstable label'
            n = len(positions); counts[i,j] = n; case = (q,values[0],n)
            if case not in cases: cases[case] = closed_logits(model,*case)
            expected[i,j] = torch.tensor(cases[case],dtype=torch.float64)
    assert torch.equal(counts,saved['target_writes']), 'Wrong target-write count'
    error = reference.independent.close(saved['logits'],expected)
    assert torch.equal(saved['logits'].argmax(-1),expected.argmax(-1)), 'Winner disagrees'
    # Every distinct retained sequence is also checked with independent matrix transitions.
    matrix_error = 0.
    with torch.no_grad():
        for n in sorted({c[2] for c in cases}):
            triples = [c for c in cases if c[2]==n]
            e = torch.tensor([[[q,v]]*n for q,v,_ in triples]); q = torch.tensor([[c[0]] for c in triples])
            z,_ = reference.independent.forward_tangent(model,e,q)
            expected_cases = torch.tensor([cases[c] for c in triples],dtype=torch.float64)
            matrix_error = max(matrix_error,reference.independent.close(z[:,0],expected_cases))
    metrics = reference.metrics(expected,labels,repeat)
    reference.check_metrics(saved['metrics'],metrics)
    reference.check_metrics(record['filtered'],metrics)
    reference.check_metrics(record['original'],reference.metrics(source['logits'],labels,repeat))
    # Discrete transitions are reconstructed without the native transition reducer.
    a = source['logits'].argmax(-1).tolist(); b = expected.argmax(-1).tolist(); y=labels.tolist(); rs=repeat.tolist()
    transitions = {}
    for name in ('all','unique','repeat'):
        t = dict(count=0,correct_both=0,wrong_both=0,rescued=0,newly_wrong=0)
        for i in range(len(y)):
            for j in range(len(y[i])):
                if name != 'all' and rs[i][j] != (name=='repeat'): continue
                ok_a = a[i][j]==y[i][j]; ok_b = b[i][j]==y[i][j]
                label = ('correct_both' if ok_a else 'rescued') if ok_b else ('newly_wrong' if ok_a else 'wrong_both')
                t['count'] += 1; t[label] += 1
        transitions[name] = t
    assert record['transitions']==transitions, 'Transitions disagree'
    assert record['retained_events']==int(counts.sum()) and record['target_writes']==int(counts[0,0])
    assert abs(record['beta']-float(model.write_logit.sigmoid().detach()))<=1e-12
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()) and torch.equal(rng,torch.get_rng_state())
    return dict(max_closed_logit_error=error,max_matrix_logit_error=matrix_error,
                distinct_sequences=len(cases),answers=labels.numel())


def main():
    torch.set_num_threads(1); out=study.OUT; replay=study.training.replay
    report=json.loads((out/'report.json').read_text()); lock=json.loads((out/'lock.json').read_text())
    replay.check_files(lock['files']); assert report['lock_sha256']==replay.digest(out/'lock.json')
    assert report['scores_sha256']==replay.digest(out/'scores.pt')
    assert report['status']=='complete' and report['training_updates']==report['optimizer_calls']==0
    assert report['device']=='cpu' and report['dtype']=='float64' and report['original_scores_reused']
    scores=torch.load(out/'scores.pt',weights_only=True); pairs=study.inputs()
    expected_ids=[(s,t,f'{split}_{m}') for s in (3461,3463) for t in (3000,12000)
                  for split in ('train','validation','test') for m in (2,8,16,32)]
    errors=[]
    for records in (scores,report['cells']):
        assert [(r['seed'],r['step'],r['pool']) for r in records]==expected_ids, 'Missing or reordered cell'
    for (source,rows), saved, record in zip(pairs,scores,report['cells']):
        for k in ('seed','step','pool','checkpoint'): assert saved[k]==record[k]==source[k]
        assert len(rows)=={'train':768,'validation':192,'test':384}[source['pool'].split('_')[0]]
        assert all(len(r['events'])==32 for r in rows)
        errors.append(audit_cell(study.ROOT/source['checkpoint'],rows,source,saved,record))
    assert len(errors)==report['cells_count']==48 and sum(e['answers'] for e in errors)==report['evaluated_answers']==311808
    assert sum(c['retained_events'] for c in report['cells'])==report['retained_events']==688128
    replay.check_files(lock['files'])
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),cells=48,evaluated_answers=311808,
        max_closed_logit_error=max(e['max_closed_logit_error'] for e in errors),
        max_matrix_logit_error=max(e['max_matrix_logit_error'] for e in errors),
        distinct_sequence_checks=sum(e['distinct_sequences'] for e in errors),protected_files=len(lock['files']),training_updates=0)
    path=out/'audit.json'; assert not path.exists(); replay.write(path,result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
