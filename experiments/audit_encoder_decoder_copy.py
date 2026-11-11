"""Independent copy labels, greedy loop, model equations and final decisions."""
import copy
import hashlib
import json

import torch
from torch.nn import functional as F
from encoder_decoder_copy import ROOT, SEEDS, COUNTS, PREFIX, generate, decision
from encoder_decoder_contract import Model, independent
from position_distractors import digest
from recurrent_retrieval import write
from mqa_training import hash_state


def audit_inputs(payload):
    assert payload['generator_seed'] == 2026100203 and payload['pools'].keys() == COUNTS.keys()
    seen, reversed_errors = set(), 0
    for key,rows in payload['pools'].items():
        n = int(key.rsplit('_',1)[1])
        assert len(rows) == COUNTS[key]
        for i,row in enumerate(rows):
            assert len(row) == n and all(3 <= v <= 10 for v in row)
            assert tuple(row) not in seen
            seen.add(tuple(row))
            assert (len(set(row)) == n) == (i < len(rows)//2)
            labels = [*row,2]
            shifted = [1,*row]
            assert all(shifted[t] == (1 if t == 0 else labels[t-1]) for t in range(n+1))
            assert labels[:-1] == row and labels[-1] == 2
            reversed_errors += int(list(reversed(row))+[2] != labels)
    assert reversed_errors > 0
    return dict(passed=True, unique_sequences=len(seen), reversed_label_errors=reversed_errors)


def baselines(payload):
    out = {}
    for key,rows in payload['pools'].items():
        out[key] = {}
        for name,transform in (('oracle',lambda x:list(x)),('repeat_first',lambda x:[x[0]]*len(x)),
                               ('sorted',lambda x:sorted(x)),('reversed',lambda x:list(reversed(x)))):
            correctness = [transform(row)+[2] == row+[2] for row in rows]
            out[key][name] = {g:sum(correctness[i] for i in indices)/len(indices) for g,indices in (
                ('all',list(range(len(rows)))),('distinct',list(range(len(rows)//2))),('repeated',list(range(len(rows)//2,len(rows)))))}
        assert out[key]['oracle']['all'] == 1.
    return out


def controller_checks():
    source = torch.tensor([[3,3,3],[4,4,4],[5,5,5]])
    def script(x,prefix):
        outputs = []
        for row,p in zip(x.tolist(),prefix.tolist()):
            token = 2 if row[0] == 3 else (7 if len(p) == 1 else 2 if p[-1] == 7 else 11) if row[0] == 4 else 8
            scores = [-100.]*16
            scores[token] = 100.
            outputs.append(scores)
        return torch.tensor(outputs)
    expected = [[2],[7,2],[8,8,8,8]]
    actual = generate(script,source,4)
    assert actual == expected
    # Intentionally retain outputs after EOS, violating the stopping contract.
    prefix = torch.ones(3,1,dtype=torch.long)
    wrong = [[] for _ in source]
    for _ in range(4):
        chosen = script(source,prefix).argmax(-1)
        for row,token in zip(wrong,chosen.tolist()):row.append(token)
        prefix = torch.cat((prefix,chosen[:,None]),1)
    assert wrong != expected and wrong[0] == [2,2,2,2]
    return dict(passed=True, expected=expected, actual=actual, continuation_after_eos_detected=True,
                prefix_dependent_continuation_checked=True, absent_eos_budget_checked=True)


def independent_metrics(scores,rows,generated):
    n = len(rows[0])
    labels = torch.tensor([row+[2] for row in rows])
    per_token = F.cross_entropy(scores.flatten(0,1),labels.flatten(),reduction='none').reshape(len(rows),n+1)
    predictions = scores.argmax(-1).tolist()
    out = {}
    for group in ('all','distinct','repeated'):
        indices = [i for i,row in enumerate(rows) if group == 'all' or (len(set(row)) == n) == (group == 'distinct')]
        count = len(indices)
        exact = tokens = content = 0
        stop = dict(early_eos=0,on_time_eos=0,late_eos=0,missing_eos=0)
        total_loss = 0.
        for i in indices:
            expected = rows[i]+[2]
            exact += int(generated[i] == expected)
            tokens += sum(a == b for a,b in zip(predictions[i],expected))
            content += sum(a == b for a,b in zip(predictions[i][:n],rows[i]))
            total_loss += sum(per_token[i].tolist())
            if 2 not in generated[i]:key='missing_eos'
            else:
                pos=generated[i].index(2)+1
                key='early_eos' if pos<n+1 else 'late_eos' if pos>n+1 else 'on_time_eos'
            stop[key] += 1
        out[group] = dict(sequences=count,targets=count*(n+1),exact_accuracy=exact/count,
            token_accuracy=tokens/(count*(n+1)),content_accuracy=content/(count*n),loss=total_loss/(count*(n+1)),
            **{k:v/count for k,v in stop.items()})
    return out


def difference(a,b):
    assert a.keys()==b.keys()
    return max(difference(a[k],b[k]) if isinstance(a[k],dict) else abs(a[k]-b[k]) for k in a)


def independent_decision(runs):
    short = len(runs)==3
    held = True
    long = {'6':True,'8':True}
    for run in runs:
        if run['status']!='trained':
            short=False
            continue
        for pool,groups in run['metrics'].items():
            for group in ('distinct','repeated'):
                a=groups[group]['exact_accuracy']
                if not pool.startswith('test_'):short = short and a>=.95
                elif pool in ('test_3','test_4'):held = held and a>=.95
                else:
                    n=pool.split('_')[-1]
                    long[n] = long[n] and a>=.90
    return dict(short_fit=short,short_held_out=short and held,length_transfer={n:short and value for n,value in long.items()})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp=ROOT/f'results/{PREFIX}.json'
    r=json.loads(rp.read_text())
    assert r['phase']=='complete'
    target=ROOT/f'results/{PREFIX}_audit.json'
    assert not target.exists()
    for name,h in {**r['provenance'],**r['historical']}.items():assert digest(ROOT/name)==h,name
    ip=ROOT/f'results/{PREFIX}_data.json'
    assert digest(ip)==r['input_sha256']
    payload=json.loads(ip.read_text())
    checks=audit_inputs(payload)
    assert controller_checks()==r['controller_checks'] and baselines(payload)==r['baselines']
    pre=json.loads((ROOT/f'results/{PREFIX}_pretest.json').read_text())
    assert len(pre['runs'])==3 and pre['phase']=='all_attempts_finished_before_test'
    assert all(not any(k.startswith('test_') for k in row['metrics']) for row in pre['runs'])
    assert [x['checkpoint_sha256'] for x in pre['runs']]==[x['checkpoint_sha256'] for x in r['runs']]
    cells=[]
    audited_runs=[]
    for run in r['runs']:
        if run['status']!='trained':
            audited_runs.append(run)
            continue
        for kind in ('checkpoint','scores'):assert digest(ROOT/run[kind])==run[kind+'_sha256']
        saved=torch.load(ROOT/run['checkpoint'],weights_only=True)
        details=torch.load(ROOT/run['scores'],weights_only=True)
        torch.manual_seed(run['seed'])
        model=Model()
        assert hash_state(model.state_dict())==run['initial_sha256']
        model.load_state_dict(saved['model'])
        assert saved['step']==run['step']==6000 and all(int(s['step'])==6000 for s in saved['optimizer']['state'].values())
        gen=torch.Generator().manual_seed(run['seed']+1000)
        stream=hashlib.sha256()
        for step in range(1,6001):
            n=3 if step%2 else 4
            half=COUNTS[f'train_{n}']//2
            idx=torch.cat((torch.randint(half,(32,),generator=gen),torch.randint(half,(32,),generator=gen)+half))
            stream.update(json.dumps(idx.tolist()).encode())
        assert stream.hexdigest()==run['stream_sha256']
        other_run={**run,'metrics':{}}
        for pool,rows in payload['pools'].items():
            source=torch.tensor(rows)
            shift=torch.tensor([[1]+row for row in rows])
            reference=[]
            with torch.no_grad():
                for x,y in zip(source.split(64),shift.split(64)):
                    s,t=model.features(x,y)
                    reference.append(independent(model,s,t,torch.zeros_like(x,dtype=torch.bool),torch.zeros_like(y,dtype=torch.bool)))
                reference=torch.cat(reference)
                generated=[]
                # Deliberately separate per-example controller, using independent layer equations.
                for row in rows:
                    x=torch.tensor([row])
                    sequence=[]
                    for step in range(len(row)+3):
                        y=torch.tensor([[1]+sequence])
                        s,t=model.features(x,y)
                        z=independent(model,s,t,torch.zeros_like(x,dtype=torch.bool),torch.zeros_like(y,dtype=torch.bool))
                        token=int(z[0,-1].argmax())
                        sequence.append(token)
                        if token==2:break
                    generated.append(sequence)
            actual=details[pool]
            logit_error=(reference-actual['logits']).abs().max().item()
            assert logit_error<=1e-9
            assert generated==actual['generated'], 'Generated token mismatch'
            actual_metric=independent_metrics(actual['logits'],rows,actual['generated'])
            assert difference(actual_metric,run['metrics'][pool])<=1e-12
            other_metric=independent_metrics(reference,rows,generated)
            metric_error=difference(other_metric,run['metrics'][pool])
            assert metric_error<=1e-9
            other_run['metrics'][pool]=other_metric
            cells.append(dict(seed=run['seed'],pool=pool,sequences=len(rows),teacher_forced_targets=len(rows)*(len(rows[0])+1),
                generated_tokens=sum(map(len,generated)),logit_error=logit_error,metric_error=metric_error,generated_differences=0))
            print(json.dumps(cells[-1]),flush=True)
        audited_runs.append(other_run)
    independent_result=independent_decision(audited_runs)
    assert independent_result==r['decision']==decision(r['runs'])==independent_decision(r['runs'])
    wrong=copy.deepcopy(r['runs'])
    for run in wrong:
        for key,groups in run['metrics'].items():
            if not key.startswith('test_'):
                for group in ('distinct','repeated'):groups[group]['exact_accuracy']=0. if r['decision']['short_fit'] else 1.
    assert independent_decision(wrong)['short_fit']!=r['decision']['short_fit']
    result=dict(passed=True,report_sha256=digest(rp),input_checks=checks,decision=independent_result,
        pools_checked=len(cells),sequences_checked=sum(c['sequences'] for c in cells),
        teacher_forced_targets_checked=sum(c['teacher_forced_targets'] for c in cells),
        generated_tokens_checked=sum(c['generated_tokens'] for c in cells),generated_differences=0,
        max_logit_error=max(c['logit_error'] for c in cells),max_metric_error=max(c['metric_error'] for c in cells),
        intentional_gate_error_detected=True,historical_artifacts_preserved=len(r['historical']),frozen_files_checked=len(r['provenance']),cells=cells)
    write(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','input_checks')}))


if __name__=='__main__':main()
