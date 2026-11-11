"""Saved-model equations, independent decoding and scalar exposure/decision audit."""
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

from copy_length_exposure import ROOT, DIRECTORY, SEEDS, ARMS, STEPS, TOKENS, COUNTS, audit_data, decision
from encoder_decoder_contract import Model, independent
from position_distractors import digest
from recurrent_retrieval import write
from mqa_training import hash_state


def reference(model,x,prefix):
    s,t=model.features(x,prefix)
    return independent(model,s,t,torch.zeros_like(x,dtype=torch.bool),torch.zeros_like(prefix,dtype=torch.bool))


def decode(model,source):
    result=[[] for _ in source]; live=[True]*len(source); prefix=[[1] for _ in source]
    for _ in range(source.shape[1]+3):
        z=reference(model,source,torch.tensor(prefix))[:,-1].tolist()
        for i,row in enumerate(z):
            token=max(range(16),key=row.__getitem__)
            if live[i]:
                result[i].append(token)
                if token==2:live[i]=False
            prefix[i].append(token)
        if not any(live):break
    return result


def scalar_metrics(z,rows,outputs):
    n=len(rows[0]); result={}
    for group in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r))==n)==(group=='distinct')]
        if not ids:continue
        losses=[];correct=content=exact=content_exact=0; stop=dict(early_eos=0,on_time_eos=0,late_eos=0,missing_eos=0)
        for i in ids:
            labels=rows[i]+[2];exact+=outputs[i]==labels;content_exact+=outputs[i][:n]==rows[i]
            for j,values in enumerate(z[i]):
                top=max(range(16),key=values.__getitem__);correct+=top==labels[j]
                if j<n:content+=top==labels[j]
                m=max(values);losses.append(m+math.log(math.fsum(math.exp(v-m) for v in values))-values[labels[j]])
            first=next((k for k,v in enumerate(outputs[i]) if v==2),None)
            kind='missing_eos' if first is None else 'early_eos' if first<n else 'late_eos' if first>n else 'on_time_eos'
            stop[kind]+=1
        count=len(ids)
        result[group]=dict(sequences=count,targets=count*(n+1),exact_accuracy=exact/count,
            generated_content_exact=content_exact/count,token_accuracy=correct/(count*(n+1)),
            content_accuracy=content/(count*n),loss=math.fsum(losses)/len(losses),**{k:v/count for k,v in stop.items()})
    return result


def own_decision(runs):
    fit={};held={};paired=[]
    for arm,lengths in ARMS.items():
        selected=[r for r in runs if r['arm']==arm]
        fit[arm]=len(selected)==3 and all(r['status']=='trained' and
            min(r['metrics'][f'train_{n}'][g]['exact_accuracy'] for n in lengths for g in ('distinct','repeated'))>=.99 for r in selected)
        held[arm]=fit[arm] and min(r['metrics'][f'test_{n}'][g]['exact_accuracy']
            for r in selected for n in lengths for g in ('distinct','repeated'))>=.95
    for seed in SEEDS:
        a=next(r for r in runs if r['seed']==seed and r['arm']=='short')
        b=next(r for r in runs if r['seed']==seed and r['arm']=='mixed')
        gain=(b['metrics']['test_6']['distinct']['exact_accuracy']+b['metrics']['test_6']['repeated']['exact_accuracy']
              -a['metrics']['test_6']['distinct']['exact_accuracy']-a['metrics']['test_6']['repeated']['exact_accuracy'])/2
        delta=min(b['metrics'][f'test_{n}'][g]['exact_accuracy']-a['metrics'][f'test_{n}'][g]['exact_accuracy']
                  for n in (3,4) for g in ('distinct','repeated'))
        paired.append(dict(seed=seed,six_symbol_gain=gain,short_preserved=delta>=-.05))
    transfer={}
    for n in (8,10):
        accepted=('distinct','repeated') if n==8 else ('repeated',)
        transfer[str(n)]=held['mixed'] and min(r['metrics'][f'test_{n}'][g]['exact_accuracy']
            for r in runs if r['arm']=='mixed' for g in accepted)>=.90
    return dict(training_fit=fit,seen_length_reliability=held,paired=paired,
        broader_support_benefit=all(fit.values()) and all(p['six_symbol_gain']>=.10 and p['short_preserved'] for p in paired),
        mixed_extrapolation=transfer)


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and not (DIRECTORY/'audit.json').exists()
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    payload=json.loads((DIRECTORY/'data.json').read_text(encoding='utf-8'))
    assert digest(DIRECTORY/'data.json')==report['data_sha256'] and audit_data(payload)==report['input_checks']
    assert digest(DIRECTORY/'metrics.csv')==report['metrics_sha256']
    pre=json.loads((DIRECTORY/'pretest.json').read_text(encoding='utf-8'))
    assert pre['status']=='all_training_finished_before_test' and len(pre['runs'])==6
    assert all(not any(k.startswith('test_') for k in r['metrics']) for r in pre['runs'])
    assert [r['checkpoint_sha256'] for r in pre['runs']]==[r['checkpoint_sha256'] for r in report['runs']]
    checked=[]; max_error=max_metric=0.;targets=tokens=sequences=0; streams={}; cells=[]
    for run in report['runs']:
        for kind in ('checkpoint','scores'):assert digest(ROOT/run[kind])==run[kind+'_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);scores=torch.load(ROOT/run['scores'],weights_only=True)
        torch.manual_seed(run['seed']);model=Model();assert hash_state(model.state_dict())==run['initial_sha256']
        model.load_state_dict(cp['model']);assert cp['step']==run['step']==STEPS
        assert all(float(v['step'])==STEPS for v in cp['optimizer']['state'].values())
        pg=cp['optimizer']['param_groups'][0];assert pg['lr']==.001 and pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
        gen=torch.Generator().manual_seed(run['seed']+1000);rank=hashlib.sha256();indices=hashlib.sha256()
        counts=dict(sequences=0,supervised_tokens=0,source_tokens=0,attention_pairs=0)
        for t in range(STEPS):
            n=ARMS[run['arm']][t%len(ARMS[run['arm']])];batch=840//(n+1);half=COUNTS[f'train_{n}']//2
            values=torch.rand(2,105,generator=gen,dtype=torch.float64).tolist()
            rank.update(struct.pack('210d',*(values[0]+values[1])))
            selected=[math.floor(v*half)+group*half for group in range(2) for v in values[group][:batch//2]]
            indices.update(json.dumps([n,selected],separators=(',',':')).encode())
            counts['sequences']+=batch;counts['supervised_tokens']+=batch*(n+1);counts['source_tokens']+=batch*n
            counts['attention_pairs']+=batch*sum((n*n,sum(range(1,n+2)),n*(n+1)))
        assert rank.hexdigest()==run['rank_stream_sha256'] and indices.hexdigest()==run['sample_stream_sha256']
        assert torch.equal(gen.get_state(),cp['generator']) and all(run[k]==v for k,v in counts.items())
        streams[run['seed'],run['arm']]=rank.hexdigest();other={**run,'metrics':{}}
        with torch.no_grad():
            for pool,data in scores.items():
                rows=payload['pools'][pool];source=torch.tensor(rows);prefix=torch.tensor([[1]+r for r in rows])
                z=[];outputs=[]
                for x,y in zip(source.split(64),prefix.split(64)):
                    z.append(reference(model,x,y));outputs.extend(decode(model,x))
                z=torch.cat(z);error=float((z-data['logits']).abs().max());max_error=max(max_error,error)
                assert error<=1e-9 and torch.equal(z.argmax(-1),data['logits'].argmax(-1)) and outputs==data['generated']
                met=scalar_metrics(z.tolist(),rows,outputs)
                for group,m in met.items():
                    assert m.keys()==run['metrics'][pool][group].keys()
                    e=max(abs(v-run['metrics'][pool][group][k]) for k,v in m.items());max_metric=max(max_metric,e);assert e<=1e-9
                other['metrics'][pool]=met;targets+=len(rows)*(len(rows[0])+1);tokens+=sum(map(len,outputs));sequences+=len(rows)
                cells.append(dict(seed=run['seed'],arm=run['arm'],pool=pool,logit_error=error))
        checked.append(other)
        print(json.dumps(dict(seed=run['seed'],arm=run['arm'],pools_checked=len(scores),max_logit_error=max_error)),flush=True)
    assert all(streams[s,'short']==streams[s,'mixed'] for s in SEEDS)
    result=own_decision(checked);assert result==own_decision(report['runs'])==decision(report['runs'])==report['decision']
    wrong=copy.deepcopy(report['runs'])
    for r in wrong:
        for n in ARMS[r['arm']]:
            for g in ('distinct','repeated'):r['metrics'][f'train_{n}'][g]['exact_accuracy']=0 if result['training_fit'][r['arm']] else 1
    assert own_decision(wrong)['training_fit']!=result['training_fit']
    audit=dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),audit_source_sha256=digest(Path(__file__)),
        pools_checked=len(cells),teacher_targets=targets,generated_sequences=sequences,generated_tokens=tokens,
        teacher_argmax_changes=0,generated_sequence_changes=0,max_logit_error=max_error,max_metric_error=max_metric,
        streams_and_generator_states_verified=True,optimizer_counters_verified=True,criteria_error_detected=True,
        prior_hashes_unchanged=True,decision=result,cells=cells)
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    write(DIRECTORY/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
