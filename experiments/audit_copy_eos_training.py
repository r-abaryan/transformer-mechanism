"""Independent equations, oracle decoding, scalar metrics and EOS-study decisions."""
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

from copy_eos_training import ROOT,DIRECTORY,DATA,SEEDS,ARMS,STEPS,decision
from copy_length_exposure import COUNTS,audit_data
from audit_copy_length_exposure import reference,decode as ordinary_decode,scalar_metrics as ordinary_metrics
from mqa_training import hash_state
from position_distractors import digest
from recurrent_retrieval import write


def oracle_decode(model,source):
    outputs=[[] for _ in source]
    for j in range(source.shape[1]):
        prefix=torch.tensor([[1]+row for row in outputs]);z=reference(model,source,prefix)[:,-1].tolist()
        for i,row in enumerate(z):
            outputs[i].append(max([k for k in range(16) if k!=2],key=row.__getitem__))
    return outputs


def scalar_metrics(z,rows,outputs,alpha):
    n=len(rows[0]); result={}
    for group in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if group=='all' or (len(set(r))==n)==(group=='distinct')]
        if not ids:continue
        content_losses=[];terminal_losses=[];generated=correct=conditional=exact=restricted_exact=eos=terminal=0
        for i in ids:
            choices=[]
            for j,values in enumerate(z[i]):
                y=rows[i][j] if j<n else 2;top=max(range(16),key=values.__getitem__)
                m=max(values);ce=m+math.log(math.fsum(math.exp(v-m) for v in values))-values[y]
                if j<n:
                    content_losses.append(ce);correct+=top==y;eos+=top==2
                    other=max([k for k in range(16) if k!=2],key=values.__getitem__)
                    choices.append(other);conditional+=other==y
                    generated+=outputs[i][j]==y
                else:terminal_losses.append(ce);terminal+=top==2
            exact+=outputs[i]==rows[i];restricted_exact+=choices==rows[i]
        count=len(ids)
        result[group]=dict(sequences=count,content_targets=count*n,oracle_content_exact=exact/count,
            oracle_token_accuracy=generated/(count*n),teacher_content_accuracy=correct/(count*n),
            teacher_no_eos_accuracy=conditional/(count*n),teacher_no_eos_exact=restricted_exact/count,
            content_loss=math.fsum(content_losses)/(count*n),terminal_eos_loss=math.fsum(terminal_losses)/count,
            weighted_objective=(math.fsum(content_losses)+alpha*math.fsum(terminal_losses))/(count*(n+1)),
            teacher_content_eos_rate=eos/(count*n),terminal_eos_accuracy=terminal/count)
    return result


def own_decision(runs):
    fitting={};held={};paired=[]
    for arm in ARMS:
        selected=[r for r in runs if r['arm']==arm];assert len(selected)==3
        fitting[arm]=all(r['status']=='trained' and min(r['metrics'][f'train_{n}'][g][key]
            for n in (3,4,5,6) for g in ('distinct','repeated') for key in ('oracle_content_exact','teacher_content_accuracy'))>=.99 for r in selected)
        held[arm]=fitting[arm] and min(r['metrics'][f'test_{n}'][g]['oracle_content_exact']
            for r in selected for n in (3,4,5,6) for g in ('distinct','repeated'))>=.95
    for seed in SEEDS:
        a,b=[next(r for r in runs if r['seed']==seed and r['arm']==arm) for arm in ARMS]
        gains={str(n):b['metrics'][f'test_{n}']['all']['oracle_content_exact']-a['metrics'][f'test_{n}']['all']['oracle_content_exact'] for n in (8,10)}
        smallest=min(b['metrics'][f'test_{n}'][g]['oracle_content_exact']-a['metrics'][f'test_{n}'][g]['oracle_content_exact'] for n in (3,4,5,6) for g in ('distinct','repeated'))
        paired.append(dict(seed=seed,long_gains=gains,seen_length_preserved=smallest>=-.05))
    transfer={}
    for n in (8,10):
        accepted=('distinct','repeated') if n==8 else ('repeated',)
        transfer[str(n)]=held['content_only'] and min(r['metrics'][f'test_{n}'][g]['oracle_content_exact']
            for r in runs if r['arm']=='content_only' for g in accepted)>=.9
    return dict(training_fit=fitting,seen_length_reliability=held,paired=paired,
        removing_eos_benefit=all(fitting.values()) and all(p['seen_length_preserved'] and min(p['long_gains'].values())>=.1 for p in paired),
        content_only_oracle_transfer=transfer)


def main():
    report=json.loads((DIRECTORY/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and not (DIRECTORY/'audit.json').exists()
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    payload=json.loads(DATA.read_text(encoding='utf-8'));assert audit_data(payload)==report['input_checks']
    assert digest(DIRECTORY/'metrics.csv')==report['metrics_sha256']
    pre=json.loads((DIRECTORY/'pretest.json').read_text(encoding='utf-8'))
    assert pre['status']=='all_training_finished_before_test' and len(pre['runs'])==6
    assert all(not any(p.startswith('test_') for p in r['metrics']) for r in pre['runs'])
    assert [r['checkpoint_sha256'] for r in pre['runs']]==[r['checkpoint_sha256'] for r in report['runs']]
    checked=[];stream_pairs={};max_error=max_metric=0.;sequences=content_tokens=teacher_rows=0;ordinary_cells=0
    cells=[]
    for run in report['runs']:
        for kind in ('checkpoint','scores'):assert digest(ROOT/run[kind])==run[kind+'_sha256']
        cp=torch.load(ROOT/run['checkpoint'],weights_only=True);scores=torch.load(ROOT/run['scores'],weights_only=True)
        torch.manual_seed(run['seed']);model=Model();assert hash_state(model.state_dict())==run['initial_sha256']
        model.load_state_dict(cp['model']);assert cp['step']==run['step']==STEPS
        assert all(float(v['step'])==STEPS for v in cp['optimizer']['state'].values())
        pg=cp['optimizer']['param_groups'][0];assert pg['lr']==.001 and pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
        gen=torch.Generator().manual_seed(run['seed']+1000);ranks=hashlib.sha256();stream=hashlib.sha256()
        counts=dict(sequences=0,content_targets=0,terminal_eos_targets=0,computed_target_rows=0,attention_pairs=0)
        for step in range(STEPS):
            n=(3,4,5,6)[step%4];batch=840//(n+1);half=COUNTS[f'train_{n}']//2
            values=torch.rand(2,105,generator=gen,dtype=torch.float64).tolist()
            ranks.update(struct.pack('210d',*(values[0]+values[1])))
            ids=[math.floor(v*half)+g*half for g in range(2) for v in values[g][:batch//2]]
            stream.update(json.dumps([n,ids],separators=(',',':')).encode())
            counts['sequences']+=batch;counts['content_targets']+=batch*n
            counts['terminal_eos_targets']+=batch if run['arm']=='eos' else 0
            counts['computed_target_rows']+=batch*(n+1)
            counts['attention_pairs']+=batch*(n*n+sum(range(1,n+2))+n*(n+1))
        assert counts=={k:run[k] for k in counts}
        assert ranks.hexdigest()==run['rank_stream_sha256'] and stream.hexdigest()==run['sample_stream_sha256']
        assert torch.equal(gen.get_state(),cp['generator']);stream_pairs[run['seed'],run['arm']]=stream.hexdigest()
        other={**run,'metrics':{}}
        with torch.no_grad():
            for pool,ev in scores.items():
                rows=payload['pools'][pool];source=torch.tensor(rows);prefix=torch.tensor([[1]+row for row in rows]);zs=[];outputs=[]
                for x,y in zip(source.split(64),prefix.split(64)):
                    zs.append(reference(model,x,y));outputs.extend(oracle_decode(model,x))
                z=torch.cat(zs);error=float((z-ev['logits']).abs().max());max_error=max(max_error,error)
                assert error<=1e-9 and torch.equal(z.argmax(-1),ev['logits'].argmax(-1)) and outputs==ev['oracle_outputs']
                met=scalar_metrics(z.tolist(),rows,outputs,ARMS[run['arm']])
                for g,m in met.items():
                    assert m.keys()==run['metrics'][pool][g].keys()
                    e=max(abs(v-run['metrics'][pool][g][k]) for k,v in m.items());max_metric=max(max_metric,e);assert e<=1e-9
                other['metrics'][pool]=met;sequences+=len(rows);content_tokens+=len(rows)*len(rows[0]);teacher_rows+=len(rows)*(len(rows[0])+1)
                cells.append(dict(seed=run['seed'],arm=run['arm'],pool=pool,logit_error=error))
                if pool in run['ordinary_stopping']:
                    generated=[]
                    for x in source.split(64):generated.extend(ordinary_decode(model,x))
                    control=ordinary_metrics(z.tolist(),rows,generated)
                    for g,m in run['ordinary_stopping'][pool].items():
                        e=max(abs(v-control[g][k]) for k,v in m.items());max_metric=max(max_metric,e);assert e<=1e-9
                    ordinary_cells+=1
        checked.append(other);print(json.dumps(dict(seed=run['seed'],arm=run['arm'],pools=len(scores),max_logit_error=max_error)),flush=True)
    assert all(stream_pairs[s,'eos']==stream_pairs[s,'content_only'] for s in SEEDS)
    result=own_decision(checked);assert result==own_decision(report['runs'])==decision(report['runs'])==report['decision']
    wrong=copy.deepcopy(report['runs'])
    for r in wrong:
        for n in (3,4,5,6):
            for g in ('distinct','repeated'):
                for k in ('oracle_content_exact','teacher_content_accuracy'):
                    r['metrics'][f'train_{n}'][g][k]=0 if result['training_fit'][r['arm']] else 1
    assert own_decision(wrong)['training_fit']!=result['training_fit']
    assert all(digest(ROOT/p)==h for p,h in report['source_dependencies'].items())
    audit=dict(passed=True,report_sha256=digest(DIRECTORY/'report.json'),audit_source_sha256=digest(Path(__file__)),
        pools_checked=len(cells),oracle_sequences=sequences,oracle_content_tokens=content_tokens,teacher_rows=teacher_rows,
        ordinary_control_cells=ordinary_cells,oracle_sequence_changes=0,teacher_argmax_changes=0,
        max_logit_error=max_error,max_metric_error=max_metric,sampling_and_counts_verified=True,
        criterion_error_detected=True,prior_hashes_unchanged=True,decision=result,cells=cells)
    write(DIRECTORY/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
