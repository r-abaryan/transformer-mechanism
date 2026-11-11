"""Loss and confidence changes in immutable paired endpoint predictions."""
import argparse
import csv
import importlib.util
import json
import unittest

import torch
import query_feedback_continuation as parent

ROOT=parent.ROOT
OUT=ROOT/'results/feedback_loss_decomposition'
TRANSITIONS=('correct_both','wrong_both','rescued','newly_wrong')
FIELDS=('old_ce','new_ce','old_true_probability','new_true_probability','old_winner_confidence',
        'new_winner_confidence','old_true_margin','new_true_margin')
SOURCES=('experiments/feedback_loss_decomposition.py','experiments/audit_feedback_loss_decomposition.py',
         'tests/test_feedback_loss_decomposition.py')
PROTOCOL='docs/protocols/feedback-loss-decomposition.md'
EPS=1e-12


def measurements(logits,labels):
    z=logits.reshape(-1,6);y=labels.flatten()
    assert torch.isfinite(z).all() and ((y>=0)&(y<6)).all()
    logp=z.log_softmax(-1);pred=z.argmax(-1)
    others=z.clone();others.scatter_(-1,y[:,None],-float('inf'))
    return dict(ce=-logp.gather(-1,y[:,None]).flatten(),true_probability=logp.gather(-1,y[:,None]).flatten().exp(),
                winner_confidence=logp.max(-1).values.exp(),true_margin=z.gather(-1,y[:,None]).flatten()-others.max(-1).values,
                correct=pred==y,prediction=pred)


def decompose(old,new):
    assert old['logits'].shape==new['logits'].shape and old['logits'].shape[-1]==6
    assert torch.equal(old['labels'],new['labels']) and torch.equal(old['repeat'],new['repeat']),'Unaligned targets'
    assert old['logits'].shape[:-1]==old['labels'].shape==old['repeat'].shape
    a=measurements(old['logits'],old['labels']);b=measurements(new['logits'],new['labels'])
    masks=(a['correct']&b['correct'],~a['correct']&~b['correct'],~a['correct']&b['correct'],a['correct']&~b['correct'])
    delta=b['ce']-a['ce'];groups={}
    for group in ('all','unique','repeat'):
        scope=torch.ones_like(a['correct']) if group=='all' else old['repeat'].flatten()==(group=='repeat')
        n=int(scope.sum());parts={}
        for name,mask in zip(TRANSITIONS,masks):
            selected=mask&scope;count=int(selected.sum());stats={}
            for endpoint,values in [('old',a),('new',b)]:
                for field in ('ce','true_probability','winner_confidence','true_margin'):
                    stats[endpoint+'_'+field]=float(values[field][selected].mean()) if count else None
            parts[name]=dict(count=count,fraction=count/n if n else None,**stats,
                mean_ce_change=float(delta[selected].mean()) if count else None,
                ce_contribution=float(delta[selected].sum())/n if n else 0.,
                loss_increased=int((selected&(delta>EPS)).sum()),
                winner_confidence_increased=int((selected&(b['winner_confidence']>a['winner_confidence']+EPS)).sum()),
                prediction_changed=int((selected&(a['prediction']!=b['prediction'])).sum()),
                old_confidence_ge_90=int((selected&(a['winner_confidence']>=.9)).sum()),
                new_confidence_ge_90=int((selected&(b['winner_confidence']>=.9)).sum()))
        groups[group]=dict(count=n,old_accuracy=float(a['correct'][scope].double().mean()) if n else None,
            new_accuracy=float(b['correct'][scope].double().mean()) if n else None,
            old_ce=float(a['ce'][scope].mean()) if n else None,new_ce=float(b['ce'][scope].mean()) if n else None,
            ce_change=float(delta[scope].mean()) if n else None,transitions=parts)
        assert sum(v['count'] for v in parts.values())==n
        if n:assert abs(sum(v['ce_contribution'] for v in parts.values())-groups[group]['ce_change'])<=1e-9
    return groups


def examples(old,new,rows,limit=5):
    a=measurements(old['logits'],old['labels']);b=measurements(new['logits'],new['labels'])
    delta=b['ce']-a['ce'];width=old['labels'].shape[1];records=[]
    ordered=sorted((i for i in range(len(delta)) if float(delta[i])>EPS),
                   key=lambda i:(-round(float(delta[i]),10),i))
    for index in ordered[:limit]:
        history,slot=divmod(index,width);key=sorted(dict(rows[history]['events']))[slot]
        old_ok,new_ok=bool(a['correct'][index]),bool(b['correct'][index])
        name=('correct_both' if old_ok else 'rescued') if new_ok else ('newly_wrong' if old_ok else 'wrong_both')
        records.append(dict(flat_index=index,history_index=history,key=key,label=int(old['labels'][history,slot]),
            old_prediction=int(a['prediction'][index]),new_prediction=int(b['prediction'][index]),transition=name,
            old_ce=float(a['ce'][index]),new_ce=float(b['ce'][index]),ce_change=float(delta[index])))
    return records


def inputs():
    original=parent.parent;old_report=json.loads((original.OUT/'report.json').read_text())
    new_report=json.loads((parent.OUT/'report.json').read_text())
    old_scores=torch.load(original.OUT/'scores.pt',weights_only=True)
    new_scores=torch.load(parent.OUT/'scores.pt',weights_only=True)
    payload=json.loads((original.parent.OUT/'development_data.json').read_text())
    test=json.loads((original.OUT/'test_data.json').read_text())
    pools={**payload['pools'],**test['pools']}
    pairs=[]
    for seed in (3461,3463):
        run=next(r for r in new_report['runs'] if r['seed']==seed)
        assert run['stop']==12000
        for split in ('train','validation','test'):
            for load in (2,8,16,32):
                pool=f'{split}_{load}'
                a=next(c for c in old_scores if (c['seed'],c['arm'],c['step'],c['pool'])==(seed,'exhaustive',3000,pool))
                b=next(c for c in new_scores if (c['seed'],c['arm'],c['step'],c['pool'])==(seed,'exhaustive',12000,pool))
                pairs.append((a,b,pools[pool]))
    return pairs,old_report,new_report


def protected():
    files=dict(json.loads((parent.OUT/'lock.json').read_text())['files'])
    for folder in (parent.parent.OUT,parent.OUT):
        report=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'audit.json').read_text())
        assert audit['passed'] and audit['report_sha256']==parent.parent.replay.digest(folder/'report.json')
        assert report['scores_sha256']==parent.parent.replay.digest(folder/'scores.pt')
        for name in ('report.json','scores.pt','audit.json','lock.json','metrics.csv'):
            p=folder/name;files[p.relative_to(ROOT).as_posix()]=parent.parent.replay.digest(p)
    return files


def preflight():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('test_feedback_loss_decomposition',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:parent.parent.replay.digest(ROOT/n) for n in SOURCES})
    name='feedback_loss_decomposition_preflight.json' if record['passed'] else 'feedback_loss_decomposition_preflight_failed.json'
    path=ROOT/'results'/name;assert not path.exists();parent.parent.replay.write(path,record);assert record['passed']


def run():
    torch.set_num_threads(1);assert not OUT.exists(),'Preserve executed evidence'
    ready_path=ROOT/'results/feedback_loss_decomposition_preflight.json';ready=json.loads(ready_path.read_text())
    hashes={n:parent.parent.replay.digest(ROOT/n) for n in SOURCES};assert ready['passed'] and ready['sources']==hashes
    files=protected()
    for name in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[name]=parent.parent.replay.digest(ROOT/name)
    parent.parent.replay.check_files(files);OUT.mkdir();parent.parent.replay.write(OUT/'lock.json',dict(files=files,
        seeds=[3461,3463],steps=[3000,12000],splits=['train','validation','test'],loads=[2,8,16,32],
        transitions=list(TRANSITIONS),comparison_epsilon=EPS,confidence_threshold=.9,top_positive_examples=5))
    pairs,old_report,new_report=inputs();cells=[]
    for a,b,rows in pairs:
        cell=dict(seed=a['seed'],pool=a['pool'],old_step=3000,new_step=12000,old_checkpoint=a['checkpoint'],
            new_checkpoint=b['checkpoint'],groups=decompose(a,b),top_loss_increases=examples(a,b,rows))
        for scope,record in cell['groups'].items():
            for scores in (a,b):
                endpoint='old' if scores is a else 'new'
                metric=scores['metrics']['groups'][scope]
                assert record['count']==metric['count']
                for name in ('ce','accuracy'):
                    value=record[endpoint+'_'+name]
                    assert value is None if metric[name] is None else abs(value-metric[name])<=1e-9
        cells.append(cell)
    paired=sum(c['groups']['all']['count'] for c in cells)
    assert len(cells)==24 and paired==2*(768+192+384)*sum((2,8,16,32))==155904
    report=dict(status='complete',training_updates=0,device='cpu',dtype='float64',cells=cells,cells_count=len(cells),
        paired_answers=paired,endpoint_vectors=2*paired,opened_development_data=True,
        lock_sha256=parent.parent.replay.digest(OUT/'lock.json'))
    parent.parent.replay.check_files(files);parent.parent.replay.write(OUT/'report.json',report)
    fields=['seed','pool','group','transition','count','fraction',*FIELDS,'mean_ce_change','ce_contribution',
        'loss_increased','winner_confidence_increased','prediction_changed','old_confidence_ge_90','new_confidence_ge_90']
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cells:
            for group,g in c['groups'].items():
                for name,values in g['transitions'].items():writer.writerow(dict(seed=c['seed'],pool=c['pool'],group=group,transition=name,**values))
    print(json.dumps({k:v for k,v in report.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
