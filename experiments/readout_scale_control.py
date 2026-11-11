"""One weight-derived classifier-scale intervention, with final memory fixed."""
import argparse
import copy
import csv
import importlib.util
import json
import math
import unittest

import torch
import feedback_loss_decomposition as previous

training=previous.parent.parent
ROOT=previous.ROOT
OUT=ROOT/'results/readout_scale_control'
SOURCES=('experiments/readout_scale_control.py','experiments/audit_readout_scale_control.py',
         'tests/test_readout_scale_control.py')
PROTOCOL='docs/protocols/readout-scale-control.md'
EPS=1e-12


def head_norm(state):
    a=torch.cat((state['output.weight'],state['output.bias'][:,None]),1)
    value=float((a-a.mean(0,keepdim=True)).norm())
    assert math.isfinite(value) and value>0,'Invalid centered head norm'
    return value


def scaling(old_state,final_state):
    a,b=head_norm(old_state),head_norm(final_state)
    s=a/b
    assert math.isfinite(s) and s>0,'Invalid positive scale'
    return dict(old_norm=a,final_norm=b,scale=s,temperature=1/s)


def intervene(model,scale):
    assert math.isfinite(scale) and scale>0
    result=copy.deepcopy(model)
    with torch.no_grad():
        for p in (result.output.weight,result.output.bias):
            common=p.mean(0,keepdim=True)
            p.copy_(common+scale*(p-common))
    for k,v in model.state_dict().items():
        if not k.startswith('output.'):assert torch.equal(v,result.state_dict()[k]),'Memory changed'
    return result


def summarize(old,final,control):
    for c in (final,control):
        assert torch.equal(old['labels'],c['labels']) and torch.equal(old['repeat'],c['repeat']),'Unaligned labels'
    a,b,c=[previous.measurements(v['logits'],v['labels']) for v in (old,final,control)]
    assert torch.equal(b['prediction'],c['prediction']),'Changed winner'
    masks=(a['correct']&b['correct'],~a['correct']&~b['correct'],~a['correct']&b['correct'],a['correct']&~b['correct'])
    delta=b['ce']-c['ce'];groups={}
    for group in ('all','unique','repeat'):
        scope=torch.ones_like(a['correct']) if group=='all' else old['repeat'].flatten()==(group=='repeat')
        n=int(scope.sum());parts={}
        for name,mask in zip(previous.TRANSITIONS,masks):
            selected=scope&mask;count=int(selected.sum());values={}
            for endpoint,data in (('final',b),('control',c)):
                for field in ('ce','true_probability','winner_confidence'):
                    values[endpoint+'_'+field]=float(data[field][selected].mean()) if count else None
            parts[name]=dict(count=count,**values,ce_removed_contribution=float(delta[selected].sum())/n if n else 0.,
                loss_increased=int((selected&(delta < -EPS)).sum()),loss_decreased=int((selected&(delta>EPS)).sum()))
        old_ce=float(a['ce'][scope].mean()) if n else None
        final_ce=float(b['ce'][scope].mean()) if n else None
        control_ce=float(c['ce'][scope].mean()) if n else None
        removed=final_ce-control_ce if n else None
        groups[group]=dict(count=n,accuracy=float(b['correct'][scope].double().mean()) if n else None,
            old_ce=old_ce,final_ce=final_ce,control_ce=control_ce,ce_removed=removed,
            excess_removed_fraction=removed/(final_ce-old_ce) if n and final_ce-old_ce>EPS else None,
            prediction_changes=0,transitions=parts)
        assert sum(v['count'] for v in parts.values())==n
        if n:assert abs(sum(v['ce_removed_contribution'] for v in parts.values())-removed)<1e-9
    return groups


def preflight():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('test_readout_scale_control',ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,
        sources={n:training.replay.digest(ROOT/n) for n in SOURCES})
    path=ROOT/'results'/('readout_scale_control_preflight.json' if record['passed'] else 'readout_scale_control_preflight_failed.json')
    assert not path.exists();training.replay.write(path,record);assert record['passed']


def run():
    torch.set_num_threads(1);assert not OUT.exists(),'Preserve evidence'
    ready_path=ROOT/'results/readout_scale_control_preflight.json';ready=json.loads(ready_path.read_text())
    assert ready['passed'] and ready['sources']=={n:training.replay.digest(ROOT/n) for n in SOURCES}
    files=previous.protected()
    for name in ('report.json','audit.json','lock.json','metrics.csv'):
        p=previous.OUT/name;files[p.relative_to(ROOT).as_posix()]=training.replay.digest(p)
    old_audit=json.loads((previous.OUT/'audit.json').read_text())
    assert old_audit['passed'] and old_audit['report_sha256']==training.replay.digest(previous.OUT/'report.json')
    for n in (*SOURCES,PROTOCOL,ready_path.relative_to(ROOT).as_posix()):files[n]=training.replay.digest(ROOT/n)
    training.replay.check_files(files);OUT.mkdir()
    training.replay.write(OUT/'lock.json',dict(files=files,seeds=[3461,3463],steps=[3000,12000],
        splits=['train','validation','test'],loads=[2,8,16,32],comparison_epsilon=EPS))
    pairs,_,_=previous.inputs();scales={};models={};checkpoints={}
    for seed in (3461,3463):
        a,b,_=next(v for v in pairs if v[0]['seed']==seed)
        old=torch.load(ROOT/a['checkpoint'],weights_only=True)['model']
        model=training.previous.load_model(ROOT/b['checkpoint'])
        scales[str(seed)]=scaling(old,model.state_dict());control=intervene(model,scales[str(seed)]['scale'])
        path=OUT/f'controlled_{seed}.pt';torch.save(dict(model=control.state_dict(),step=12000),path)
        models[seed]=(model,training.previous.load_model(path));checkpoints[seed]=path.relative_to(ROOT).as_posix()
    training.replay.write(OUT/'scales.json',scales) # Before any controlled score evaluation.
    cells=[];scores=[];max_error=0.
    for a,b,rows in pairs:
        seed=a['seed'];native,control=models[seed]
        events,_,_,_=training.replay.pressure.prepare(rows[:2],'all_queries')
        with torch.no_grad():
            assert torch.equal(native.prefill(events)[0],control.prefill(events)[0]),'Memory state changed'
        cell=training.evaluate(control,rows)
        z=b['logits'];mean=z.mean(-1,keepdim=True)
        error=float((cell['logits']-(mean+scales[str(seed)]['scale']*(z-mean))).abs().max())
        assert error<=1e-9,'Head intervention disagrees with affine logits';max_error=max(max_error,error)
        assert cell['metrics']['complete_maps']==b['metrics']['complete_maps']
        for group in ('all','unique','repeat'):
            assert cell['metrics']['groups'][group]['accuracy']==b['metrics']['groups'][group]['accuracy']
        cell.update(seed=seed,pool=a['pool'],checkpoint=checkpoints[seed]);scores.append(cell)
        cells.append(dict(seed=seed,pool=a['pool'],old_checkpoint=a['checkpoint'],final_checkpoint=b['checkpoint'],
            controlled_checkpoint=checkpoints[seed],groups=summarize(a,b,cell),complete_maps=cell['metrics']['complete_maps']))
    answers=sum(c['groups']['all']['count'] for c in cells)
    assert len(cells)==24 and answers==2*(768+192+384)*sum((2,8,16,32))==155904
    torch.save(scores,OUT/'scores.pt')
    report=dict(status='complete',training_updates=0,cells_count=24,answers=answers,scales=scales,cells=cells,
        device='cpu',dtype='float64',max_affine_logit_error=max_error,
        checkpoint_hashes={str(s):training.replay.digest(ROOT/p) for s,p in checkpoints.items()},
        scores_sha256=training.replay.digest(OUT/'scores.pt'),scales_sha256=training.replay.digest(OUT/'scales.json'),
        lock_sha256=training.replay.digest(OUT/'lock.json'))
    training.replay.check_files(files);training.replay.write(OUT/'report.json',report)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['seed','pool','group','transition',*next(iter(cells[0]['groups']['all']['transitions'].values())).keys()]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for cell in cells:
            for group,g in cell['groups'].items():
                for t,v in g['transitions'].items():writer.writerow(dict(seed=cell['seed'],pool=cell['pool'],group=group,transition=t,**v))
    print(json.dumps({k:v for k,v in report.items() if k!='cells'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    if parser.parse_args().preflight:preflight()
    else:run()
