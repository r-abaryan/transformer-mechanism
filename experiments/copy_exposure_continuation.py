"""Shared-checkpoint exposure continuation; reuse frozen native training code."""
import csv
import json
import sys
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

from copy_position_sites import ROOT,make_model,forward,inputs,evaluate,digest,write
from copy_eos_development import continuation
from copy_learning_trajectory import equal
from audit_copy_position_sites import audit_cell

OUT=ROOT/'results/copy_exposure_continuation'
START=ROOT/'results/copy_learning_trajectory/3767_cross_only_6000.pt'
STAGES=(6500,7000,7500,8000)
ARMS=('cross_only','both')


def pools():
    data=json.loads((ROOT/'results/copy_position_sites/data.json').read_text())
    return dict(pools={p:r for p,r in data['pools'].items() if not p.startswith('uniform_')})


def peak_working_set():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
          ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
           'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters();c.cb=ctypes.sizeof(c)
    get_process=ctypes.windll.kernel32.GetCurrentProcess;get_process.restype=wintypes.HANDLE
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    assert fn(get_process(),ctypes.byref(c),c.cb)
    return c.PeakWorkingSetSize


@torch.no_grad()
def evaluate_checkpoint(path,data,directory):
    model=make_model('rope').eval();model.load_state_dict(torch.load(path,weights_only=True)['model'])
    ms={};saved={};maximum=0.
    for p,rows in data['pools'].items():
        m,s=evaluate(model,rows,'cross_only')
        err,_=audit_cell(model,rows,None,'cross_only',s,m);maximum=max(maximum,err)
        ms[p]=m;saved[p]=s
    dest=directory/(path.stem+'_scores.pt');assert not dest.exists();torch.save(saved,dest)
    return dict(metrics=ms,scores=str(dest.relative_to(ROOT)),scores_sha256=digest(dest),max_reference_error=maximum)


def gate(ms):
    short=all(v[g]['exact_accuracy']>=.99 for p,v in ms.items() if int(p.rsplit('_',1)[1])<=6 for g in ('distinct','repeated'))
    long=all(ms[p][g]['exact_accuracy']>=.95 and ms[p][g]['on_time_eos']>=.95
             for p,g in (('test_8','distinct'),('test_8','repeated'),('test_10','repeated')))
    return dict(short_preserved=short,long_recovered=long,repair=short and long)


def preflight(data):
    directory=ROOT/'results/test_runs'/('exposure-continuation-'+uuid.uuid4().hex);directory.mkdir(parents=True)
    torch.manual_seed(117);model=make_model('rope');opt=torch.optim.Adam(model.parameters(),lr=.001)
    path=directory/'initialized.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0,
        generator=torch.Generator().manual_seed(1117).get_state(),position_generator=torch.Generator().manual_seed(2117).get_state()),path)
    fixture=dict(pools={'test_3':[[3,4,5],[3,3,4]],'test_10':[[3]*10]})
    evaluation=evaluate_checkpoint(path,fixture,directory)
    saved=torch.load(ROOT/evaluation['scores'],weights_only=True)
    assert evaluation['metrics']['test_10']['distinct'] is None
    bad=json.loads(json.dumps(evaluation['metrics']['test_3']));bad.pop('distinct')
    try:audit_cell(model,fixture['pools']['test_3'],None,'cross_only',saved['test_3'],bad)
    except (AssertionError,KeyError):pass
    else:raise AssertionError('Missing group accepted')
    saved['test_3']['generated'][0][0]=(saved['test_3']['generated'][0][0]+1)%16
    try:audit_cell(model,fixture['pools']['test_3'],None,'cross_only',saved['test_3'],evaluation['metrics']['test_3'])
    except AssertionError:pass
    else:raise AssertionError('Corrupted generation accepted')
    x=torch.tensor(fixture['pools']['test_3']);prefix,y=inputs(x);pos=torch.tensor([0,3,7,10]);outputs={}
    for arm in ARMS:
        grads=[]
        for reference in (False,True):
            z=forward(model,x,prefix,pos if arm=='both' else None,pos,reference)[0]
            grads.append(torch.autograd.grad(F.cross_entropy(z.flatten(0,1),y.flatten()),tuple(model.parameters())))
            if not reference:outputs[arm]=z.detach()
            else:torch.testing.assert_close(z,outputs[arm],atol=1e-10,rtol=0)
        for a,b in zip(*grads):torch.testing.assert_close(a,b,atol=1e-10,rtol=0)
    assert (outputs['both']-outputs['cross_only']).abs().max()>1e-5,'Missing self exposure undetected'
    from copy_cross_output_transfer import preflight as evaluator_fixtures
    supplied_checks=evaluator_fixtures()
    profiles=[]
    for arm in ARMS:
        dest=directory/f'{arm}_profile.pt'
        p=continuation(path,arm,data,'native',1,dest)
        p['checkpoint_sha256']=digest(dest);p['peak_process_working_set_bytes']=peak_working_set()
        p['evaluation']=evaluate_checkpoint(dest,fixture,directory);profiles.append(p)
    a,b=[torch.load(ROOT/p['checkpoint'],weights_only=True) for p in profiles]
    assert torch.equal(a['generator'],b['generator']) and torch.equal(a['position_generator'],b['position_generator'])
    assert not equal(a['model'],b['model'])
    return dict(passed=True,initialized_saved_evaluator=True,missing_group_rejected=True,corrupted_generation_rejected=True,
        reference_parameter_gradients=True,missing_self_exposure_detected=True,supplied_evaluator_checks=supplied_checks,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),complete_step_profiles=profiles)


def main():
    assert not OUT.exists(),'Preserve evidence'
    data=pools();pre=preflight(data)
    replay=json.loads((ROOT/'results/copy_learning_trajectory/persisted-audit.json').read_text())
    assert [h for p,h in replay['snapshot_artifact_hashes'].items() if p.replace('\\','/')==START.relative_to(ROOT).as_posix()]==[digest(START)]
    protected=json.loads((ROOT/'results/copy_cross_output_transfer/plan.json').read_text())['protected_hashes']
    paths=[ROOT/'docs/protocols/copy-exposure-continuation.md',ROOT/'experiments/copy_exposure_continuation.py',ROOT/'experiments/audit_copy_exposure_continuation.py',START]
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and (ROOT/'experiments') in Path(p).resolve().parents:paths.append(Path(p).resolve())
    for p in paths:protected[p.relative_to(ROOT).as_posix()]=digest(p)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    OUT.mkdir();write(OUT/'preflight.json',pre)
    write(OUT/'plan.json',dict(start=6000,stages=STAGES,arms=ARMS,protected=protected,preflight_sha256=digest(OUT/'preflight.json')))
    report=dict(status='running',plan_sha256=digest(OUT/'plan.json'),runs=[],initial=evaluate_checkpoint(START,data,OUT))
    original=torch.load(ROOT/'results/copy_position_sites/3767_cross_only_final_scores.pt',weights_only=True)
    initial=torch.load(ROOT/report['initial']['scores'],weights_only=True)
    for p in data['pools']:
        assert torch.equal(initial[p]['logits'],original[p]['logits']) and initial[p]['generated']==original[p]['generated']
    write(OUT/'report.json',report)
    starts={a:START for a in ARMS}
    for stop in STAGES:
        pair=[]
        for arm in ARMS:
            dest=OUT/f'3767_{arm}_{stop}.pt'
            run=continuation(starts[arm],arm,data,'native',stop,dest)
            run.update(arm=arm,step=stop,start_sha256=digest(starts[arm]),checkpoint_sha256=digest(dest),evaluation=evaluate_checkpoint(dest,data,OUT))
            pair.append(run);report['runs'].append(run);starts[arm]=dest
            write(OUT/'report.json',report)
            print(arm,stop,run['evaluation']['metrics']['test_10']['all']['exact_accuracy'],flush=True)
        assert pair[0]['stream']==pair[1]['stream'] and pair[0]['position_stream']==pair[1]['position_stream']
        aa,bb=[torch.load(ROOT/r['checkpoint'],weights_only=True) for r in pair]
        for k in ('generator','position_generator'):assert torch.equal(aa[k],bb[k])
    final={r['arm']:r['evaluation']['metrics'] for r in report['runs'] if r['step']==8000}
    gates={a:gate(m) for a,m in final.items()}
    gain=final['both']['test_10']['repeated']['exact_accuracy']-final['cross_only']['test_10']['repeated']['exact_accuracy']
    flat=[]
    for r in report['runs']:
        for p,gs in r['evaluation']['metrics'].items():
            for g,m in gs.items():
                if m is not None:flat.append(dict(arm=r['arm'],step=r['step'],pool=p,group=g,**m))
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    report.update(status='complete',gates=gates,length10_gain=gain,exposure_specific_benefit=gates['both']['repair'] and gain>=.10,
        metrics_sha256=digest(OUT/'metrics.csv'),peak_process_working_set_bytes=peak_working_set(),protected_files=len(protected))
    write(OUT/'report.json',report)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
