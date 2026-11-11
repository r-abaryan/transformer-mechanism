"""Descriptive stage projections on matched exposure continuations, no intervention."""
import json
import uuid
import torch
from copy_position_sites import ROOT,make_model,forward,inputs,digest,write
from audit_copy_cross_output_transfer import parts,ff,norm

OUT=ROOT/'results/copy_exposure_stage_trace'
STAGES=('pre_cross','cross','cross_sum','cross_norm','ffn_sum','final')


@torch.no_grad()
def trace(model,rows):
    x=torch.tensor(rows);prefix,y=inputs(x);d=model.base.decoder;cache={};hooks=[]
    def post(name):return lambda m,a,o:cache.update({name:o.detach().clone()})
    def pre(name):return lambda m,a:cache.update({name:a[0].detach().clone()})
    for module,name in ((d.norm1,'pre_cross'),(d.norm2,'cross_norm'),(d.norm3,'final')):
        hooks.append(module.register_forward_hook(post(name)))
    for module,name in ((d.norm2,'cross_sum'),(d.norm3,'ffn_sum')):hooks.append(module.register_forward_pre_hook(pre(name)))
    try:z=forward(model,x,prefix)[0]
    finally:
        for hook in hooks:hook.remove()
    cache['cross']=cache['cross_sum']-cache['pre_cross']
    competitor=y.clone();terminal=z[:,-1].clone();terminal[:,2]=-torch.inf;competitor[:,-1]=terminal.argmax(-1)
    w=model.base.output.weight;axis=w[2]-w[competitor]
    margins={s:(cache[s]*axis).sum(-1) for s in STAGES}
    return dict(stages=cache,logits=z,labels=y,competitor=competitor,readout=w.detach().clone(),margins=margins)


@torch.no_grad()
def audit(model,rows,e):
    x=torch.tensor(rows);prefix=torch.tensor([[1]+r for r in rows]);a,c=parts(model,x,prefix);d=model.base.decoder
    ref=dict(pre_cross=a,cross=c,cross_sum=a+c);ref['cross_norm']=norm(ref['cross_sum'],d.norm2)
    ref['ffn_sum']=ref['cross_norm']+ff(d,ref['cross_norm']);ref['final']=norm(ref['ffn_sum'],d.norm3)
    maximum=0.
    for s in STAGES:
        err=float((ref[s]-e['stages'][s]).abs().max());assert err<1e-9;maximum=max(maximum,err)
        # Scalar dot products audit the diagnostic readout separately.
        vals=[]
        for i,row in enumerate(ref[s].tolist()):
            vals.append([sum(v*(float(e['readout'][2,k])-float(e['readout'][e['competitor'][i,j],k])) for k,v in enumerate(h)) for j,h in enumerate(row)])
        torch.testing.assert_close(torch.tensor(vals,dtype=torch.float64),e['margins'][s],atol=1e-9,rtol=0)
    rz=ref['final']@model.base.output.weight.T
    torch.testing.assert_close(rz,e['logits'],atol=1e-9,rtol=0)
    assert e['labels'].tolist()==[r+[2] for r in rows]
    expected=e['logits'][:,:,2]-e['logits'].gather(-1,e['competitor'][...,None]).squeeze(-1)
    torch.testing.assert_close(expected,e['margins']['final'],atol=1e-9,rtol=0)
    chain=('pre_cross','cross_sum','cross_norm','ffn_sum','final')
    increments=[e['margins'][b]-e['margins'][a] for a,b in zip(chain,chain[1:])]
    torch.testing.assert_close(sum(increments)+e['margins']['pre_cross'],expected,atol=1e-9,rtol=0)
    return maximum


def summarize(e,ids,kind,anchors):
    if not ids:return None
    n=e['labels'].shape[1]-1
    cells=[(i,j) for i in ids for j in (range(n) if kind=='content' else [n] if kind=='terminal' else [anchors[i]]) if j is not None]
    if not cells:return None
    margins={s:float(torch.stack([e['margins'][s][i,j] for i,j in cells]).mean()) for s in STAGES}
    return dict(positions=len(cells),examples=len(set(i for i,j in cells)),margin=margins,
        final_accuracy=sum(int(e['logits'][i,j].argmax())==int(e['labels'][i,j]) for i,j in cells)/len(cells),
        positive_final_margin=sum(float(e['margins']['final'][i,j])>0 for i,j in cells)/len(cells),
        cross_norm=float(torch.stack([e['stages']['cross'][i,j].norm() for i,j in cells]).mean()))


def main():
    assert not OUT.exists();torch.manual_seed(127)
    folder=ROOT/'results/test_runs'/('exposure-trace-'+uuid.uuid4().hex);folder.mkdir(parents=True)
    m=make_model('rope').eval();torch.save(m.state_dict(),folder/'initial.pt');m.load_state_dict(torch.load(folder/'initial.pt',weights_only=True))
    fixture=[[3,4,5],[3,3,4]];e=trace(m,fixture);audit(m,fixture,e)
    assert summarize(e,[],'terminal',[]) is None
    e['stages']['cross_sum'][0,0,0]+=.01
    try:audit(m,fixture,e)
    except AssertionError:pass
    else:raise AssertionError('Corrupted stage accepted')
    parent=ROOT/'results/copy_exposure_continuation';r=json.loads((parent/'report.json').read_text())
    protected=json.loads((parent/'plan.json').read_text())['protected']
    paths=[parent/'report.json',parent/'audit.json',ROOT/'docs/protocols/copy-exposure-stage-trace.md',ROOT/'experiments/copy_exposure_stage_trace.py']
    paths.extend(ROOT/run['checkpoint'] for run in r['runs']);paths.extend(ROOT/run['evaluation']['scores'] for run in r['runs'])
    for p in paths:protected[p.relative_to(ROOT).as_posix()]=digest(p)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    final={run['arm']:torch.load(ROOT/run['evaluation']['scores'],weights_only=True)['test_10']['generated'] for run in r['runs'] if run['step']==8000}
    data=json.loads((ROOT/'results/copy_position_sites/data.json').read_text())['pools'];rows=data['test_10']
    cohorts={k:[] for k in ('recovered','persistent','preserved','harmed')};anchors=[]
    for i,row in enumerate(rows):
        a,b=[final[k][i]==row+[2] for k in ('cross_only','both')]
        cohorts['preserved' if a and b else 'harmed' if a else 'recovered' if b else 'persistent'].append(i)
        p=next((j for j,t in enumerate(final['cross_only'][i]) if t==2),None);anchors.append(p if p is not None and p<len(row) else None)
    OUT.mkdir();write(OUT/'plan.json',dict(protected=protected,cohorts=cohorts,anchors=anchors,preflight_passed=True))
    runs=[dict(arm='shared',step=6000,checkpoint='results/copy_learning_trajectory/3767_cross_only_6000.pt')]+r['runs']
    saved={};summary={};maximum=0.
    for run in runs:
        key=f"{run['arm']}_{run['step']}";m=make_model('rope').eval();m.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        saved[key]={};summary[key]={}
        for pool in ('test_6','test_10'):
            e=trace(m,data[pool]);maximum=max(maximum,audit(m,data[pool],e));saved[key][pool]=e
            groups=cohorts if pool=='test_10' else {'short_control':list(range(len(data[pool])))}
            summary[key][pool]={g:{k:summarize(e,ids,k,anchors) for k in ('content','terminal','control_stop')} if pool=='test_10'
                               else {k:summarize(e,ids,k,[]) for k in ('content','terminal')} for g,ids in groups.items()}
        print(key,flush=True)
    torch.save(saved,OUT/'traces.pt')
    # Persisted reload: equations and scalar summaries, all positions and checkpoints.
    loaded=torch.load(OUT/'traces.pt',weights_only=True)
    for run in runs:
        key=f"{run['arm']}_{run['step']}";m=make_model('rope').eval();m.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        for pool in ('test_6','test_10'):
            audit(m,data[pool],loaded[key][pool])
            groups=cohorts if pool=='test_10' else {'short_control':list(range(len(data[pool])))}
            for g,ids in groups.items():
                for kind,val in summary[key][pool][g].items():assert summarize(loaded[key][pool],ids,kind,anchors)==val
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    write(OUT/'report.json',dict(status='complete',summary=summary,cohort_counts={k:len(v) for k,v in cohorts.items()},
        max_reference_error=maximum,audited_cells=len(runs)*2,positions=sum(e['labels'].numel() for run in loaded.values() for e in run.values()),
        plan_sha256=digest(OUT/'plan.json'),traces_sha256=digest(OUT/'traces.pt'),protected_files=len(protected),persisted_audit_passed=True))


if __name__=='__main__':
    torch.set_num_threads(1);main()
