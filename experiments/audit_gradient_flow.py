"""Reload all measured training states and reproduce gradient observations."""
import json
from pathlib import Path
import torch
from gradient_flow import probe
from reserved_cycle import model_new
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def difference(a,b):
    if isinstance(a,dict):
        return max(difference(a[k],b[k]) for k in a)
    return abs(a-b)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    path=ROOT/'results/gradient_flow.json'; r=json.loads(path.read_text()); assert r['phase']=='complete'
    for n,h in r['source_sha256'].items(): assert digest(ROOT/'experiments'/n)==h
    for file,key in [('position_training.json','prior_sha256'),('position_training_splits.json','split_sha256'),('gradient_flow_probe.json','probe_sha256')]: assert digest(ROOT/'results'/file)==r[key]
    assert digest(ROOT/'docs/gradient-flow-protocol.md')==r['protocol_sha256']
    data=json.loads((ROOT/'results/position_training_splits.json').read_text())['validation']; indices=json.loads((ROOT/'results/gradient_flow_probe.json').read_text())
    assert len(indices)==len(set(indices))==128 and min(indices)>=0 and max(indices)<len(data)
    x=torch.tensor([data[i][0] for i in indices]); y=torch.tensor([data[i][1] for i in indices])
    prior=json.loads((ROOT/'results/position_training.json').read_text()); error=0.; count=0
    assert len(r['runs'])==5
    for run in r['runs']:
        old=next(v for v in prior['runs'] if v['seed']==run['seed'] and v['group']=='varied')
        assert run['minibatch_sha256']==old['minibatch_sha256'] and run['final_replay_error']==0
        assert [s['step'] for s in run['snapshots']]==[0,500,1500,3000]
        for snap in run['snapshots']:
            cp=ROOT/snap['checkpoint']; assert digest(cp)==snap['checkpoint_sha256']
            model=model_new(); model.load_state_dict(torch.load(cp,weights_only=True))
            measured=probe(model,x,y); error=max(error,difference(measured,{k:snap[k] for k in measured})); count+=1
            for site in snap['residuals'].values():
                a,b,c=site['skip_norm'],site['branch_norm'],site['total_norm']
                assert abs(c*c-(a*a+b*b+2*a*b*site['cosine']))<1e-10
            if snap['step']==3000:
                cpold=ROOT/old['checkpoint']; assert digest(cpold)==old['checkpoint_sha256']
                original=torch.load(cpold,weights_only=True)
                assert all(torch.equal(t,original[n]) for n,t in model.state_dict().items())
        print(run['seed'],'four measured states reproduced',flush=True)
    assert error<1e-10
    result=dict(passed=True,snapshots=count,max_metric_error=error,report_sha256=digest(path),audit_source_sha256=digest(Path(__file__)),scope='Checkpoint reloads and same-probe reproduction, independent norm-sum arithmetic, final replay equality; not an independent framework.')
    out=ROOT/'results/gradient_flow_audit.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
