"""Signed opposition annotation of audited saved gradients; no model evaluation."""
import json
import torch
import write_credit_assignment as study


def terms(vectors):
    x=vectors.flatten(1);energy=float(x.square().sum());joint=float(x.sum(0).square().sum())
    cross=joint-energy;dots=[]
    for i in range(len(x)):
        for j in range(i):dots.append(sum(float(a)*float(b) for a,b in zip(x[i],x[j])))
    independent=2*sum(dots)
    assert abs(cross-independent)<=1e-12*(1+abs(independent))
    return dict(position_energy=energy,joint_energy=joint,cross_energy=cross,
        joint_to_position_energy=joint/energy if energy else None,
        negative_pairs=sum(d<-1e-14 for d in dots),positive_pairs=sum(d>1e-14 for d in dots),
        unresolved_pairs=sum(abs(d)<=1e-14 for d in dots),pairs=len(dots),
        identity_error=abs(cross-independent))


def run():
    torch.set_num_threads(1)
    assert terms(torch.tensor([[1.,0.],[-1.,0.]],dtype=torch.float64))['cross_energy']==-2.
    assert terms(torch.eye(2,dtype=torch.float64))['cross_energy']==0.
    assert terms(torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64))['cross_energy']==2.
    out=study.OUT/'cross_terms.json';assert not out.exists(),'Preserve annotations'
    files={p:study.prior.digest(study.ROOT/p) for p in ('experiments/annotate_write_credit_assignment.py','docs/protocols/write-credit-cross-terms.md')}
    for p in ('report.json','audit.json','gradients.pt'):
        files[(study.OUT/p).relative_to(study.ROOT).as_posix()]=study.prior.digest(study.OUT/p)
    report=json.loads((study.OUT/'report.json').read_text());audit=json.loads((study.OUT/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==study.prior.digest(study.OUT/'report.json')
    assert report['gradients_sha256']==study.prior.digest(study.OUT/'gradients.pt')
    bundles=torch.load(study.OUT/'gradients.pt',weights_only=True)
    rows=[dict(seed=b['seed'],group=group,**terms(b['gradients'][group]['positions'])) for b in bundles for group in study.GROUPS]
    assert len(rows)==6 and sum(r['pairs'] for r in rows)==2*3*(32*31//2)
    study.prior.check_files(files)
    study.prior.write(out,dict(passed=True,post_primary_supplement=True,files=files,controls=3,
        model_evaluations=0,optimizer_calls=0,rows=rows,pairs=sum(r['pairs'] for r in rows)))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':run()
