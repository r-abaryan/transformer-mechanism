"""Single oracle EOS exclusion with independent decoding and saved native controls."""
import json
import uuid
from pathlib import Path
import torch
from copy_site_failure import ROOT,PARENT,MODELS,POOLS,digest,write
from copy_position_sites import forward,make_model,inputs,generate
from copy_stopping import intervene

OUT=ROOT/'results/copy_site_eos'


def reference_decode(model,x,blocked):
    n=x.shape[1];prefix=[[1] for _ in x];answers=[[] for _ in x];done=[False]*len(x)
    for j in range(n+3):
        z=forward(model,x,torch.tensor(prefix),reference=True)[0][:,-1].tolist()
        allowed=[k for k in range(16) if not blocked or j>=n or k!=2]
        for i,row in enumerate(z):
            token=max(allowed,key=row.__getitem__);prefix[i].append(token)
            if not done[i]:answers[i].append(token)
            done[i] |= token==2
        if all(done):break
    return answers


def counts(rows,native,changed):
    n=len(rows[0]);out={}
    for g in ('all','distinct','repeated'):
        ids=[i for i,r in enumerate(rows) if g=='all' or (len(set(r))==n)==(g=='distinct')]
        if not ids:out[g]=None;continue
        a=[native[i]==rows[i]+[2] for i in ids];b=[changed[i]==rows[i]+[2] for i in ids]
        q=dict(count=len(ids),native_full=sum(a),blocked_full=sum(b),
            native_content=sum(native[i][:n]==rows[i] for i in ids),
            blocked_content=sum(changed[i][:n]==rows[i] for i in ids),
            rescues=sum(not x and y for x,y in zip(a,b)),harms=sum(x and not y for x,y in zip(a,b)))
        # Separate padded-tensor reduction, including exact generated length.
        gold=torch.tensor([rows[i]+[2] for i in ids])
        flags=[]
        for label,answers in (('native',native),('blocked',changed)):
            pad=torch.full_like(gold,-1)
            for j,i in enumerate(ids):pad[j,:min(len(answers[i]),n+1)]=torch.tensor(answers[i][:n+1])
            content=(pad[:,:n]==gold[:,:n]).all(1)
            full=(pad==gold).all(1)&torch.tensor([len(answers[i])==n+1 for i in ids])
            assert int(content.sum())==q[label+'_content'] and int(full.sum())==q[label+'_full']
            flags.append(full)
        assert int((~flags[0]&flags[1]).sum())==q['rescues']
        assert int((flags[0]&~flags[1]).sum())==q['harms']
        out[g]=q
    return out


def preflight():
    x=torch.tensor([[3,3,3]])
    def z(a,p):
        v=torch.zeros(len(a),16,dtype=torch.float64);v[:,2]=2;v[:,3]=1;return v
    assert intervene(z,x)==[[3,3,3,2]] and generate(z,x,6)==[[2]]
    def wrong(a,p):
        v=z(a,p)
        if p.shape[1]<3:v[:,2]=-torch.inf
        return v
    assert generate(wrong,x,6)!=intervene(z,x)
    c=counts(x.tolist(),[[2]],[[3,3,3,2]])
    assert c['all']['rescues']==1 and c['distinct'] is None
    torch.manual_seed(59);model=make_model('rope').eval()
    folder=ROOT/'results/test_runs'/('copy_site_eos_'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    p=Path(folder)/'initialized.pt';torch.save(model.state_dict(),p)
    loaded=make_model('rope').eval();loaded.load_state_dict(torch.load(p,weights_only=True))
    with torch.no_grad():
        for blocked in (False,True):
            fn=lambda a,b:forward(loaded,a,b)[0][:,-1]
            actual=intervene(fn,x) if blocked else generate(fn,x,6)
            assert actual==reference_decode(loaded,x,blocked)
    return dict(passed=True,initialized_reload=True,wrong_boundary_rejected=True,updates=0)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists()
    pre=preflight();write(OUT/'preflight.json',pre)
    parent=json.loads((PARENT/'report.json').read_text());audit=json.loads((PARENT/'audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==digest(PARENT/'report.json')
    runs=[next(r for r in parent['confirmation'] if (r['seed'],r['arm'])==pair) for pair in MODELS]
    paths=[Path(__file__),ROOT/'experiments/copy_stopping.py',ROOT/'experiments/copy_site_failure.py',
        ROOT/'docs/protocols/copy-site-eos.md',ROOT/'results/copy_site_failure/report.json',OUT/'preflight.json']
    oldlock=json.loads((PARENT/'lock.json').read_text())
    protected=dict(oldlock['protected'])
    for r in runs:
        for key in ('checkpoint','scores'):
            assert digest(ROOT/r[key])==r[key+'_sha256'];paths.append(ROOT/r[key])
    paths += [PARENT/'report.json',PARENT/'audit.json',PARENT/'data.json']
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
    assert all(digest(ROOT/p)==sha for p,sha in protected.items())
    write(OUT/'plan.json',dict(protected=protected,models=MODELS,pools=POOLS,training_updates=0))
    pools=json.loads((PARENT/'data.json').read_text())['pools'];cells=[];error=0.;evaluations=0
    for run in runs:
        model=make_model('rope').eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        saved=torch.load(ROOT/run['scores'],weights_only=True)
        for pool in POOLS:
            rows=pools[pool];native=[];changed=[];zs=[]
            with torch.no_grad():
                for x in torch.tensor(rows).split(64):
                    prefix,_=inputs(x);z=forward(model,x,prefix,reference=True)[0];zs.append(z)
                    fn=lambda a,b:forward(model,a,b)[0][:,-1]
                    a=generate(fn,x,x.shape[1]+3);b=intervene(fn,x)
                    assert a==reference_decode(model,x,False) and b==reference_decode(model,x,True)
                    native.extend(a);changed.extend(b)
            z=torch.cat(zs);delta=float((z-saved[pool]['logits']).abs().max());error=max(error,delta)
            assert delta<=1e-9 and torch.equal(z.argmax(-1),saved[pool]['logits'].argmax(-1))
            assert native==saved[pool]['generated']
            cell=dict(seed=run['seed'],arm=run['arm'],pool=pool,groups=counts(rows,native,changed),native=native,blocked=changed)
            for g,m in cell['groups'].items():
                if m:assert m['native_full']/m['count']==run['metrics'][pool][g]['exact_accuracy']
            cells.append(cell);evaluations+=len(rows)
            print(json.dumps({k:cell[k] for k in ('seed','arm','pool')}|cell['groups']['all']),flush=True)
    assert len(cells)==28 and all(digest(ROOT/p)==sha for p,sha in protected.items())
    write(OUT/'report.json',dict(cells=cells,training_updates=0,plan_sha256=digest(OUT/'plan.json'),
        audit=dict(passed=True,cells=len(cells),sequence_evaluations=evaluations,max_logit_error=error,generated_differences=0,protected_files=len(protected)),
        limitation='Opened checkpoints and oracle content length; shared forward structure. Conditional inference intervention, not learned repair or training mechanism.'))


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
