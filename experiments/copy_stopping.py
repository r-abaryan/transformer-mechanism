"""Saved-checkpoint EOS intervention, with independent equations and controller."""
import json
import torch
from encoder_decoder_copy import ROOT, Model, logits, generate
from encoder_decoder_contract import independent
from position_distractors import digest
from recurrent_retrieval import write


def intervene(next_logits, source):
    n = source.shape[1]
    def constrained(x, prefix):
        z = next_logits(x, prefix).clone()
        if prefix.shape[1] <= n:
            z[:, 2] = -torch.inf
        return z
    return generate(constrained, source, n+3)


def measures(rows, scores, baseline, changed):
    n = len(rows[0])
    out = {}
    for group in ('all', 'distinct', 'repeated'):
        ids = [i for i,r in enumerate(rows) if group == 'all' or (len(set(r)) == n) == (group == 'distinct')]
        positions = []
        for j in range(n+1):
            targets = [rows[i][j] if j<n else 2 for i in ids]
            positions.append(dict(position=j+1, correct=sum(int(scores[i,j].argmax()) == t for i,t in zip(ids,targets)),
                eos_top=sum(int(scores[i,j].argmax()) == 2 for i in ids),
                eos_minus_correct=sum(float(scores[i,j,2]-scores[i,j,t]) for i,t in zip(ids,targets))/len(ids)))
        row = dict(count=len(ids), teacher_positions=positions)
        for name, outputs in (('ordinary',baseline),('blocked_eos',changed)):
            row[name] = dict(content_exact=sum(outputs[i][:n] == rows[i] for i in ids),
                full_exact=sum(outputs[i] == rows[i]+[2] for i in ids),
                position_correct=[sum(len(outputs[i])>j and outputs[i][j]==rows[i][j] for i in ids) for j in range(n)],
                early=sum(2 in outputs[i] and outputs[i].index(2)<n for i in ids),
                on_time=sum(2 in outputs[i] and outputs[i].index(2)==n for i in ids),
                late=sum(2 in outputs[i] and outputs[i].index(2)>n for i in ids),
                missing=sum(2 not in outputs[i] for i in ids))
        out[group] = row
    return out


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    target = ROOT/'results/copy_stopping.json'
    assert not target.exists(), 'Preserve completed attempts.'
    old = json.loads((ROOT/'results/encoder_decoder_copy.json').read_text())
    data = json.loads((ROOT/'results/encoder_decoder_copy_data.json').read_text())['pools']
    paths = [ROOT/'results/encoder_decoder_copy.json', ROOT/'results/encoder_decoder_copy_data.json',
             ROOT/'results/encoder_decoder_copy_audit.json', ROOT/'experiments/encoder_decoder_copy.py',
             ROOT/'experiments/encoder_decoder_contract.py', ROOT/'docs/protocols/encoder-decoder-copy.md']
    paths += [ROOT/r[k] for r in old['runs'] for k in ('checkpoint','scores')]
    preserved = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    provenance = {p:digest(ROOT/p) for p in ('experiments/copy_stopping.py','docs/protocols/copy-stopping.md')}
    # EOS always wins unless blocked; an off-by-one controller fails this boundary.
    x = torch.tensor([[3,3,3]])
    def script(x,y):
        z = torch.zeros(len(x),16,dtype=torch.float64)
        z[:,2], z[:,3] = 2., 1.
        return z
    assert intervene(script,x) == [[3,3,3,2]]
    assert generate(script,x,6) == [[2]]
    def wrong(x,y):
        z = script(x,y)
        if y.shape[1] < 3: z[:,2] = -torch.inf
        return z
    assert generate(wrong,x,6) == [[3,3,2]]
    cells, details = [], {}
    max_error = 0.
    audited_tokens = 0
    for run in old['runs']:
        for k in ('checkpoint','scores'): assert digest(ROOT/run[k]) == run[k+'_sha256']
        model = Model().eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        saved = torch.load(ROOT/run['scores'],weights_only=True)
        for pool,rows in data.items():
            source = torch.tensor(rows)
            with torch.no_grad():
                changed = intervene(lambda a,b:logits(model,a,b)[:,-1],source)
                reference = []
                shift = torch.tensor([[1]+r for r in rows])
                s,t = model.features(source,shift)
                z = independent(model,s,t,torch.zeros_like(source,dtype=torch.bool),torch.zeros_like(shift,dtype=torch.bool))
                error = (z-saved[pool]['logits']).abs().max().item()
                max_error = max(max_error,error)
                assert error < 1e-9
                for row in rows:
                    sequence = []
                    for step in range(len(row)+3):
                        a,b = torch.tensor([row]),torch.tensor([[1]+sequence])
                        s,t = model.features(a,b)
                        v = independent(model,s,t,torch.zeros_like(a,dtype=torch.bool),torch.zeros_like(b,dtype=torch.bool))[0,-1]
                        # Independent selection excludes EOS by candidate enumeration.
                        candidates = [c for c in range(16) if c != 2 or step >= len(row)]
                        token = max(candidates,key=lambda c:float(v[c]))
                        sequence.append(token)
                        if token == 2: break
                    reference.append(sequence)
                assert changed == reference
                audited_tokens += sum(map(len,reference))
            m = measures(rows,saved[pool]['logits'],saved[pool]['generated'],changed)
            # Separate tensor arithmetic checks content/sequence counts and teacher metrics.
            for g,q in m.items():
                ids = [i for i,r in enumerate(rows) if g=='all' or (len(set(r))==len(r)) == (g=='distinct')]
                labels = torch.tensor([rows[i]+[2] for i in ids])
                zz = z[ids]
                assert (zz.argmax(-1)==labels).sum(0).tolist() == [v['correct'] for v in q['teacher_positions']]
                margins = (zz[:,:,2]-zz.gather(2,labels[:,:,None]).squeeze(2)).mean(0)
                assert max(abs(float(v)-p['eos_minus_correct']) for v,p in zip(margins,q['teacher_positions'])) < 1e-9
                for key,output in (('ordinary',saved[pool]['generated']),('blocked_eos',reference)):
                    padded = torch.full_like(labels,-1)
                    for j,i in enumerate(ids):
                        seq = output[i][:labels.shape[1]]
                        padded[j,:len(seq)] = torch.tensor(seq)
                    assert int((padded[:,:-1]==labels[:,:-1]).all(1).sum()) == q[key]['content_exact']
                    assert int((padded==labels).all(1).sum()) == q[key]['full_exact']
            cells.append(dict(seed=run['seed'],pool=pool,groups=m))
            details[f"{run['seed']}/{pool}"] = changed
            print(json.dumps(dict(seed=run['seed'],pool=pool,blocked=m['all']['blocked_eos'])),flush=True)
    for name,h in preserved.items(): assert digest(ROOT/name)==h
    decisions = {str(n):{kind:all(c['groups'][g]['blocked_eos'][kind]/c['groups'][g]['count']>=.95
        for c in cells if c['pool']==f'test_{n}' for g in ('distinct','repeated'))
        for kind in ('content_exact','full_exact')} for n in (3,4,6,8)}
    write(target,dict(status='complete',device='cpu',dtype='float64',training_updates=0,
        provenance=provenance,preserved=preserved,cells=cells,generated=details,decisions=decisions,
        audit=dict(passed=True,pools=len(cells),generated_tokens=audited_tokens,generated_differences=0,
                   max_logit_error=max_error,early_eos_and_off_by_one_controls=True)))


if __name__=='__main__': main()
