"""Read-only checkpoint diagnostic; no optimizer or original-artifact changes."""
import json
from pathlib import Path

import torch
from state_tracking import ROOT, model_new, forward, tensors, metrics, provenance, save_json
from audit_state_tracking import independent, oracle, audit_data
from position_distractors import digest


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT/'results/state_tracking_fit.json'
    assert not output.exists()
    rp = ROOT/'results/state_tracking.json'
    ip = ROOT/'results/state_tracking_data.json'
    ap = ROOT/'results/state_tracking_audit.json'
    report = json.loads(rp.read_text())
    audit = json.loads(ap.read_text())
    assert report['provenance']==provenance() and audit['passed']
    assert audit['report_sha256']==digest(rp) and report['input_sha256']==digest(ip)
    cells = json.loads(ip.read_text())
    assert audit_data(cells)==report['data_checks']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in (rp,ip,ap)}
    rows = []
    maximum = loss_maximum = 0.
    scores = {}
    for run in report['runs']:
        assert run['status']=='trained'
        cp = ROOT/run['checkpoint']
        assert digest(cp)==run['checkpoint_sha256']
        hashes[str(cp.relative_to(ROOT))] = digest(cp)
        model = model_new().eval()
        model.load_state_dict(torch.load(cp,weights_only=True)['model'])
        scores[run['seed']] = {}
        for key,data in cells.items():
            if not key.startswith('train/'):
                continue
            x,y = tensors(data)
            assert y.tolist()==[oracle(ids) for ids,_ in data]
            with torch.no_grad():
                a = torch.cat([forward(model,b)[:,-1] for b in x.split(128)])
                b = torch.cat([independent(model,c)[:,-1] for c in x.split(128)])
            error = (a-b).abs().max().item()
            assert error<=1e-10 and torch.equal(a.argmax(-1),b.argmax(-1))
            m = metrics(a,y)
            correct = b.argmax(-1)==y
            assert int(correct.sum())==m['correct'] and int(correct.sum())/len(y)==m['accuracy']
            loss = (b.logsumexp(-1)-b[torch.arange(len(y)),y]).mean().item()
            loss_error = abs(loss-m['loss'])
            assert loss_error<=1e-10
            maximum = max(maximum,error)
            loss_maximum = max(loss_maximum,loss_error)
            suffix = key.removeprefix('train/')
            val,test = [run['evaluation'][f'{s}/{suffix}'] for s in ('validation','test')]
            rows.append(dict(seed=run['seed'],cell=suffix,train=m,validation=val,test=test,
                             train_test_gap_pp=100*(m['accuracy']-test['accuracy'])))
            scores[run['seed']][key] = a
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    sp = ROOT/'results/state_tracking_fit_scores.pt'
    assert not sp.exists()
    torch.save(scores,sp)
    result = dict(passed=True,scope='post-hoc training-pool evaluation; no retraining',
                  inputs_sha256=hashes,source_sha256=digest(Path(__file__)),
                  provenance=provenance(),scores_sha256=digest(sp),
                  evaluations=len(rows),predictions_checked=sum(r['train']['count'] for r in rows),
                  max_logit_error=maximum,max_loss_error=loss_maximum,argmax_differences=0,rows=rows)
    save_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','provenance','inputs_sha256')}))
    for cell in dict.fromkeys(r['cell'] for r in rows):
        selected = [r for r in rows if r['cell']==cell]
        print(cell,json.dumps({s:[round(100*r[s]['accuracy'],4) for r in selected] for s in ('train','validation','test')}))


if __name__=='__main__':
    main()
