"""Independent prefix labels, losses, gradients and final-checkpoint audit."""
import hashlib
import json

import torch
from state_tracking import ROOT, tensors, metrics, model_new, forward, save_json
from state_supervision import SEEDS, ARMS, STEPS, labels, objective, provenance
from audit_state_tracking import oracle, audit_data, independent
from mqa_training import hash_state
from position_distractors import digest


def targets_independent(ids):
    return torch.tensor([[oracle(row[:p+1]+[6]) for p in range(1,len(row)-1)] for row in ids.tolist()])


def loss_independent(logits, target, arm):
    loss = (logits[:,-1].logsumexp(-1)-logits[:,-1].gather(1,target[:,-1,None]).squeeze(1)).mean()
    if arm=='intermediate':
        z = logits[:,1:-1]
        loss = loss+(z.logsumexp(-1)-z.gather(-1,target[:,:,None]).squeeze(-1)).mean()
    return loss


def preflight(cells):
    audit_data(cells)
    prior = json.loads((ROOT/'results/state_tracking.json').read_text())
    assert prior['preflight']['passed']
    checked = 0
    for rows in cells.values():
        x,_ = tensors(rows)
        expected = targets_independent(x)
        assert torch.equal(labels(x),expected)
        checked += expected.numel()
    ids = torch.tensor([[4,2,0,1,1,3,0,1,0,6],[5,3,0,1,1,2,0,1,0,6]])
    target = targets_independent(ids)
    shifted = target.roll(1,1)
    assert not torch.equal(target,shifted)
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = model_new()
        for arm in ARMS:
            a,b = forward(model,ids),independent(model,ids)
            la,lb = objective(a,target,arm),loss_independent(b,target,arm)
            assert abs(la.item()-lb.item())<=1e-10
            ga = torch.autograd.grad(la,list(model.parameters()),retain_graph=True)
            gb = torch.autograd.grad(lb,list(model.parameters()))
            error = max((x-y).abs().max().item() for x,y in zip(ga,gb))
            assert error<=1e-10
            if arm=='intermediate':
                wrong = objective(a,shifted,arm)
                gw = torch.autograd.grad(wrong,list(model.parameters()))
                assert abs(wrong.item()-la.item())>1e-6
                assert max((x-y).abs().max().item() for x,y in zip(ga,gw))>1e-6
            records.append(dict(seed=seed,arm=arm,gradient_error=error,logit_error=(a-b).abs().max().item()))
    return dict(passed=True,prefix_labels_checked=checked,shifted_target_control_detected=True,models=records)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rp = ROOT/'results/state_supervision.json'
    r = json.loads(rp.read_text())
    assert r['phase']=='complete' and r['provenance']==provenance()
    ip = ROOT/'results/state_tracking_data.json'
    assert digest(ip)==r['input_sha256']
    cells = json.loads(ip.read_text())
    audit_data(cells)
    sp = ROOT/'results/state_supervision_scores.pt'
    assert digest(sp)==r['scores_sha256']
    saved = torch.load(sp,weights_only=True)
    dp = ROOT/'results/state_supervision_decision.json'
    assert digest(dp)==r['decision_sha256']
    frozen = json.loads(dp.read_text())
    assert frozen['decision']==r['decision'] and frozen['provenance']==r['provenance']
    assert {(x['seed'],x['arm']) for x in r['runs']}=={(s,a) for s in SEEDS for a in ARMS}
    max_error = max_loss = 0.
    evaluations = predictions = operation_predictions = 0
    independent_metrics = {}
    for run in r['runs']:
        torch.manual_seed(run['seed'])
        model = model_new()
        assert hash_state(model.state_dict())==run['initial_sha256']
        assert sum(p.numel() for p in model.parameters())==run['parameters']==4080
        cp = ROOT/run['checkpoint']
        assert digest(cp)==run['checkpoint_sha256']
        state = torch.load(cp,weights_only=True)
        assert state['provenance']==r['provenance']
        gen = torch.Generator().manual_seed(run['seed']+1000)
        stream = hashlib.sha256()
        for step in range(1,run['step']+1):
            n = 8 if step%2 else 12
            count = sum(len(v) for k,v in cells.items() if k.startswith(f'train/{n}/'))
            ix = torch.randint(count,(64,),generator=gen)
            stream.update(json.dumps(ix.tolist()).encode())
        assert stream.hexdigest()==run['stream_sha256'] and torch.equal(gen.get_state(),state['generator'])
        if run['status']!='trained':
            continue
        assert run['step']==STEPS and all(s['step'].item()==STEPS for s in state['optimizer']['state'].values())
        model.load_state_dict(state['model'])
        name = f"{run['seed']}/{run['arm']}"
        independent_metrics[name] = {}
        for key,rows in cells.items():
            x,y = tensors(rows)
            target = targets_independent(x)
            assert torch.equal(target[:,-1],y)
            with torch.no_grad():
                logits = torch.cat([independent(model,b) for b in x.split(128)])
            error = (logits-saved[name][key]).abs().max().item()
            assert error<=1e-10 and torch.equal(logits.argmax(-1),saved[name][key].argmax(-1))
            maximum_loss = (logits[:,-1].logsumexp(-1)-logits[:,-1].gather(1,y[:,None]).squeeze(1)).mean().item()
            m = run['evaluation'][key]
            good = logits[:,-1].argmax(-1)==y
            accuracy = int(good.sum())/len(y)
            assert accuracy==m['accuracy'] and int(good.sum())==m['correct'] and len(y)==m['count']
            for c in (0,1):
                assert int(good[y==c].sum())/int((y==c).sum())==m['class_accuracy'][str(c)]
            operation_accuracy = int((logits[:,1:-1].argmax(-1)==target).sum())/target.numel()
            assert operation_accuracy==m['operation_accuracy']
            loss_error = abs(maximum_loss-m['loss'])
            assert loss_error<=1e-10
            max_error,max_loss = max(max_error,error),max(max_loss,loss_error)
            evaluations += 1
            predictions += len(y)
            operation_predictions += target.numel()
            independent_metrics[name][key] = accuracy
    checks = {'completed':len(r['runs'])==6 and all(x['status']=='trained' and x['step']==STEPS for x in r['runs'])}
    for seed in SEEDS:
        a,b = [next(x for x in r['runs'] if x['seed']==seed and x['arm']==arm) for arm in ARMS]
        assert a['initial_sha256']==b['initial_sha256'] and a['stream_sha256']==b['stream_sha256']
        if not checks['completed']:
            for k in ('validation_gain','training_fit','mixed_retention'):
                checks[f'{seed}/{k}']=False
            continue
        am,bm = [independent_metrics[f'{seed}/{arm}'] for arm in ARMS]
        def avg(values,split):
            return (values[f'{split}/12/reset_free']+values[f'{split}/12/long_tail'])/2
        checks[f'{seed}/validation_gain'] = avg(bm,'validation')-avg(am,'validation')>=.10-1e-12
        checks[f'{seed}/training_fit'] = avg(bm,'train')>=.90
        checks[f'{seed}/mixed_retention'] = all(bm[f'validation/{n}/mixed']>=am[f'validation/{n}/mixed']-.05-1e-12 for n in (8,12))
    assert checks==r['decision']['checks'] and all(checks.values())==r['decision']['passed']
    result = dict(passed=True,report_sha256=digest(rp),provenance=provenance(),evaluations=evaluations,
                  final_predictions_checked=predictions,operation_predictions_checked=operation_predictions,
                  max_logit_error=max_error,max_loss_error=max_loss,argmax_differences=0,
                  decision_verified=True,paired_initialization_and_streams_verified=True)
    output = ROOT/'results/state_supervision_audit.json'
    assert not output.exists()
    save_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'}))


if __name__=='__main__':
    main()
