"""Checkpoint-only group gradients and temporary CE descent probes."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from state_tracking import ROOT, model_new, forward, tensors, answer, save_json
from audit_state_tracking import independent, oracle
from state_reset_exposure import paired_reset_metrics
from position_distractors import digest

DIRECTORY = ROOT/'results/state_training_gradients'
PROTOCOL = ROOT/'docs/protocols/state-training-gradients.md'
SEEDS = (3011, 3019, 3023)
GROUPS = ('reset_free', 'mixed', 'long_tail_agree', 'long_tail_disagree')
KEYS = tuple(f'train/{n}/{kind}' for n in (8, 12) for kind in GROUPS)
WEIGHTS = {k: 1/12 if '/long_tail_' in k else 1/6 for k in KEYS}


def calculate(model, rows, reference=False, gradient=False, wrong=False):
    x, y = tensors(rows)
    params = list(model.parameters())
    total = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    scores = []
    for bx, by in zip(x.split(128), y.split(128)):
        with torch.set_grad_enabled(gradient):
            z = (independent if reference else forward)(model, bx)[:, -1]
            target = 1-by if wrong else by
            loss = F.softplus((1-2*target)*(z[:, 1]-z[:, 0])).mean() if reference else F.cross_entropy(z, target)
            if gradient:
                gs = torch.autograd.grad(loss, params)
                total += torch.cat([g.reshape(-1) for g in gs]) * (len(by)/len(y))
            scores.append(z.detach())
    z = torch.cat(scores)
    by = 1-y if wrong else y
    ce = float(F.cross_entropy(z, by))
    return z, total, dict(count=len(y), loss=ce, accuracy=float((z.argmax(-1)==by).double().mean()))


def scalar_loss(scores, labels):
    # Binary CE without the tensor CE implementation.
    losses = []
    for (a, b), y in zip(scores.tolist(), labels.tolist()):
        t = (1-2*y)*(b-a)
        losses.append(max(t, 0)+math.log1p(math.exp(-abs(t))))
    return math.fsum(losses)/len(losses)


def scalar_pairs(scores, pools):
    result = {}
    for n in (8, 12):
        prefix = f'train/{n}'
        a, b = [scores[prefix+'/long_tail_'+kind].tolist() for kind in ('agree', 'disagree')]
        labels = tensors(pools[prefix+'/long_tail_agree'])[1].tolist()
        probs = []
        changed = correct = 0
        for u, v, y in zip(a, b, labels):
            pu, pv = int(u[1]>u[0]), int(v[1]>v[0])
            changed += pu != pv
            correct += pu == y and pv == y
            # Stable probability from binary logits.
            def sigmoid(t):
                return 1/(1+math.exp(-t)) if t >= 0 else math.exp(t)/(1+math.exp(t))
            probs.append(abs(sigmoid(u[1]-u[0])-sigmoid(v[1]-v[0])))
        result[prefix] = dict(pairs=len(labels), prediction_changes=changed/len(labels),
                              both_correct=correct/len(labels), mean_probability_change=math.fsum(probs)/len(labels))
    return result


def set_parameters(params, original, direction, distance):
    at = 0
    with torch.no_grad():
        for p, saved in zip(params, original):
            p.copy_(saved+distance*direction[at:at+p.numel()].reshape_as(p))
            at += p.numel()
    assert at == len(direction)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert PROTOCOL.exists() and not DIRECTORY.exists(), 'Preserve executed attempts.'
    parent = ROOT/'results/state_reset_exposure'
    previous = json.loads((parent/'report.json').read_text(encoding='utf-8'))
    data = json.loads((parent/'data.json').read_text(encoding='utf-8'))
    pools = {k: data['evaluation'][k] for k in KEYS}
    runs = [r for r in previous['runs'] if r['arm']=='independent']
    assert {r['seed'] for r in runs} == set(SEEDS)
    paths = [PROTOCOL, Path(__file__), parent/'report.json', parent/'data.json', parent/'audit.json']
    paths += [ROOT/k for k in previous['source_dependencies']]
    paths += [ROOT/r[k] for r in runs for k in ('checkpoint', 'scores')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    assert digest(parent/'data.json') == previous['data_sha256']
    assert all(digest(ROOT/k)==h for k,h in previous['source_dependencies'].items())
    assert abs(math.fsum(WEIGHTS.values())-1) <= 1e-15
    for n in (8, 12):
        expanded = Counter()
        for kind in ('reset_free', 'long_tail', 'mixed'):
            base = data['training_base'][f'train/{n}/{kind}']
            for ids, y in base:
                for coin in (0, 1):
                    x = ids.copy()
                    if kind=='long_tail' and coin: x[0] = 9-x[0]
                    expanded[(tuple(x), y)] += 1
        expected = Counter()
        for kind in GROUPS:
            for ids, y in pools[f'train/{n}/{kind}']:
                assert oracle(ids)==answer(ids[0]-4,ids[1:-1])==y
                expected[(tuple(ids), y)] += 1 if kind.startswith('long_tail') else 2
        assert expanded == expected
        # Uniform base sampling, fair augmentation, and equal time per length.
        for kind in GROUPS:
            mass = sum(expected[(tuple(x),y)] for x,y in pools[f'train/{n}/{kind}'])/sum(expanded.values())/2
            assert abs(mass-WEIGHTS[f'train/{n}/{kind}']) <= 1e-15
        a, b = [tensors(pools[f'train/{n}/long_tail_{kind}']) for kind in ('agree','disagree')]
        assert torch.equal(a[0][:,1:], b[0][:,1:]) and torch.equal(a[1], b[1])
    DIRECTORY.mkdir()
    report = dict(status='running', protocol_sha256=digest(PROTOCOL), source_dependencies=hashes,
                  device='cpu', dtype='float64', threads=1, torch=str(torch.__version__),
                  new_training_updates=0, weights=WEIGHTS, distribution_enumeration_pass=True, runs=[])
    save_json(DIRECTORY/'report.json', report)
    saved = {}
    errors = dict(logit=0., gradient=0., scalar=0., archive=0., argmax_changes=0)
    evaluations = 0
    for run in runs:
        seed = run['seed']
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        assert digest(ROOT/run['scores'])==run['scores_sha256']
        model = model_new().eval()
        model.load_state_dict(torch.load(ROOT/run['checkpoint'], weights_only=True)['model'])
        params = list(model.parameters())
        original = [p.detach().clone() for p in params]
        archived = torch.load(ROOT/run['scores'], weights_only=True)
        grads, baseline, records = [], {}, []
        for key in KEYS:
            z, g, m = calculate(model, pools[key], gradient=True)
            rz, rg, rm = calculate(model, pools[key], reference=True, gradient=True)
            _, bad, _ = calculate(model, pools[key], gradient=True, wrong=True)
            le = float((z-rz).abs().max()); ge = float((g-rg).abs().max())
            ae = float((z-archived[key]).abs().max())
            assert max(le, ge, ae) <= 1e-10
            assert torch.equal(z.argmax(-1), rz.argmax(-1))
            bad_effect = float((g-bad).abs().max()); assert bad_effect > 1e-6
            assert bool(torch.isfinite(g).all())
            errors['logit']=max(errors['logit'],le); errors['gradient']=max(errors['gradient'],ge)
            errors['archive']=max(errors['archive'],ae)
            ce = scalar_loss(z, tensors(pools[key])[1])
            errors['scalar']=max(errors['scalar'],abs(m['loss']-ce),abs(m['loss']-rm['loss']))
            evaluations += len(z)
            baseline[key]=z; grads.append(g)
            records.append(dict(group=key, weight=WEIGHTS[key], **m, gradient_norm=float(g.norm()),
                                logit_error=le, gradient_error=ge, wrong_label_gradient_effect=bad_effect))
        matrix = torch.stack(grads)
        weights = torch.tensor([WEIGHTS[k] for k in KEYS], dtype=torch.float64)
        g = (weights[:,None]*matrix).sum(0)
        gnorm = float(g.norm()); assert gnorm > 0 and math.isfinite(gnorm)
        direction = -g/gnorm
        dot = matrix@matrix.T
        norms = matrix.norm(dim=1)
        cosine = [[float(dot[i,j]/(norms[i]*norms[j])) if norms[i]>0 and norms[j]>0 else None
                   for j in range(len(KEYS))] for i in range(len(KEYS))]
        slopes = matrix@direction
        magnitude = float(torch.cat([p.reshape(-1) for p in original]).norm())
        h = 1e-7*max(1., magnitude)
        conditions = {'baseline': baseline}
        displacements = []
        try:
            for name, distance in (('plus',h), ('minus',-h), ('plus_fine',h/10), ('minus_fine',-h/10)):
                set_parameters(params, original, direction, distance)
                scores, cell_metrics = {}, {}
                for key in KEYS:
                    z, _, m = calculate(model, pools[key])
                    rz, _, rm = calculate(model, pools[key], reference=True)
                    le = float((z-rz).abs().max())
                    assert le<=1e-10 and torch.equal(z.argmax(-1),rz.argmax(-1))
                    errors['logit']=max(errors['logit'],le)
                    errors['scalar']=max(errors['scalar'],abs(m['loss']-scalar_loss(z,tensors(pools[key])[1])), abs(m['loss']-rm['loss']))
                    evaluations += len(z)
                    scores[key]=z; cell_metrics[key]=m
                conditions[name]=scores
                displacements.append(dict(condition=name, distance=distance, metrics=cell_metrics,
                                          weighted_loss=math.fsum(WEIGHTS[k]*v['loss'] for k,v in cell_metrics.items())))
        finally:
            set_parameters(params, original, direction, 0.)
        assert all(torch.equal(p, q) for p,q in zip(params,original))
        dmetrics = {v['condition']:v['metrics'] for v in displacements}
        fine_pass = []
        for j, record in enumerate(records):
            key = record['group']
            coarse = (dmetrics['plus'][key]['loss']-dmetrics['minus'][key]['loss'])/(2*h)
            fine = (dmetrics['plus_fine'][key]['loss']-dmetrics['minus_fine'][key]['loss'])/(2*h/10)
            tolerance = 1e-5+1e-3*abs(float(slopes[j]))
            record.update(slope=float(slopes[j]), fd_coarse=coarse, fd_fine=fine,
                          fd_error=abs(fine-float(slopes[j])), fd_tolerance=tolerance,
                          fd_pass=abs(fine-float(slopes[j]))<=tolerance)
            fine_pass.append(record['fd_pass'])
        biggest = max(records, key=lambda v: abs(v['slope']))
        sign_error = abs(biggest['fd_fine']+biggest['slope'])
        sign_detected = abs(biggest['slope'])>1e-6 and sign_error>biggest['fd_tolerance']
        pair_metrics = {name:paired_reset_metrics(z,pools) for name,z in conditions.items()}
        for name, z in conditions.items():
            ref = scalar_pairs(z,pools)
            for prefix in ref:
                for key, v in ref[prefix].items():
                    errors['scalar']=max(errors['scalar'],abs(v-pair_metrics[name][prefix][key]))
        sr = dict(seed=seed, parameters=len(g), parameter_norm=magnitude, displacement=h,
                  weighted_loss=math.fsum(r['weight']*r['loss'] for r in records),
                  weighted_gradient_norm=gnorm, expected_slope=-gnorm,
                  negative_cosine_pairs=sum(cosine[i][j]<0 for i in range(8) for j in range(i+1,8) if cosine[i][j] is not None),
                  opposing_expected_direction_groups=[r['group'] for r in records if r['slope']>1e-8],
                  groups=records, cosine=cosine, displacements=displacements, reset_pairs=pair_metrics,
                  parameter_restoration_exact=True, finite_differences_pass=all(fine_pass),
                  reversed_slope_control_detected=sign_detected, reversed_slope_error=sign_error)
        saved[seed]=dict(keys=list(KEYS), parameter_names=[n for n,p in model.named_parameters()],
                         parameter_shapes=[list(p.shape) for p in params], gradients=matrix,
                         weighted_gradient=g, direction=direction, dot=dot, conditions=conditions)
        report['runs'].append(sr)
        save_json(DIRECTORY/'report.json',report)
        print(json.dumps(dict(seed=seed,gradient_norm=gnorm,opposing_groups=sr['opposing_expected_direction_groups'],
                              negative_pairs=sr['negative_cosine_pairs'],fd_pass=all(fine_pass))),flush=True)
    torch.save(saved,DIRECTORY/'evidence.pt')
    with (DIRECTORY/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        columns = ['seed']+list(report['runs'][0]['groups'][0])
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for run in report['runs']:
            writer.writerows(dict(seed=run['seed'],**r) for r in run['groups'])
    # Independent scalar reductions from persisted gradient tensors.
    loaded = torch.load(DIRECTORY/'evidence.pt',weights_only=True)
    for run in report['runs']:
        e = loaded[run['seed']];gs=e['gradients'].tolist();g=e['weighted_gradient'].tolist()
        wg=[math.fsum(WEIGHTS[k]*gs[i][j] for i,k in enumerate(KEYS)) for j in range(len(g))]
        errors['scalar']=max(errors['scalar'],max(abs(u-v) for u,v in zip(wg,g)))
        ng=math.sqrt(math.fsum(v*v for v in wg))
        errors['scalar']=max(errors['scalar'],abs(ng-run['weighted_gradient_norm']))
        for i in range(8):
            ni=math.sqrt(math.fsum(v*v for v in gs[i]))
            slope=-math.fsum(u*v for u,v in zip(gs[i],wg))/ng
            errors['scalar']=max(errors['scalar'],abs(slope-run['groups'][i]['slope']),abs(ni-run['groups'][i]['gradient_norm']))
            for j in range(8):
                nj=math.sqrt(math.fsum(v*v for v in gs[j]))
                d=math.fsum(u*v for u,v in zip(gs[i],gs[j]))
                errors['scalar']=max(errors['scalar'],abs(d-float(e['dot'][i,j])))
                if ni*nj>0: errors['scalar']=max(errors['scalar'],abs(d/(ni*nj)-run['cosine'][i][j]))
    assert max(errors.values())<=1e-10
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    report.update(status='complete', independent_errors=errors, baseline_and_displaced_evaluations=evaluations,
                  evidence_sha256=digest(DIRECTORY/'evidence.pt'), metrics_sha256=digest(DIRECTORY/'metrics.csv'),
                  prior_hashes_unchanged=True,
                  checks_pass=all(r['finite_differences_pass'] and r['reversed_slope_control_detected'] for r in report['runs']))
    save_json(DIRECTORY/'report.json',report)
    print(json.dumps(dict(status=report['status'],checks_pass=report['checks_pass'],errors=errors,evaluations=evaluations)))


if __name__=='__main__':
    main()
