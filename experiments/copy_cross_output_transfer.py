"""Frozen endpoint diagnostic; hooks preserve historical execution helpers."""
import argparse
import csv
import json
import time
import uuid

import torch

from copy_position_sites import ROOT, forward, make_model, inputs, generate, digest, write
from copy_length_exposure import metrics

OUT = ROOT / 'results/copy_cross_output_transfer'
ARMS = ('cross_only', 'both')
MODES = ('native', 'self', 'transfer', 'magnitude')


def load(path):
    model = make_model('rope').eval()
    model.load_state_dict(torch.load(path, weights_only=True)['model'])
    return model


def complete_groups(m):
    return {g: m.get(g) for g in ('all', 'distinct', 'repeated')}


def hybrid(recipient, donor, source, prefix, mode):
    donor_state = {}
    def remember_x(module, args, out):
        donor_state['x'] = out
    def remember_cross(module, args):
        donor_state['cross'] = args[0] - donor_state['x']
    hs = [donor.base.decoder.norm1.register_forward_hook(remember_x),
          donor.base.decoder.norm2.register_forward_pre_hook(remember_cross)]
    try:
        forward(donor, source, prefix)
    finally:
        for h in hs: h.remove()
    state = {}
    def capture(module, args, out):
        state['x'] = out
    def replace(module, args):
        cross = args[0] - state['x']
        other = donor_state['cross']
        rn, dn = cross.norm(dim=-1), other.norm(dim=-1)
        ratio = torch.where(rn > 1e-12, dn / rn.clamp_min(1e-12), torch.ones_like(rn))
        state['stats'] = torch.stack((rn, dn, ratio, (rn <= 1e-12).to(rn.dtype)), -1)
        if mode in ('self', 'transfer'): return (state['x'] + other,)
        if mode == 'magnitude': return (state['x'] + cross * ratio[..., None],)
        assert mode == 'native'
    hs = [recipient.base.decoder.norm1.register_forward_hook(capture),
          recipient.base.decoder.norm2.register_forward_pre_hook(replace)]
    try:
        z = forward(recipient, source, prefix)[0]
    finally:
        for h in hs: h.remove()
    return z, state['stats']


@torch.no_grad()
def evaluate(recipient, donor, rows, mode):
    zs, outputs, teacher_stats, generated_stats = [], [], [], []
    for x in torch.tensor(rows).split(64):
        prefix, labels = inputs(x)
        assert labels.tolist() == [r + [2] for r in x.tolist()]
        assert prefix.tolist() == [[1] + r for r in x.tolist()]
        z, stat = hybrid(recipient, donor, x, prefix, mode)
        zs.append(z); teacher_stats.append(stat.reshape(-1, 4))
        def step(xx, pp):
            zz, ss = hybrid(recipient, donor, xx, pp, mode)
            active = ~(pp == 2).any(-1)
            generated_stats.append(ss[:, -1][active])
            return zz[:, -1]
        outputs.extend(generate(step, x, x.shape[1] + 3))
    z = torch.cat(zs)
    saved = dict(logits=z, generated=outputs, teacher_stats=torch.cat(teacher_stats),
                 generated_stats=torch.cat(generated_stats))
    return complete_groups(metrics(z, rows, outputs)), saved


def decisions(ms):
    short = [p for p in ms['cross_only']['native'] if int(p.rsplit('_', 1)[1]) <= 6]
    cells = [('test_8', 'distinct'), ('test_8', 'repeated'), ('test_10', 'repeated')]
    def acc(arm, mode, p, g): return ms[arm][mode][p][g]['exact_accuracy']
    preserve = {arm: all(acc(arm, 'transfer', p, g) >= .99
                        for p in short for g in ('distinct', 'repeated')) for arm in ARMS}
    gain = acc('cross_only', 'transfer', 'test_10', 'repeated') - acc('cross_only', 'native', 'test_10', 'repeated')
    primary = preserve['cross_only'] and gain >= .10 and all(
        acc('cross_only', 'transfer', p, g) >= .95 and
        ms['cross_only']['transfer'][p][g]['on_time_eos'] >= .95 for p, g in cells)
    reciprocal = preserve['both'] and acc('both', 'native', 'test_10', 'repeated') - acc('both', 'transfer', 'test_10', 'repeated') >= .10
    return dict(short_preserved=preserve, primary_sufficiency=primary,
                reciprocal_sensitivity=reciprocal, magnitude_specificity=primary and
                acc('cross_only', 'transfer', 'test_10', 'repeated') - acc('cross_only', 'magnitude', 'test_10', 'repeated') >= .10)


def preflight():
    from audit_copy_cross_output_transfer import audit_cell
    directory = ROOT / 'results/test_runs' / ('cross-output-' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    models = []
    for seed in (41, 43):
        torch.manual_seed(seed); model = make_model('rope').eval()
        path = directory / f'{seed}.pt'; torch.save({'model': model.state_dict()}, path)
        models.append(load(path))
    rows = [[3, 4, 5], [4, 4, 3]]
    maximum = 0.
    for mode in MODES:
        donor = models[0] if mode == 'self' else models[1]
        m, s = evaluate(models[0], donor, rows, mode)
        maximum = max(maximum, audit_cell(models[0], donor, rows, mode, m, s))
        if mode == 'native': native = s
        if mode == 'self':
            torch.testing.assert_close(s['logits'], native['logits'], atol=1e-9, rtol=0)
            assert s['generated'] == native['generated']
    m, s = evaluate(models[0], models[1], [[3]*10], 'transfer')
    assert m['distinct'] is None
    audit_cell(models[0], models[1], [[3]*10], 'transfer', m, s)
    torch.save(s, directory/'saved.pt'); s = torch.load(directory/'saved.pt', weights_only=True)
    s['generated'][0][0] = (s['generated'][0][0]+1) % 16
    rejected = False
    try: audit_cell(models[0], models[1], [[3]*10], 'transfer', m, s)
    except AssertionError: rejected = True
    assert rejected
    from audit_copy_length_exposure import scalar_metrics
    labels = torch.tensor([[3,4,5,2],[4,4,3,2]])
    z = torch.full((2,4,16), -10., dtype=torch.float64).scatter_(-1, labels[...,None], 10.)
    for zz in (z, z.roll(1,-1)):
        a = complete_groups(metrics(zz, rows, [r+[2] for r in rows]))
        b = complete_groups(scalar_metrics(zz.tolist(), rows, [r+[2] for r in rows]))
        from audit_copy_position_exposure import close
        close(a,b)
    assert metrics(z.roll(1,-1), rows, [r+[2] for r in rows])['all']['token_accuracy'] == 0
    return dict(status='pass', fixture=str(directory.relative_to(ROOT)), max_logit_error=maximum,
                corrupted_generation_rejected=rejected, missing_group_explicit=True)


def run():
    from audit_copy_cross_output_transfer import audit_cell
    assert not OUT.exists(), 'Do not overwrite evidence'
    pre = preflight()
    protocol = ROOT/'docs/protocols/copy-cross-output-transfer.md'
    assert protocol.exists()
    protected = json.loads((ROOT/'results/copy_terminal_gradient_route/plan.json').read_text())
    # Carry forward the predecessor's explicitly protected evidence.
    protected = protected['protected']
    paths = [protocol, ROOT/'experiments/copy_cross_output_transfer.py',
             ROOT/'experiments/audit_copy_cross_output_transfer.py',
             ROOT/'results/copy_position_sites/data.json']
    paths += [ROOT/f'results/copy_learning_trajectory/3767_{a}_6000.pt' for a in ARMS]
    paths += [ROOT/f'results/copy_position_sites/3767_{a}_final_scores.pt' for a in ARMS]
    for path in paths: protected[path.relative_to(ROOT).as_posix()] = digest(path)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    replay = json.loads((ROOT/'results/copy_learning_trajectory/persisted-audit.json').read_text())
    for arm in ARMS:
        rel = f'results/copy_learning_trajectory/3767_{arm}_6000.pt'
        match = [h for p,h in replay['snapshot_artifact_hashes'].items() if p.replace('\\','/')==rel]
        assert match == [digest(ROOT/rel)]
    data = json.loads((ROOT/'results/copy_position_sites/data.json').read_text())['pools']
    data = {p:r for p,r in data.items() if not p.startswith('uniform_')}
    OUT.mkdir()
    write(OUT/'preflight.json',pre)
    write(OUT/'plan.json',dict(protected_hashes=protected, pools={p:len(r) for p,r in data.items()},
                               seed=3767,update=6000,modes=MODES,preflight_sha256=digest(OUT/'preflight.json')))
    models = {a:load(ROOT/f'results/copy_learning_trajectory/3767_{a}_6000.pt') for a in ARMS}
    ms, saved, maximum = {}, {}, 0.
    tick=time.perf_counter()
    for arm in ARMS:
        ms[arm]={}; saved[arm]={}
        old=torch.load(ROOT/f'results/copy_position_sites/3767_{arm}_final_scores.pt', weights_only=True)
        for mode in MODES:
            donor=models[arm] if mode=='self' else models[ARMS[1-ARMS.index(arm)]]
            ms[arm][mode]={}; saved[arm][mode]={}
            for p,rows in data.items():
                m,s=evaluate(models[arm],donor,rows,mode)
                maximum=max(maximum,audit_cell(models[arm],donor,rows,mode,m,s))
                if mode in ('native','self'):
                    torch.testing.assert_close(s['logits'],old[p]['logits'],atol=1e-9,rtol=0)
                    assert s['generated']==old[p]['generated']
                ms[arm][mode][p]=m; saved[arm][mode][p]=s
            print(arm,mode,'test10',ms[arm][mode]['test_10']['all']['exact_accuracy'],flush=True)
    torch.save(saved,OUT/'evidence.pt')
    flat=[]; norms={}
    for arm in ARMS:
        norms[arm]={}
        for mode in MODES:
            norms[arm][mode]={}
            for p, groups in ms[arm][mode].items():
                for g,m in groups.items():
                    if m is not None: flat.append(dict(recipient=arm,mode=mode,pool=p,group=g,**m))
                norms[arm][mode][p]={}
                for kind in ('teacher_stats','generated_stats'):
                    s=saved[arm][mode][p][kind]
                    norms[arm][mode][p][kind]=dict(rows=len(s),recipient_mean=float(s[:,0].mean()),
                        donor_mean=float(s[:,1].mean()),ratio_min=float(s[:,2].min()),
                        ratio_max=float(s[:,2].max()),ratio_mean=float(s[:,2].mean()),near_zero=int(s[:,3].sum()))
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    write(OUT/'report.json',dict(status='complete',metrics=ms,norms=norms,decisions=decisions(ms),
        seconds=time.perf_counter()-tick,max_reference_logit_error=maximum,protected_files=len(protected),
        plan_sha256=digest(OUT/'plan.json'),evidence_sha256=digest(OUT/'evidence.pt'),metrics_sha256=digest(OUT/'metrics.csv')))


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','run'))
    args=parser.parse_args()
    if args.action=='preflight': print(json.dumps(preflight(),indent=2))
    else: run()
