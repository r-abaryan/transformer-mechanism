"""Saved-gradient transport review. No model evaluations or optimizer calls."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import fitting_representation as previous
import audit_full_pool_fitting as numerical

ROOT = previous.ROOT
SOURCE = ROOT / 'results/fitting_direction'
OUT = ROOT / 'results/fitting_transport_review'
SOURCES = ('experiments/review_fitting_transport.py', 'tests/test_fitting_transport.py')
PROTOCOL = 'docs/protocols/fitting-transport-review.md'
replay = previous.replay


def partition(values):
    names = list(values)
    assert set(names) == previous.ROUTING | previous.DECODER, 'Partition coverage'
    return names, torch.cat([torch.full((values[n].numel(),), n in previous.ROUTING, dtype=torch.bool) for n in names])


def components(g, state, mask):
    assert mask.dtype == torch.bool and mask.shape == g.shape and mask.any() and (~mask).any(), 'Partition shape'
    from_r = numerical.two_loop(g * mask, state)
    from_d = numerical.two_loop(g * ~mask, state)
    within = from_r * mask + from_d * ~mask
    cross = from_d * mask + from_r * ~mask
    return within, cross


def stats(g, direction, within, cross, mask):
    assert float(direction.norm()) > 0
    return dict(cross_norm_ratio=float(cross.norm()/direction.norm()),
                native_projection=float(g.dot(direction)),
                within_projection=float(g.dot(within)),
                cross_projection=float(g.dot(cross)),
                decoder_to_routing_projection=float((g*mask).dot(cross)),
                routing_to_decoder_projection=float((g*~mask).dot(cross)),
                within_cosine=float(direction.dot(within)/(direction.norm()*within.norm())))


def dense_inverse(state, n):
    """Rank-two dense inverse-BFGS updates, independent of two-loop application."""
    h = torch.eye(n, dtype=torch.float64) * state['H_diag']
    assert float(state['H_diag']) > 0
    assert len(state['old_dirs']) == len(state['old_stps']) == len(state['ro'])
    for y, s, rho in zip(state['old_dirs'], state['old_stps'], state['ro']):
        assert float(y.dot(s)) > 0 and abs(float(rho*y.dot(s))-1) < 1e-10, 'Secant curvature'
        hy = h @ y
        h = h + rho*(1+rho*y.dot(hy))*torch.outer(s,s) - rho*(torch.outer(s,hy)+torch.outer(hy,s))
    return h


def execute(out, runs, files, fixture=False):
    assert not out.exists(), 'Preserve evidence'
    replay.check_files(files)
    started = time.perf_counter(); records = []; inventory = []
    for run in runs:
        trace = torch.load(ROOT/run['trace'], weights_only=True)
        assert len(trace['calls']) == run['closure_evaluations'] and len(trace['steps']) == run['optimizer_calls']
        names, mask = partition(trace['calls'][0]['gradients'])
        inventory.append(dict(seed=run['seed'], arm=run['arm'], calls=len(trace['calls']), steps=len(trace['steps']),
                              routing_parameters=int(mask.sum()), decoder_parameters=int((~mask).sum())))
        for i, step in enumerate(trace['steps']):
            state = next(iter(step['optimizer']['state'].values()))
            g = numerical.flat(trace['calls'][step['first']]['gradients'], names)
            native = state['d']
            if run['arm'] == 'lbfgs':
                within, cross = components(g, state, mask)
                torch.testing.assert_close(within+cross, native, atol=1e-10, rtol=1e-8)
            else:
                torch.testing.assert_close(native, -g, atol=0, rtol=0)
                within, cross = native, torch.zeros_like(native)
            before, after = [numerical.flat(step[x], names) for x in ('before', 'after')]
            torch.testing.assert_close(after-before, float(state['t'])*native, atol=1e-10, rtol=1e-8)
            if i == 0: assert float(cross.norm()) == 0, 'First-step null'
            records.append(dict(seed=run['seed'], arm=run['arm'], call=i+1, first=step['first'], last=step['last'],
                                pairs=len(state['old_dirs']), step_length=float(state['t']), **stats(g,native,within,cross,mask)))
    replay.check_files(files)
    out.mkdir(parents=True)
    replay.write(out/'lock.json', dict(files=files, runs=runs, fixture=fixture, torch=torch.__version__))
    fields=list(records[0])
    with (out/'directions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(records)
    replay.write(out/'report.json',dict(inventory=inventory,records=records,seconds=time.perf_counter()-started,
        optimizer_calls=0,gradient_calls=0,model_forwards=0,device='cpu',dtype='float64',threads=1,
        lock_sha256=replay.digest(out/'lock.json'),csv_sha256=replay.digest(out/'directions.csv')))


def audit(out, persist=True):
    lock=json.loads((out/'lock.json').read_text());report=json.loads((out/'report.json').read_text())
    replay.check_files(lock['files']);assert lock['torch']==torch.__version__
    assert report['lock_sha256']==replay.digest(out/'lock.json') and report['csv_sha256']==replay.digest(out/'directions.csv')
    records=report['records'];expected=[];inventory=[];dense_count=0;max_error=0.
    for run in lock['runs']:
        trace=torch.load(ROOT/run['trace'],weights_only=True)
        assert len(trace['calls'])==run['closure_evaluations'] and len(trace['steps'])==run['optimizer_calls']
        names=list(trace['calls'][0]['gradients'])
        # Independently derive slices from the declared names and tensor lengths.
        ids=[];offset=0
        for name in names:
            assert name in previous.ROUTING|previous.DECODER
            count=trace['calls'][0]['gradients'][name].numel()
            if name in ('write_logit','key.weight'):ids.extend(range(offset,offset+count))
            offset+=count
        assert set(names)==previous.ROUTING|previous.DECODER, 'Partition coverage'
        mask=torch.zeros(offset,dtype=torch.bool);mask[ids]=True
        inventory.append(dict(seed=run['seed'],arm=run['arm'],calls=len(trace['calls']),steps=len(trace['steps']),
                              routing_parameters=len(ids),decoder_parameters=offset-len(ids)))
        for i,step in enumerate(trace['steps']):
            expected.append((run['seed'],run['arm'],i+1))
            row=next(r for r in records if (r['seed'],r['arm'],r['call'])==expected[-1])
            state=next(iter(step['optimizer']['state'].values()));g=numerical.flat(trace['calls'][step['first']]['gradients'],names)
            direction=state['d']
            if run['arm']=='lbfgs':
                h=dense_inverse(state,len(g));dense_count+=1
                from_r=-(h@(g*mask));from_d=-(h@(g*~mask))
                within=from_r*mask+from_d*~mask;cross=from_d*mask+from_r*~mask
                error=float((within+cross-direction).abs().max());max_error=max(max_error,error)
                torch.testing.assert_close(within+cross,direction,atol=1e-10,rtol=1e-8)
                torch.testing.assert_close(h,h.T,atol=1e-8,rtol=1e-8)
            else:
                torch.testing.assert_close(direction,-g,atol=0,rtol=0);within=direction;cross=torch.zeros_like(g)
            # Independent scalar sums check the reported projections and geometry.
            import math
            norm=lambda x:math.sqrt(math.fsum(float(v)**2 for v in x))
            dot=lambda x,y:math.fsum(float(a)*float(b) for a,b in zip(x,y))
            wanted=dict(cross_norm_ratio=norm(cross)/norm(direction),native_projection=dot(g,direction),
                        within_projection=dot(g,within),cross_projection=dot(g,cross),
                        decoder_to_routing_projection=dot(g*mask,cross),routing_to_decoder_projection=dot(g*~mask,cross),
                        within_cosine=dot(direction,within)/(norm(direction)*norm(within)))
            for k,v in wanted.items():assert math.isclose(row[k],v,rel_tol=1e-7,abs_tol=1e-10), (expected[-1],k,row[k],v)
            assert row['first']==step['first'] and row['last']==step['last'] and row['pairs']==len(state['old_dirs'])
            assert row['step_length']==float(state['t'])
            before,after=[numerical.flat(step[k],names) for k in ['before','after']]
            torch.testing.assert_close(after,before+float(state['t'])*direction,atol=1e-10,rtol=1e-8)
            assert math.isclose(wanted['decoder_to_routing_projection'],wanted['routing_to_decoder_projection'],rel_tol=1e-7,abs_tol=1e-10)
            if i==0:assert row['cross_norm_ratio']==0
    assert len(records)==len(expected) and [(r['seed'],r['arm'],r['call']) for r in records]==expected, 'Missing step'
    assert report['inventory']==inventory
    assert all(report[k]==0 for k in ['optimizer_calls','gradient_calls','model_forwards'])
    with (out/'directions.csv').open(newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==len(records)
    for row,saved in zip(rows,records):
        assert all(float(row[k])==v if isinstance(v,(int,float)) else row[k]==v for k,v in saved.items())
    result=dict(passed=True,archived_calls=len(records),dense_directions=dense_count,max_direction_error=max_error,
                protected_files=len(lock['files']),report_sha256=replay.digest(out/'report.json'))
    if persist:
        assert not (out/'audit.json').exists();replay.write(out/'audit.json',result)
    return result


def inputs():
    report=json.loads((SOURCE/'report.json').read_text());audit_record=json.loads((SOURCE/'audit.json').read_text())
    assert audit_record['passed'] and audit_record['report_sha256']==replay.digest(SOURCE/'report.json')
    lock=json.loads((SOURCE/'lock.json').read_text());files=dict(lock['files'])
    for run in report['runs']:
        assert replay.digest(ROOT/run['trace'])==run['trace_sha256'];files[run['trace']]=run['trace_sha256']
    for name in [*SOURCES,PROTOCOL,'results/fitting_direction/report.json','results/fitting_direction/audit.json',
                 'results/fitting_direction/lock.json','results/fitting_transport_review_preflight.json']:
        files[name]=replay.digest(ROOT/name)
    pre=json.loads((ROOT/'results/fitting_transport_review_preflight.json').read_text())
    assert pre['passed'] and pre['sources']=={n:files[n] for n in SOURCES}
    return report['runs'],files


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:print(json.dumps(audit(OUT)))
    else:execute(OUT,*inputs())
