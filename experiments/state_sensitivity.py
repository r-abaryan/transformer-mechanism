"""Read-only interventions on original state-tracking checkpoints."""
import csv
import json
import math
import platform
from pathlib import Path

import torch

from state_tracking import ROOT, model_new, forward, answer, tensors, save_json
from audit_state_tracking import independent, oracle
from position_distractors import digest


def variants(ids):
    ops = ids[1:-1]
    last_set = max((i for i, op in enumerate(ops) if op >= 2), default=-1)
    eligible = [i for i, op in enumerate(ops) if op < 2]
    rows = [('initial', [0], last_set == -1)]
    if eligible:
        for name, index in [('first', 0), ('middle', len(eligible)//2), ('last', len(eligible)-1)]:
            pos = eligible[index]
            rows.append((name, [pos+1], pos > last_set))
    suffix = [i for i in eligible if i > last_set]
    if len(suffix) >= 2:
        rows.append(('two_suffix', [suffix[0]+1, suffix[-1]+1], False))
    result = []
    for name, positions, expected in rows:
        new = ids.copy()
        for pos in positions:
            new[pos] = 9-new[pos] if pos == 0 else 1-new[pos]
        old_label, new_label = oracle(ids), oracle(new)
        assert old_label == answer(ids[0]-4, ops)
        assert new_label == answer(new[0]-4, new[1:-1])
        assert bool(old_label ^ new_label) == expected
        result.append(dict(name=name, positions=positions, ids=new, label=new_label, relevant=expected))
    return result


def summarize(base, new, y, yn):
    a, b = base.argmax(-1), new.argmax(-1)
    gap = base[:, 1]-base[:, 0]
    ngap = new[:, 1]-new[:, 0]
    changed = a != b
    required = y != yn
    return dict(count=len(y), original_accuracy=float((a==y).double().mean()),
                intervened_accuracy=float((b==yn).double().mean()),
                both_correct=float(((a==y)&(b==yn)).double().mean()),
                prediction_changes=float(changed.double().mean()),
                response_agreement=float((changed==required).double().mean()),
                mean_probability_change=float((gap.sigmoid()-ngap.sigmoid()).abs().mean()),
                mean_gap_change=float((gap-ngap).abs().mean()),
                max_gap_change=float((gap-ngap).abs().max()))


def scalar_summary(base, new, ys, yns):
    values = {k: [] for k in ['original_accuracy','intervened_accuracy','both_correct','prediction_changes',
                             'response_agreement','mean_probability_change','mean_gap_change']}
    gaps = []
    def sigmoid(v):
        return 1/(1+math.exp(-v)) if v >= 0 else math.exp(v)/(1+math.exp(v))
    for x,z,y,yn in zip(base.tolist(),new.tolist(),ys.tolist(),yns.tolist()):
        a = int(x[1]>x[0]);b = int(z[1]>z[0]);g=x[1]-x[0];ng=z[1]-z[0]
        for k,v in dict(original_accuracy=a==y,intervened_accuracy=b==yn,both_correct=a==y and b==yn,
                        prediction_changes=a!=b,response_agreement=(a!=b)==(y!=yn),
                        mean_probability_change=abs(sigmoid(g)-sigmoid(ng)),mean_gap_change=abs(g-ng)).items():
            values[k].append(v)
        gaps.append(abs(g-ng))
    return dict(count=len(ys), **{k:sum(v)/len(v) for k,v in values.items()}, max_gap_change=max(gaps))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output = ROOT/'results/state_sensitivity.json'
    scorepath = ROOT/'results/state_sensitivity_scores.pt'
    csvpath = ROOT/'results/state_sensitivity_metrics.csv'
    assert not any(p.exists() for p in (output,scorepath,csvpath))
    protocol = ROOT/'docs/protocols/state-sensitivity.md'
    assert protocol.exists()
    original = json.loads((ROOT/'results/state_tracking.json').read_text())
    data = json.loads((ROOT/'results/state_tracking_data.json').read_text())
    paths = ['results/state_tracking.json','results/state_tracking_data.json','results/state_tracking_audit.json',
             'results/state_tracking_scores.pt','results/state_tracking_fit.json','results/state_tracking_fit_scores.pt',
             'experiments/state_tracking_fit.py','docs/protocols/state-tracking.md']
    paths += list(original['provenance'])
    paths += [run['checkpoint'] for run in original['runs']]
    hashes = {p:digest(ROOT/p) for p in dict.fromkeys(paths)}
    assert hashes['results/state_tracking_data.json'] == original['input_sha256']
    assert all(hashes[p]==h for p,h in original['provenance'].items())
    archived_test = torch.load(ROOT/'results/state_tracking_scores.pt',weights_only=True)
    archived_train = torch.load(ROOT/'results/state_tracking_fit_scores.pt',weights_only=True)
    selected = {k:v for k,v in data.items() if k.split('/')[0] in ('train','test') and
                (int(k.split('/')[1]) in (8,12) or k.startswith('test/24/'))}
    prepared = {}
    negative_labels = 0
    for key, rows in selected.items():
        batch = [];metadata = [];skipped = 0
        for i,(ids,y) in enumerate(rows):
            assert y == oracle(ids) == answer(ids[0]-4,ids[1:-1])
            vs = variants(ids)
            skipped += not any(v['name']=='two_suffix' for v in vs)
            for v in vs:
                batch.append([v['ids'],v['label']]);metadata.append(dict(original_index=i,**{k:v[k] for k in ('name','positions','relevant')}))
                # Wrong rule: assert every initial/toggle change flips the answer.
                if v['name']!='two_suffix':negative_labels += (1-y)!=v['label']
        prepared[key] = dict(rows=batch,metadata=metadata,two_suffix_skipped=skipped)
    assert negative_labels > 0
    metrics = [];baseline = [];strata = [];scores = {};evaluations = 0
    max_error = max_archive = max_scalar = 0.
    for run in original['runs']:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        m=model_new().eval();m.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        scores[run['seed']] = {}
        for key, rows in selected.items():
            x,y = tensors(rows);nx,ny=tensors(prepared[key]['rows'])
            def compute(ids,fn):
                with torch.no_grad():return torch.cat([fn(m,b)[:,-1] for b in ids.split(128)])
            a,ra=compute(x,forward),compute(x,independent)
            b,rb=compute(nx,forward),compute(nx,independent)
            oe=max(float((a-ra).abs().max()),float((b-rb).abs().max()))
            assert oe<=1e-10 and torch.equal(a.argmax(-1),ra.argmax(-1)) and torch.equal(b.argmax(-1),rb.argmax(-1))
            archived=(archived_train if key.startswith('train/') else archived_test)[run['seed']][key]
            ae=float((a-archived).abs().max());assert ae<=1e-10
            max_error=max(max_error,oe);max_archive=max(max_archive,ae);evaluations+=len(a)+len(b)
            scores[run['seed']][key]=dict(original=a,intervened=b)
            baseline.append(dict(seed=run['seed'],pool=key,count=len(y),accuracy=float((a.argmax(-1)==y).double().mean()),
                                 loss=float((a.logsumexp(-1)-a.gather(1,y[:,None]).squeeze(1)).mean())))
            metadata=prepared[key]['metadata']
            for name in ['initial','first','middle','last','two_suffix']:
                for relevant in [False,True]:
                    ids=[i for i,v in enumerate(metadata) if v['name']==name and v['relevant']==relevant]
                    if not ids:continue
                    orig=[metadata[i]['original_index'] for i in ids]
                    vals=summarize(a[orig],b[ids],y[orig],ny[ids]);sv=scalar_summary(ra[orig],rb[ids],y[orig],ny[ids])
                    se=max(abs(vals[k]-sv[k]) for k in vals);assert se<=1e-10;max_scalar=max(max_scalar,se)
                    positions=[metadata[i]['positions'] for i in ids]
                    metrics.append(dict(seed=run['seed'],pool=key,intervention=name,label_changes=relevant,
                                        operation_positions_min=min(p for ps in positions for p in ps),
                                        operation_positions_max=max(p for ps in positions for p in ps),**vals))
            if key.endswith(('reset_free','long_tail')):
                counts=[ids[1:-1].count(1) for ids,_ in rows]
                for flips in sorted(set(counts)):
                    ix=[i for i,c in enumerate(counts) if c==flips]
                    strata.append(dict(seed=run['seed'],pool=key,flip_count=flips,count=len(ix),
                                       accuracy=float((a[ix].argmax(-1)==y[ix]).double().mean()),
                                       mean_class_one_probability=float((a[ix,1]-a[ix,0]).sigmoid().mean())))
            print(json.dumps(dict(seed=run['seed'],pool=key,original=len(a),interventions=len(b),logit_error=oe)),flush=True)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    torch.save(scores,scorepath)
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    result=dict(scope='checkpoint-only diagnostic; reused inputs and derived interventions',
                python=platform.python_version(),torch=torch.__version__,device='cpu',dtype='float64',
                provenance={'experiments/state_sensitivity.py':digest(Path(__file__)),str(protocol.relative_to(ROOT)):digest(protocol)},
                preserved_inputs_sha256=hashes,baseline=baseline,metrics=metrics,flip_count_strata=strata,
                intervention_metadata={k:{'metadata':v['metadata'],'two_suffix_skipped':v['two_suffix_skipped']} for k,v in prepared.items()},
                verification=dict(passed=True,negative_label_disagreements=negative_labels,evaluations=evaluations,
                                  max_logit_error=max_error,max_archive_error=max_archive,max_scalar_metric_error=max_scalar,argmax_changes=0),
                scores_sha256=digest(scorepath),metrics_sha256=digest(csvpath))
    save_json(output,result);print(json.dumps(result['verification']))


if __name__=='__main__':main()
