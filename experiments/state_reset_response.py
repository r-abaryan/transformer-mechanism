"""Finite Boolean response grids and suffix interventions on saved readers."""
import csv
import itertools
import json
from pathlib import Path

import torch

from state_tracking import ROOT, model_new, forward, tensors, answer, save_json
from audit_state_tracking import independent, oracle
from state_sensitivity import summarize, scalar_summary
from position_distractors import digest

PATTERNS = ('correct_reset', 'inverted_reset', 'initial_only', 'constant', 'other')
DIRECTORY = ROOT/'results/state_reset_response'


def classify(bits, parity):
    # Order: (initial,reset) = 00,01,10,11.
    c=bits[0];a=bits[2]^c;b=bits[1]^c;d=bits[3]^bits[2]^bits[1]^c
    if (a,b,d)==(0,1,0): pattern='correct_reset' if c==parity else 'inverted_reset'
    elif (a,b,d)==(1,0,0): pattern='initial_only'
    elif (a,b,d)==(0,0,0): pattern='constant'
    else: pattern='other'
    return pattern,dict(constant=c,initial=a,reset=b,interaction=d)


def truth_classify(bits, parity):
    correct=tuple(r^parity for i,r in itertools.product((0,1),repeat=2))
    if tuple(bits)==correct:return 'correct_reset'
    if tuple(bits)==tuple(1-v for v in correct):return 'inverted_reset'
    if bits[0]==bits[1] and bits[2]==bits[3] and bits[0]!=bits[2]:return 'initial_only'
    if len(set(bits))==1:return 'constant'
    return 'other'


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not DIRECTORY.exists()
    protocol=ROOT/'docs/protocols/state-reset-response.md'
    parent=ROOT/'results/state_reset_exposure'
    old=json.loads((parent/'report.json').read_text(encoding='utf-8'))
    data=json.loads((parent/'data.json').read_text(encoding='utf-8'))['evaluation']
    runs=[v for v in old['runs'] if v['arm']=='independent']
    assert {r['seed'] for r in runs}=={3011,3019,3023}
    paths=[protocol,Path(__file__),parent/'report.json',parent/'data.json',parent/'audit.json',
           ROOT/'experiments/state_sensitivity.py']+[ROOT/p for p in old['source_dependencies']]
    paths += [ROOT/r[k] for r in runs for k in ('checkpoint','scores')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    assert digest(parent/'data.json')==old['data_sha256']
    assert all(digest(ROOT/p)==h for p,h in old['source_dependencies'].items())
    checked=0
    for bits in itertools.product((0,1),repeat=4):
        for parity in (0,1):
            pattern,coeff=classify(bits,parity)
            assert pattern==truth_classify(bits,parity)
            for j,(i,r) in enumerate(itertools.product((0,1),repeat=2)):
                assert bits[j]==(coeff['constant']^(coeff['initial']*i)^(coeff['reset']*r)^(coeff['interaction']*i*r))
            checked+=1
    prefixes=sorted({k.rsplit('/',1)[0] for k in data if '/long_tail_' in k})
    grids=[];counts=[];strata=[];metrics=[];scores={}
    evaluations=0;max_logit=max_archive=max_scalar=0.;wrong_labels=0
    for run in runs:
        assert digest(ROOT/run['checkpoint'])==run['checkpoint_sha256'] and digest(ROOT/run['scores'])==run['scores_sha256']
        archive=torch.load(ROOT/run['scores'],weights_only=True)
        model=model_new().eval();model.load_state_dict(torch.load(ROOT/run['checkpoint'],weights_only=True)['model'])
        scores[run['seed']]={}
        for prefix in prefixes:
            families={}
            for kind in ('agree','disagree'):
                key=prefix+'/long_tail_'+kind
                for j,(ids,y) in enumerate(data[key]):
                    suffix=tuple(ids[2:-1]);i,r=ids[0]-4,ids[1]-2
                    assert y==oracle(ids)==answer(i,ids[1:-1])
                    entry=families.setdefault(suffix,{})
                    assert (i,r) not in entry
                    entry[i,r]=dict(ids=ids,label=y,logits=archive[key][j])
            pool_grids=[]
            for suffix,family in families.items():
                assert set(family)==set(itertools.product((0,1),repeat=2))
                bits=[int(family[i,r]['logits'].argmax()) for i,r in itertools.product((0,1),repeat=2)]
                parity=sum(suffix)%2;pattern,coeff=classify(bits,parity)
                assert pattern==truth_classify(bits,parity)
                initial_changes=sum(bits[r]!=bits[2+r] for r in (0,1))
                correct=sum(bits[2*i+r]==(r^parity) for i,r in itertools.product((0,1),repeat=2))
                wrong_labels+=sum((i^parity)!=(r^parity) for i,r in itertools.product((0,1),repeat=2))
                row=dict(seed=run['seed'],pool=prefix,suffix=list(suffix),flips=sum(suffix),pattern=pattern,
                         coefficients=coeff,initial_changed_pairs=initial_changes,correct_inputs=correct,predictions=bits)
                grids.append(row);pool_grids.append(row)
            for pattern in PATTERNS:
                chosen=[v for v in pool_grids if v['pattern']==pattern]
                counts.append(dict(seed=run['seed'],pool=prefix,pattern=pattern,families=len(chosen),total_families=len(pool_grids)))
            for flips in sorted({v['flips'] for v in pool_grids}):
                chosen=[v for v in pool_grids if v['flips']==flips]
                strata.append(dict(seed=run['seed'],pool=prefix,flips=flips,families=len(chosen),
                                   correct_reset=sum(v['pattern']=='correct_reset' for v in chosen),
                                   initial_dependent=sum(v['initial_changed_pairs']>0 for v in chosen),
                                   correct_inputs=sum(v['correct_inputs'] for v in chosen)))
            # Response-grid counts must reproduce existing accuracy and initial-change metrics.
            native_accuracy=(oldval:=run['metrics'][prefix+'/long_tail_agree']['accuracy'])/2+run['metrics'][prefix+'/long_tail_disagree']['accuracy']/2
            assert abs(sum(v['correct_inputs'] for v in pool_grids)/(4*len(pool_grids))-native_accuracy)<=1e-10
            assert abs(sum(v['initial_changed_pairs'] for v in pool_grids)/(2*len(pool_grids))-run['reset_pairs'][prefix]['prediction_changes'])<=1e-10
            if prefix not in ('train/12','test/12'):continue
            ordered=sorted(families)
            rows=[(families[s][i,r]['ids'],families[s][i,r]['label']) for s in ordered for i,r in itertools.product((0,1),repeat=2)]
            cached=torch.stack([families[s][i,r]['logits'] for s in ordered for i,r in itertools.product((0,1),repeat=2)])
            x,y=tensors(rows)
            def compute(inputs,fn):
                with torch.no_grad():return torch.cat([fn(model,b)[:,-1] for b in inputs.split(128)])
            base,reference=compute(x,forward),compute(x,independent)
            ae=float((base-cached).abs().max());oe=float((base-reference).abs().max())
            assert max(ae,oe)<=1e-10 and torch.equal(base.argmax(-1),reference.argmax(-1))
            max_archive=max(max_archive,ae);max_logit=max(max_logit,oe);evaluations+=len(x)
            saved=dict(original=base);scores[run['seed']][prefix]=saved
            baseline_patterns=[classify([int(families[s][i,r]['logits'].argmax()) for i,r in itertools.product((0,1),repeat=2)],sum(s)%2)[0] for s in ordered for _ in range(4)]
            for name,positions in [('first',[2]),('last',[x.shape[1]-2]),('two',[2,x.shape[1]-2])]:
                changed=x.clone()
                for pos in positions:changed[:,pos]=1-changed[:,pos]
                labels=torch.tensor([oracle(v) for v in changed.tolist()])
                assert torch.equal(labels,1-y if len(positions)==1 else y)
                assert all(answer(ids[0]-4,ids[1:-1])==int(yy) for ids,yy in zip(changed.tolist(),labels))
                b,rb=compute(changed,forward),compute(changed,independent)
                error=float((b-rb).abs().max());assert error<=1e-10 and torch.equal(b.argmax(-1),rb.argmax(-1))
                max_logit=max(max_logit,error);evaluations+=len(changed);saved[name]=b
                for pattern in ('all',)+PATTERNS:
                    ix=[i for i,p in enumerate(baseline_patterns) if pattern=='all' or p==pattern]
                    if not ix:continue
                    values=summarize(base[ix],b[ix],y[ix],labels[ix]);scalar=scalar_summary(reference[ix],rb[ix],y[ix],labels[ix])
                    se=max(abs(values[k]-scalar[k]) for k in values);assert se<=1e-10;max_scalar=max(max_scalar,se)
                    metrics.append(dict(seed=run['seed'],pool=prefix,intervention=name,baseline_pattern=pattern,**values))
        print(f"Checked reset grids and toggles for {run['seed']}",flush=True)
    assert wrong_labels>0 and all(digest(ROOT/p)==h for p,h in hashes.items())
    DIRECTORY.mkdir();torch.save(scores,DIRECTORY/'scores.pt')
    report=dict(scope='cached response grids and checkpoint-only interventions; reused inputs',preserved_sha256=hashes,
                pattern_counts=counts,flip_strata=strata,grids=grids,interventions=metrics,scores_sha256=digest(DIRECTORY/'scores.pt'),
                verification=dict(passed=True,boolean_cases=checked,cached_grids=len(grids),evaluations=evaluations,
                                  wrong_initial_rule_errors=wrong_labels,argmax_changes=0,max_logit_error=max_logit,
                                  max_archive_error=max_archive,max_scalar_error=max_scalar))
    save_json(DIRECTORY/'report.json',report)
    for name,rows in [('patterns',counts),('flip_strata',strata),('interventions',metrics)]:
        with (DIRECTORY/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report['verification']),flush=True)


if __name__=='__main__':main()
