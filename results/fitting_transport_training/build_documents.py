"""Build a compact report and complete score CSV from the passing audit."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'report.json').read_text());a=json.loads((OUT/'audit.json').read_text())
assert a['passed']
destination=ROOT/'docs/topics/attention/fitting-transport-results.md'
assert not destination.exists() and not (OUT/'evaluation.csv').exists(), 'Preserve evidence'
rows=[]
for c in r['cells']:
    for group,g in c['metrics']['groups'].items():
        m=c['metrics']['complete_maps']
        rows.append(dict(seed=c['seed'],arm=c['arm'],step=c['step'],pool=c['pool'],group=group,
            count=g['count'],accuracy=g['accuracy'],ce=g['ce'],maps_count=m['count'],maps_correct=m['correct'],maps_accuracy=m['accuracy']))
with (OUT/'evaluation.csv').open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
label={'lbfgs':'Full','block_diagonal':'Block'}
train=['| Seed | Direction | All-load errors | Equal-load CE | Train-32 accuracy | Train-32 maps / 768 |','|---|---|---:|---:|---:|---:|']
heldout={p:['| Seed | Direction | Accuracy | CE | Complete maps |','|---|---|---:|---:|---:|'] for p in ['validation','test']}
budget=['| Seed | Full / block gradients | Full / block optimizer calls |','|---|---:|---:|']
for run in r['runs']:
    s=run['snapshots'][-1];pair=next(x for x in r['decision']['pairs'] if x['seed']==run['seed'])
    stat=pair['full' if run['arm']=='lbfgs' else 'block_diagonal'];m=s['training']['32']
    train.append(f"| {run['seed']} | {label[run['arm']]} | {stat['errors']} | {stat['ce']:.8g} | {100*m['groups']['all']['accuracy']:.4f}% | {m['complete_maps']['correct']} |")
    for p in heldout:
        c=next(x for x in r['cells'] if (x['seed'],x['arm'],x['step'],x['pool'])==(run['seed'],run['arm'],s['step'],p+'_32'))['metrics']
        heldout[p].append(f"| {run['seed']} | {label[run['arm']]} | {100*c['groups']['all']['accuracy']:.4f}% | {c['groups']['all']['ce']:.4f} | {c['complete_maps']['correct']} / {c['complete_maps']['count']} |")
for seed in [3473,3479,3481]:
    x,y=[next(z for z in r['runs'] if (z['seed'],z['arm'])==(seed,arm)) for arm in ['lbfgs','block_diagonal']]
    budget.append(f"| {seed} | {x['closure_evaluations']} / {y['closure_evaluations']} | {x['optimizer_calls']} / {y['optimizer_calls']} |")
moved=[s for c in a['checks'] for s in c['steps'] if s['moved']]
wolfe=sum(s['strong_wolfe'] for s in moved)
assert len(a['pair_controls'])==3 and all(c['historical_native_exact'] and c['first_step_identical'] and c['later_directions_differ'] for c in a['pair_controls'])
text=f'''# Curvature transport has a seed-dependent fitting effect

The predeclared all-seed fitting-benefit rule **fails**. Removing direct routing/decoder transport preserves zero training errors in seeds 3473/3479, but increases seed 3481's errors from 6 to 24. Full transport helps the weakest seed under this search policy and cap; it is not needed to reach perfect finite-pool fitting in the other two.

Training CE falls in all three removal branches. Held-out load-32 CE rises in every pair, with mixed accuracy/map effects. Lower training loss is not reliable retrieval on unseen histories.

## Method and prediction

The unchanged reader starts from the same saved Adam-12000 weights. Full L-BFGS uses d=-Hg. Block removal uses only H_RR and H_DD while retaining global secant history, search policy and the 128-gradient cap. Both branches still share the model objective; we remove direct cross-block action, not all joint learning. The [guide](fitting-transport-guide.md) explains the equations; the [protocol](../../protocols/fitting-transport-training.md) was frozen after preflight/profiling, before formal training.

The criterion required strictly more all-load training errors after removal in **every seed**. It passes only for 3481. CE is descriptive and does not replace that error criterion. This is a causal direction-rule contrast on reused starts/pools, not fresh confirmation or an architecture comparison.

## Final training

{chr(10).join(train)}

All six runs pass the existing 99% nonempty training-group gate. Both procedures leave errors in seed 3481; all-seed perfection remains unmet. Final loads 2/8/16, query groups and intermediate snapshots remain in the complete CSV. Its map fields measure whole histories and repeat across group rows; absent-group CE/accuracy are blank. Accuracy/map preservation here does not imply equal margins or equal learned representations.

## Held-out load 32

Validation:

{chr(10).join(heldout['validation'])}

Test:

{chr(10).join(heldout['test'])}

Test complete-map reliability remains weak. The removal branch loses test accuracy in two seeds and gains it in one; complete maps increase in two and decrease in the weakest. CE is substantially worse in every validation/test pair. Held-out load-16 CE also increases in every seed/split pair, with mixed accuracy effects. We did not perform a scale decomposition or measure calibration for these new endpoints, so its loss increase cannot be assigned to one cause from the earlier scale study.

## Budget and actual execution

{chr(10).join(budget)}

Formal execution: **{r['gradient_evaluations']:,} gradients, {r['optimizer_calls']:,} optimizer calls, {r['answer_targets']:,} supervised answers**. Each gradient covers 3,072 histories, 44,544 query answers and 98,304 input events. Total history/event exposures are {r['histories']:,}/{r['input_events']:,}. Different trial/update counts follow from the direction intervention; equal caps do not imply equal updates or wall time.

Six serial native Windows runs took {r['seconds']:.2f} seconds excluding imports, preflight, profiling and final audit. CPU float64, one thread. GPU peak allocated/reserved memory is inapplicable. Complete-step profiling included optimizer state; its tensor-payload counts are not process peak memory. No speed comparison is claimed.

## Verification

Two actual saved-runtime preflight methods passed in 9.651 seconds, including supplied logits/empty groups and deliberate gradient/direction/counter/loss/label/logit/missing-group/decision faults. The audit separately recomputes matrix gradients and source labels, replays every trial/state, checks removal directions through dense inverse-BFGS operators and checks every final/intermediate score cell through an independent forward/metric path. Final error counts are independently recounted from logits; some bookkeeping/CE reducers and the line-search replay are shared.

All native control traces and endpoints reproduce historical controls exactly. First steps agree; later directions differ in every pair. {wolfe}/{len(moved)} moved steps satisfy measured strong-Wolfe conditions. All **{a['cells']} score cells / {a['evaluation_targets']:,} repeated evaluated answers** pass audit, with {a['protected_files']} protected files unchanged. Maximum discrepancies: gradients {a['max_gradient_error']:.2e}, directions {a['max_direction_error']:.2e}, logits {a['max_logit_error']:.2e}. [Audit](../../../results/fitting_transport_training/audit.json), [raw report](../../../results/fitting_transport_training/report.json), [{len(rows)}-row CSV](../../../results/fitting_transport_training/evaluation.csv).

Three initialization seeds share one opened dataset; the six value shifts and repeated predictions are correlated. This 128-gradient phase follows an Adam warm-up, so it does not establish what from-scratch learning would do. No longer-input or independent-dataset test was added.

## What closes, what remains

Close cross-block direction variants at the declared limit. Presence and local cancellation did not imply an all-seed fitting benefit. We now have a measured conditional benefit in the weakest seed and two perfect witnesses without direct cross-block action, alongside poor held-out reliability.

The mechanisms of within-block preconditioning, representation learning and unseen-map failure remain unresolved. The next step is to consolidate the procedure and representation evidence into one causal account with explicit unknowns, before choosing another experiment. No new optimizer candidates, partitions or fitting extensions are queued. RL remains parked.

Sources inspected: [PyTorch 2.11 documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) and [implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/lbfgs.py), secant updates, directions and strong-Wolfe search; [Liu/Nocedal (1989), author abstract](https://users.iems.northwestern.edu/~nocedal/Abstracts/limited-memory.html), limited-memory/convex-theory scope. The block intervention, fitting criterion and measured scores are local. These sources neither predict the outcome nor prove convergence for this reader.
'''
destination.write_text(text,encoding='utf-8')
print(json.dumps(dict(report=str(destination),csv_rows=len(rows),measured_table_rows=21,wolfe_passes=wolfe,moved_steps=len(moved))))
