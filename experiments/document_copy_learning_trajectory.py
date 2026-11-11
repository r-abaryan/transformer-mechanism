"""Verify persisted trajectory evidence and produce its concise tables and figure."""
import csv
import json
import sys
from pathlib import Path

import torch
from copy_learning_trajectory import ROOT,OUT,PARENT,STEPS,summarize,audit_logits,equal
from copy_position_sites import digest,write


def main():
    report=json.loads((OUT/'report.json').read_text());assert report['status']=='complete'
    plan=json.loads((OUT/'plan.json').read_text());data=json.loads((PARENT/'data.json').read_text())
    assert digest(OUT/'plan.json')==report['plan_sha256']
    assert all(digest(ROOT/p)==v for p,v in plan['protected'].items())
    rows=[];files={};snapshots=0
    for run in report['runs']:
        assert [r['step'] for r in run['records']]==list(STEPS)
        for rec in run['records']:
            cp=ROOT/rec['checkpoint'];scores=cp.with_name(cp.stem+'_logits.pt')
            saved=torch.load(scores,weights_only=True);assert set(saved)==set(rec['metrics'])
            for pool,e in saved.items():
                source=data['pools']['test_10' if pool.startswith('paired_') else pool]
                expected=torch.tensor([r[4] for r in source]) if pool.startswith('paired_') else torch.tensor([r+[2] for r in source])
                assert torch.equal(e['labels'],expected)
                audit_logits(e['logits'],expected,source,rec['metrics'][pool],pool.startswith('paired_'))
                for group,m in rec['metrics'][pool].items():
                    if m is None:rows.append(dict(seed=run['seed'],arm=run['arm'],step=rec['step'],pool=pool,group=group,count=0,token_loss='',token_accuracy='',exact_accuracy='',correct='',eos='',margin=''))
                    else:rows.append(dict(seed=run['seed'],arm=run['arm'],step=rec['step'],pool=pool,group=group,**{k:m.get(k,'') for k in ('count','token_loss','token_accuracy','exact_accuracy','correct','eos','margin')}))
            files[str(cp.relative_to(ROOT))]=digest(cp);files[str(scores.relative_to(ROOT))]=digest(scores);snapshots+=1
    assert snapshots==52
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'metrics.csv').open(newline='',encoding='utf-8') as f:assert list(csv.DictReader(f))==[{k:str(v) for k,v in row.items()} for row in rows]
    # Standalone SVG from measured paired-source mean margins.
    fig=ROOT/'docs/figures/copy-learning-trajectory.svg';colors=['#b33131','#1676a3','#267d46','#7a49aa']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="420" viewBox="0 0 920 420">',
         '<rect width="920" height="420" fill="white"/>','<g font-family="Arial,sans-serif" font-size="13" fill="#222">',
         '<text x="28" y="23" font-size="17">EOS preference during exact historical replay</text>',
         '<text x="28" y="43">Mean margin = EOS logit minus correct fifth-token logit. Above zero favors EOS.</text>']
    allm=[rec['metrics'][f'paired_{n}']['all']['margin'] for run in report['runs'] for rec in run['records'] for n in (6,10)]
    low=min(-15,min(allm)-1);high=max(10,max(allm)+1)
    for panel,n in enumerate((6,10)):
        x0=65+panel*445;y0=80;w=370;h=245
        yy=lambda value:y0+h-(value-low)/(high-low)*h
        svg.append(f'<text x="{x0}" y="65">{n}-symbol source, identical correct prefix</text>')
        for value in (-10,0,10):
            svg.append(f'<line x1="{x0}" x2="{x0+w}" y1="{yy(value):.2f}" y2="{yy(value):.2f}" stroke="#bbb"/><text x="{x0-27}" y="{yy(value)+4:.2f}">{value}</text>')
        for step in (0,2000,4000,6000):
            xx=x0+w*step/6000;svg.append(f'<text x="{xx-14:.2f}" y="345">{step}</text>')
        svg.append(f'<text x="{x0+135}" y="365">Training update</text>')
        for color,run in zip(colors,report['runs']):
            points=' '.join(f"{x0+w*r['step']/6000:.2f},{yy(r['metrics'][f'paired_{n}']['all']['margin']):.2f}" for r in run['records'])
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i,(color,run) in enumerate(zip(colors,report['runs'])):
        x=30+i*225;svg.append(f'<text x="{x}" y="400" fill="{color}">{run["arm"]} {run["seed"]}</text>')
    svg.extend(['</g>','</svg>']);fig.write_text('\n'.join(svg),encoding='utf-8')
    table=[]
    for run in report['runs']:
        fit=run['short_fit_steps'];sig=run['failure_signature_steps'];final=run['records'][-1]['metrics']
        table.append(f"| {run['arm']} {run['seed']} | {fit[0] if fit else 'None'} | {', '.join(map(str,sig)) or 'None'} | {final['paired_6']['all']['correct']}/128 | {final['paired_10']['all']['eos']}/128 |")
    weak=report['runs'][0]
    details=[]
    for rec in weak['records']:
        m=rec['metrics'];short=m['paired_6']['all'];long=m['paired_10']['all'];fit=rec['step'] in weak['short_fit_steps']
        details.append(f"| {rec['step']} | {'Yes' if fit else 'No'} | {short['correct']} | {long['correct']} | {long['eos']} | {long['margin']-short['margin']:+.3f} |")
    text='''# When the copy failure develops

2026-10-07. **The weak seed's failure signature is present by update 1,000, before its first sampled short-training fit at 2,000.** None of the three successful controls meets that failure signature at any saved point. All four replays exactly reproduce historical endpoints, optimizer states, generators and sampling hashes. Intermediate weights were not saved originally; these are verified reconstructions.

[Frozen protocol](../../protocols/copy-learning-trajectory.md) · [Report and audit](../../../results/copy_learning_trajectory/report.json) · [All snapshot groups](../../../results/copy_learning_trajectory/metrics.csv)

## The question and method

When does source extension induce EOS while the short task is being learned? Replay the unchanged 6,000-update recipes for the weak cross-only seed, two successful cross-only seeds and the same-seed combined model. Save every 500 updates, including initialization. At each snapshot measure full short train/validation/test pools and paired six/ten-symbol sources at output position 5. Both paired sources receive BOS plus the same first four correct tokens.

Short-fit below means >=99% teacher-forced exact sequence accuracy in every short training-length distinct/repeated group, including terminal EOS. It differs from free generation. Failure signature means >=99% paired-short correctness with >=90% paired-long EOS. Thresholds were frozen; all occurrences are shown, not only the first.

| Model | First sampled short fit | Failure-signature updates | Final paired-short correct | Final paired-long EOS |
|---|---:|---|---:|---:|
'''+ '\n'.join(table)+'''

## Weak seed across training

All counts below are out of 128. Extension margin is the long minus short mean of `z_EOS - z_target`. Positive means extension increases EOS preference relative to the correct symbol.

| Update | Short training fit | Paired-short correct | Paired-long correct | Paired-long EOS | Extension margin |
|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(details)+'''

![Measured paired-source margins](../../figures/copy-learning-trajectory.svg)

## What this establishes

The weak model already predicts EOS on 79/128 long sources at update 500. By 1,000, paired-short predictions are all correct but 126/128 long predictions select EOS. Short fitting therefore does not remove this early divergence. Successful cross-only seed 3779 and combined seed 3767 reach 128/128 paired-long correctness at 1,000; cross-only 3793 reaches it at 2,000.

The weak trajectory is not monotonic. Long EOS counts fall to 92/128 and 103/128 at 5,000 and 5,500, then return to 128/128 at 6,000. Its short-training fit gate also fails at 5,000 after passing earlier. Do not describe these as stable convergence or a failure that begins only after short-task mastery. The 500-to-1,000 interval is a useful observed divergence window, not a uniquely identified causal transition.

This locates the failure along an exactly reproduced learning trajectory. It does not identify the parameter changes that create it. Different seeds have different initializations; the combined same-seed comparison changes positional exposure and therefore gradients throughout the model. The plotted margin is an actual final-output quantity, not an intermediate readout attribution.

The reused long pool is opened diagnostic evidence. Every snapshot uses supplied correct prefixes; the curves do not measure complete generated long copying. Transitions between saved points remain unresolved within 500 updates. Falling losses do not prove convergence. A future causal intervention needs a specific prediction derived from these trajectories.

## Checks, budget and sources

52 saved snapshots; four runs, 24,000 updates, 20.16M supervised targets and 3,828,000 training examples. CPU float64, one thread; batches 210/168/140/120 cycle over lengths 3/4/5/6 with 840 targets per update, no accumulation. The original complete-step profile is reused. CPU peak memory is unmeasured; GPU allocated/reserved memory is inapplicable.

Initialized saved reload, actual scalar audit, ties/absent groups and deliberate label/logit/missing-group faults pass. A two-update replay with observers exactly matches the unchanged trainer. Every snapshot's reference forward and scalar metric audit pass; persisted logits are checked separately. Final generated train/validation/test and layout metrics exactly reproduce the historical report. Shared forward helpers and PyTorch optimizer limit independence. Sources, checkpoints and protocols remain protected. The first preflight failed because synthetic logits were float32 against a float64 tolerance; fixtures were corrected before freezing or formal training, with no criterion change.

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3), inspected, describes ordered positional subset sampling. It supports the existing recipe, not the local scores or onset times. The replay, margins and temporal signatures are local measurements. Stop after this reconstruction; no mask, checkpoint-selected intervention or extra replay is queued.
'''
    dest=ROOT/'docs/topics/benchmarks/copy-learning-trajectory-results.md';dest.write_text(text,encoding='utf-8')
    write(OUT/'persisted-audit.json',dict(passed=True,snapshots=snapshots,csv_rows=len(rows),protected_files_unchanged=len(plan['protected']),
        snapshot_artifact_hashes=files,report_sha256=digest(OUT/'report.json'),csv_sha256=digest(OUT/'metrics.csv'),figure_sha256=digest(fig)))
    print(json.dumps(dict(snapshots=snapshots,csv_rows=len(rows),short_fit=[(r['seed'],r['arm'],r['short_fit_steps']) for r in report['runs']],signatures=[(r['seed'],r['arm'],r['failure_signature_steps']) for r in report['runs']]),indent=2))


if __name__=='__main__':main()
