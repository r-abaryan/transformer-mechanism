"""Tables and native SVG from audited replay records, without model execution."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/retention_learning_replay'


def csv_file(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def render_beta(trajectories):
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 405" role="img" aria-labelledby="title desc">',
           '<title id="title">Beta during exact retention training reproduction</title>',
           '<desc id="desc">Two seeds start at beta 0.881. Fixed-query beta rises toward 0.95; resampled-query beta falls toward 0.32. Curves show every recorded update.</desc>',
           '<rect width="1000" height="405" fill="#fff"/>',
           '<g font-family="Arial,sans-serif" font-size="13" fill="#202938">',
           '<text x="35" y="27" font-size="19">The same initial beta follows different learning paths</text>',
           '<text x="35" y="50">Actual beta after each Adam update. Two reused seeds; no new confirmation.</text>']
    colors={'fixed':'#b45309','resampled':'#0369a1'}
    for col,seed in enumerate((3461,3463)):
        left=70+col*480;top=90;width=390;height=230
        parts.append(f'<text x="{left}" y="76" font-size="16">Seed {seed}</text>')
        for y in (.2,.4,.6,.8,1.):
            py=top+(1-y)/.8*height
            parts.extend([f'<path d="M{left},{py}h{width}" stroke="#e5e7eb"/>',
                          f'<text x="{left-32}" y="{py+4}">{y:.1f}</text>'])
        for x in (0,1000,2000,3000):
            px=left+x/3000*width
            parts.append(f'<text x="{px-12}" y="{top+height+21}">{x}</text>')
        parts.append(f'<text x="{left+150}" y="{top+height+45}">Training update</text>')
        for arm in ('fixed','resampled'):
            trace=trajectories[seed,arm]
            points=[(0,trace['steps'][0]['beta_before'])]+[(r['step'],r['beta_after']) for r in trace['steps']]
            coords=' '.join(f'{left+x/3000*width:.3f},{top+(1-y)/.8*height:.3f}' for x,y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[arm]}" stroke-width="2"/>')
    parts.extend(['<text x="90" y="393" fill="#b45309">Fixed assigned query</text>',
                  '<text x="295" y="393" fill="#0369a1">Uniformly resampled query</text>',
                  '<text x="585" y="393">Beta alone does not explain every error.</text>','</g></svg>'])
    path=ROOT/'docs/figures/retention-learning-beta.svg'
    path.write_text('\n'.join(parts),encoding='utf-8')


def main():
    import sys
    sys.path.insert(0,str(ROOT/'experiments'))
    import retention_learning_replay as study
    import torch
    audit=json.loads((OUT/'audit.json').read_text())
    report=json.loads((OUT/'report.json').read_text())
    assert audit['passed'] and audit['report_sha256']==study.digest(OUT/'report.json')
    trajectories={};snapshots=[];step_rows=[];groups=[];summary=[];windows=[]
    for run in report['runs']:
        seed,arm=run['seed'],run['arm']
        t=json.loads((OUT/f'trajectory_{seed}_{arm}.json').read_text())
        trajectories[seed,arm]=t
        against=lambda r:abs(r['gradient'])>1e-7 and abs(r['alpha_after']-r['alpha_before'])>1e-12 and r['gradient']*(r['alpha_after']-r['alpha_before'])>0
        conflict=[p['step'] for p in t['snapshots'] if p['comparison']['opposite_resolved_signs']]
        checked=next(c for c in audit['runs'] if (c['seed'],c['arm'])==(seed,arm))
        count=sum(against(r) for r in t['steps'])
        assert count==checked['resolved_against_current_gradient']
        summary.append(dict(seed=seed,arm=arm,beta_initial=t['steps'][0]['beta_before'],beta_final=run['beta'],
                            first_measured_conflict=min(conflict) if conflict else None,
                            against_gradient_steps=count,total_steps=3000))
        for p in t['snapshots']:
            snapshots.append(dict(seed=seed,arm=arm,step=p['step'],beta=p['beta'],
                assigned_ce=p['objectives']['assigned']['ce'],all_query_ce=p['objectives']['all_queries']['ce'],
                assigned_accuracy=p['objectives']['assigned']['accuracy'],all_query_accuracy=p['objectives']['all_queries']['accuracy'],
                assigned_alpha_gradient=p['objectives']['assigned']['alpha_gradient'],
                all_query_alpha_gradient=p['objectives']['all_queries']['alpha_gradient'],
                conflict=p['comparison']['opposite_resolved_signs'],cosine=p['comparison']['gradient_cosines']['whole_model']))
            cp=torch.load(ROOT/p['checkpoint'],weights_only=True)
            for c in cp['cells']:
                for group,values in c['metrics'].items():
                    groups.append(dict(seed=seed,arm=arm,step=p['step'],load=c['load'],objective=c['objective'],group=group,
                                       **values,alpha_gradient=float(c['gradients']['write_logit'])))
        for row in t['steps']:
            step_rows.append(dict(seed=seed,arm=arm,**row,delta_alpha=row['alpha_after']-row['alpha_before'],
                                  against_gradient=against(row)))
        for begin,end in ((1,1000),(1001,2000),(2001,3000)):
            for load in (2,8,16,32):
                selected=[r for r in t['steps'] if begin<=r['step']<=end and r['load']==load]
                windows.append(dict(seed=seed,arm=arm,begin=begin,end=end,load=load,steps=len(selected),
                                    against_gradient_steps=sum(against(r) for r in selected),
                                    mean_gradient=math.fsum(r['gradient'] for r in selected)/len(selected),
                                    delta_alpha=math.fsum(r['alpha_after']-r['alpha_before'] for r in selected)))
    separations=[]
    for seed in (3461,3463):
        a,b=(trajectories[seed,arm]['snapshots'] for arm in ('fixed','resampled'))
        positive=[x['step'] for x,y in zip(a,b) if x['beta']-y['beta']>.01]
        separations.append(dict(seed=seed,first_measured_beta_gap_over_01=min(positive) if positive else None))
    csv_file('snapshots.csv',snapshots);csv_file('step_trace.csv',step_rows)
    csv_file('group_metrics.csv',groups);csv_file('load_interval_updates.csv',windows)
    render_beta(trajectories)
    output=dict(runs=summary,separations=separations,snapshot_rows=len(snapshots),group_rows=len(groups),
                step_rows=len(step_rows),load_interval_rows=len(windows))
    assert (len(snapshots),len(groups),len(step_rows),len(windows))==(28,672,12000,48)
    study.write(OUT/'summary.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
