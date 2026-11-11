"""Standalone validation and paired test plots from audited external results."""
import csv
import html
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from position_distractors import digest
ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/wikitext2_training.json';ap=ROOT/'results/wikitext2_training_audit.json'
    r=json.loads(rp.read_text());a=json.loads(ap.read_text());assert r['phase']=='complete' and a['passed'] and a['report_sha256']==digest(rp)
    out=ROOT/'docs/figures/wikitext2-training.svg';cp=ROOT/'results/wikitext2_training_plot_data.csv';mp=ROOT/'results/wikitext2_training_plot_manifest.json'
    assert not any(p.exists() for p in (out,cp,mp))
    colors=dict(mha='#0072b2',mqa='#d55e00');rows=[]
    def text(x,y,value,size=14,color='#243449'):
        return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(str(value))}</text>'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="790" viewBox="0 0 1100 790" role="img">',
         '<title>WikiText-2: matched MHA and MQA training</title>',
         '<desc>Top: validation mean negative log-likelihood by update, with thin lines for paired seeds and thick lines for means. Bottom: final test perplexity by seed and architecture, with the training-count unigram baseline. Lower is better.</desc>',
         '<rect width="1100" height="790" fill="white"/>',text(30,34,'WikiText-2: fixed small-model comparison',24),
         text(30,62,'Width16; two blocks; word vocabulary33,278; FP32 RTX4060; three paired seeds; context resets every64 words.'),
         text(30,89,'MHA',14,colors['mha']),text(150,89,'MQA',14,colors['mqa'])]
    left=90;width=920;top=135;height=235
    maximum=math.ceil(max(p['validation']['loss'] for run in r['runs'] for p in run['curve']))
    xx=lambda step:left+width*step/6000;yy=lambda loss:top+height*(1-loss/maximum)
    svg.append(text(30,120,'Validation loss in nats per target; PPL = exp(loss); lower is better',16))
    for i in range(7):
        value=maximum*i/6;y=yy(value);svg.append(f'<path d="M{left} {y}h{width}" stroke="#dce2e8"/>');svg.append(text(40,y+4,f'{value:.1f}',12))
    for step in range(0,6001,1000):svg.append(text(xx(step)-15,395,step,12))
    for group,color in colors.items():
        runs=[run for run in r['runs'] if run['group']==group]
        for run in runs:
            points=[]
            for p in run['curve']:
                points.append((xx(p['step']),yy(p['validation']['loss'])));rows.append(['validation',group,run['seed'],p['step'],p['validation']['loss'],p['validation']['perplexity']])
            svg.append(f'<polyline points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in points)}" stroke="{color}" stroke-opacity=".3" fill="none" stroke-width="1.4"/>')
        means=[(step,sum(next(p['validation']['loss'] for p in run['curve'] if p['step']==step) for run in runs)/len(runs)) for step in range(0,6001,1000)]
        svg.append(f'<polyline points="{" ".join(f"{xx(t):.3f},{yy(v):.3f}" for t,v in means)}" stroke="{color}" fill="none" stroke-width="3"/>')
    svg.append(text(320,420,'Optimizer updates; thin = seed, thick = arithmetic mean',13))
    top=485;height=220;maximum=math.ceil(max([run['test']['perplexity'] for run in r['runs']]+[r['unigram_test']['perplexity']])/100)*100
    yy=lambda ppl:top+height*(1-ppl/maximum)
    svg.append(text(30,460,'Final test perplexity by paired seed; all test targets audited independently',16))
    for i in range(6):
        value=maximum*i/5;y=yy(value);svg.append(f'<path d="M{left} {y}h{width}" stroke="#dce2e8"/>');svg.append(text(30,y+4,f'{value:.0f}',12))
    baseline=r['unigram_test']['perplexity'];svg.append(f'<path d="M{left} {yy(baseline)}h{width}" stroke="#777" stroke-dasharray="5 4"/>')
    svg.append(text(720,yy(baseline)-8,f'Unigram {baseline:.2f}',12))
    seeds=sorted({run['seed'] for run in r['runs']})
    for i,seed in enumerate(seeds):
        x=left+width*(i+.5)/3
        for group,shift in (('mha',-32),('mqa',32)):
            run=next(v for v in r['runs'] if (v['seed'],v['group'])==(seed,group));p=run['test']['perplexity'];y=yy(p)
            svg.append(f'<rect x="{x+shift-24}" y="{y}" width="48" height="{top+height-y}" fill="{colors[group]}"/>')
            svg.append(text(x+shift-27,y-10,f'{p:.2f}',11,colors[group]));rows.append(['test',group,seed,6000,run['test']['loss'],p])
        svg.append(text(x-20,731,seed,12))
    svg.extend([text(300,757,'Seed; final checkpoints only; no leaderboard comparison',13),text(30,780,'Context, token handling and training budget are fixed by decision0003. Cache payload is separate from model quality.',12),'</svg>'])
    out.write_text('\n'.join(svg),encoding='utf-8');ET.parse(out)
    with cp.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['split','group','seed','step','loss','perplexity']);writer.writerows(rows)
    manifest=dict(report_sha256=digest(rp),audit_sha256=digest(ap),source_sha256=digest(Path(__file__)),
                  figure_sha256=digest(out),csv_sha256=digest(cp),rows=len(rows))
    mp.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
