"""Plot recorded validation accuracy and loss; preserve individual seeds."""
import csv
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/norm_extension.json'; report=json.loads(rp.read_text())
    assert report['phase']=='complete'
    rows=[dict(seed=r['seed'],placement=r['placement'],schedule=r['schedule'],**v) for r in report['runs'] for v in r['curve']]
    cp=ROOT/'results/norm_extension_plot_data.csv'
    with cp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="730" viewBox="0 0 1100 730"><title>Fixed normalization training extension</title><desc>Validation accuracy and log-scale loss from 3000 to 6000 updates. Thin lines show each seed; thick lines show arithmetic means.</desc><rect width="1100" height="730" fill="white"/>']
    def text(x,y,t,size=14,color='#243449'):
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" fill="{color}">{t}</text>')
    text(30,30,'Training-budget extension: all twenty runs',24)
    text(30,57,'Existing validation set, offset 0. Thin = seed; thick = five-seed mean.')
    colors=['#0072b2','#009e73','#d55e00','#8b47a8']; groups=[(p,w) for p in ('pre','post') for w in ('constant','warmup')]
    for j,((p,w),color) in enumerate(zip(groups,colors)): text(55+260*j,83,p+' / '+w,14,color)
    for panel,quantity in enumerate(('accuracy','loss')):
        top=120+panel*290; X=lambda t:90+(t-3000)/3000*940
        low=math.floor(math.log10(min(r['loss'] for r in rows)))
        high=math.ceil(math.log10(max(r['loss'] for r in rows)))
        Y=(lambda v:top+200-200*v) if quantity=='accuracy' else (lambda v:top+200-200*(math.log10(v)-low)/(high-low))
        text(30,top-12,'Validation accuracy' if panel==0 else 'Validation cross-entropy loss (nats, logarithmic axis)',16)
        ticks=[0,.25,.5,.75,1] if panel==0 else [10**i for i in range(low,high+1)]
        for v in ticks:
            svg.append(f'<path d="M90 {Y(v)}H1030" stroke="#dce2e8"/>'); text(20,Y(v)+5,f'{v:.0%}' if panel==0 else f'{v:.0e}',12)
        for t in range(3000,6001,500): text(X(t)-16,top+223,t,12)
        for (p,w),color in zip(groups,colors):
            selected=[r for r in report['runs'] if r['placement']==p and r['schedule']==w]
            series=[[(v['step'],v[quantity]) for v in r['curve']] for r in selected]
            means=[(t,statistics.mean(v for s in series for step,v in s if step==t)) for t in sorted({t for s in series for t,v in s})]
            for index,s in enumerate(series+[means]):
                points=' '.join(f'{X(t):.2f},{Y(v):.2f}' for t,v in s)
                svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{3 if index==len(series) else 1}" stroke-opacity="{1 if index==len(series) else .25}"/>')
    text(470,677,'Optimizer updates'); text(30,707,'Lines connect sampled checkpoints. Intermediate behavior is unmeasured; test examples are separate.',13)
    svg.append('</svg>'); fp=ROOT/'docs/figures/norm-extension.svg'; fp.write_text('\n'.join(svg)); ET.parse(fp)
    manifest=dict(report_sha256=digest(rp),source_sha256=digest(Path(__file__)),files={str(p.relative_to(ROOT)):digest(p) for p in (cp,fp)})
    (ROOT/'results/norm_extension_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SVG XML and data export complete.')


if __name__=='__main__': main()
