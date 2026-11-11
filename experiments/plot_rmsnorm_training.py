"""Standalone SVG and CSV from audited normalization-training measurements."""
import csv
import html
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/rmsnorm_training.json';ap=ROOT/'results/rmsnorm_training_audit.json'
    r=json.loads(rp.read_text());audit=json.loads(ap.read_text())
    assert r['phase']=='complete' and audit['passed'] and digest(rp)==audit['report_sha256']
    out=ROOT/'docs/figures/rmsnorm-training.svg';cp=ROOT/'results/rmsnorm_training_plot_data.csv'
    mp=ROOT/'results/rmsnorm_training_plot_manifest.json';assert not any(p.exists() for p in (out,cp,mp))
    colors={'ln':'#0072b2','rms':'#d55e00'};rows=[]
    def text(x,y,value,size=14,color='#243449'):
        return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(str(value))}</text>'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="780" viewBox="0 0 1120 780" role="img">',
         '<title>LayerNorm versus RMSNorm: matched recall training</title>',
         '<desc>Top: minimum sampled validation accuracy across both query types and training lengths, individual seeds and means. Bottom: six-pair unique-query held-out accuracy by paired seed. Both plots use percent axes from zero to one hundred.</desc>',
         '<rect width="1120" height="780" fill="white"/>',text(30,34,'LayerNorm versus RMSNorm: centering-only comparison',24),
         text(30,61,'Equal 4,464 parameters; gain only; epsilon 1e-5; five paired seeds. CPU float64.'),
         text(55,88,'LayerNorm',14,colors['ln']),text(245,88,'RMSNorm',14,colors['rms'])]
    top=130;left=95;width=950;height=220
    xx=lambda step:left+width*step/6000;yy=lambda value:top+height*(1-value)
    svg.append(text(30,top-15,'Sampled validation: minimum accuracy across 3/4 pairs and repeated/unique queries',16))
    for tick in (0,.25,.5,.75,1):
        y=yy(tick);svg.append(f'<path d="M{left} {y}H1045" stroke="#dce2e8"/>');svg.append(text(36,y+5,f'{tick:.0%}',12))
    threshold=yy(.99);svg.append(f'<path d="M{left} {threshold}H1045" stroke="#777" stroke-dasharray="5 4"/>')
    svg.append(text(1047,threshold+4,'99%',11))
    for step in range(0,6001,1000):svg.append(text(xx(step)-18,top+height+24,step,12))
    for group,color in colors.items():
        runs=[a for a in r['runs'] if a['group']==group]
        for run in runs:
            points=[]
            for point in run['curve']:
                value=min(point['validation'][str(n)][kind+'_accuracy'] for n in (3,4) for kind in ('repeated','unique'))
                points.append((point['step'],value));rows.append(['validation_minimum',group,run['seed'],point['step'],value])
            svg.append(f'<polyline points="{" ".join(f"{xx(t):.3f},{yy(v):.3f}" for t,v in points)}" fill="none" stroke="{color}" stroke-opacity=".25" stroke-width="1.4"/>')
        means=[(step,statistics.mean(row[4] for row in rows if row[0]=='validation_minimum' and row[1]==group and row[3]==step)) for step in range(0,6001,1000)]
        svg.append(f'<polyline points="{" ".join(f"{xx(t):.3f},{yy(v):.3f}" for t,v in means)}" fill="none" stroke="{color}" stroke-width="3"/>')
    svg.append(text(420,402,'Optimizer updates; thin = seed, thick = arithmetic mean',13))
    top=460;yy=lambda value:top+height*(1-value)
    svg.append(text(30,top-35,'Six-pair unique-query test accuracy: paired final checkpoints',16))
    for tick in (0,.25,.5,.75,1):
        y=yy(tick);svg.append(f'<path d="M{left} {y}H1045" stroke="#dce2e8"/>');svg.append(text(36,y+5,f'{tick:.0%}',12))
    seeds=sorted({run['seed'] for run in r['runs']})
    for i,seed in enumerate(seeds):
        x=left+width*(i+.5)/len(seeds);points=[]
        for group,shift in (('ln',-20),('rms',20)):
            run=next(a for a in r['runs'] if a['seed']==seed and a['group']==group)
            value=run['test']['6']['unique_accuracy'];points.append((x+shift,yy(value),group,value))
            rows.append(['test_6_unique',group,seed,6000,value])
        svg.append(f'<path d="M{points[0][0]} {points[0][1]}L{points[1][0]} {points[1][1]}" stroke="#888"/>')
        for px,py,group,value in points:
            svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{colors[group]}"><title>{group}, seed {seed}: {value:.8%}</title></circle>')
            svg.append(text(px-28,py-(22 if group=='ln' else 7),f'{value:.2%}',11,colors[group]))
        svg.append(text(x-22,top+height+25,seed,12))
    svg.extend([text(380,735,'Seed; lines connect matched arms, not confidence intervals',13),
                text(30,767,'Validation points are 1,000 updates apart. Six-pair test mappings were reserved before training.',12),'</svg>'])
    out.write_text('\n'.join(svg),encoding='utf-8');ET.parse(out)
    with cp.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['quantity','group','seed','step','accuracy']);writer.writerows(rows)
    manifest=dict(report_sha256=digest(rp),audit_sha256=digest(ap),source_sha256=digest(Path(__file__)),
                  files={str(p.relative_to(ROOT)):digest(p) for p in (out,cp)},rows=len(rows))
    mp.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
