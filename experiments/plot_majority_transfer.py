"""Audited per-seed aggregation results as a standalone SVG plus source CSV."""
import csv
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/majority_transfer.json'; ap=ROOT/'results/majority_transfer_audit.json'
    r=json.loads(rp.read_text()); audit=json.loads(ap.read_text())
    assert r['phase']=='complete' and audit['passed'] and digest(rp)==audit['report_sha256']
    out=ROOT/'docs/figures/majority-transfer.svg'; cp=ROOT/'results/majority_transfer_plot_data.csv'; mp=ROOT/'results/majority_transfer_plot_manifest.json'
    assert not any(p.exists() for p in (out,cp,mp))
    colors=('#0072b2','#d55e00','#009e73','#cc79a7','#5b4b8a'); lengths=(9,11,15,21); rows=[]
    def text(x,y,value,size=14,color='#243449'):
        return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(str(value))}</text>'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="550" viewBox="0 0 1100 550" role="img">',
         '<title>Majority task transfer: five seeds, overall and one-bit-majority accuracy</title>',
         '<desc>Per-seed held-out accuracy. Both vertical axes run from zero to one hundred percent. Train lengths are nine and eleven bits; longer lengths are fifteen and twenty-one. Dashed segments show fixed pass floors.</desc>',
         '<rect width="1100" height="550" fill="white"/>',text(25,32,'Majority: aggregation beyond recall',24),
         text(25,60,'Fixed MQA/RoPE/ReLU/LayerNorm recipe; five fresh seeds; 6,000 updates; CPU float64.')]
    for i,run in enumerate(r['runs']): svg.append(text(30+210*i,88,f'Seed {run["seed"]}',14,colors[i]))
    for panel,metric,title in ((0,'accuracy','All held-out strings'),(1,'boundary_accuracy','Majority by one bit only')):
        left=75+panel*535; top=145; width=440; height=300
        xx=lambda n:left+width*(n-9)/12
        yy=lambda value:top+height*(1-value)
        svg.append(text(left,top-20,title,18))
        for tick in (0,.25,.5,.75,1):
            y=yy(tick); svg.append(f'<path d="M{left} {y}h{width}" stroke="#dce2e8"/>'); svg.append(text(left-45,y+4,f'{tick:.0%}',12))
        for n in lengths:
            svg.append(text(xx(n)-9,475,n,13)); svg.append(text(xx(n)-18,496,'train' if n<15 else 'long',11))
        if metric=='accuracy':
            svg.append(f'<path d="M{xx(9)} {yy(.99)}H{xx(11)}" stroke="#777" stroke-dasharray="5 4"/>')
        svg.append(f'<path d="M{xx(15)} {yy(.90)}H{xx(21)}" stroke="#777" stroke-dasharray="5 4"/>')
        for i,run in enumerate(r['runs']):
            points=[]
            for n in lengths:
                value=run['test'][str(n)][metric]; points.append((xx(n),yy(value)))
                rows.append([metric,run['seed'],n,value])
                svg.append(f'<circle cx="{xx(n):.3f}" cy="{yy(value):.3f}" r="3" fill="{colors[i]}"><title>{run["seed"]}, {n} bits: {value:.8%}</title></circle>')
            svg.append(f'<polyline points="{" ".join(f"{x:.3f},{y:.3f}" for x,y in points)}" fill="none" stroke="{colors[i]}" stroke-width="1.5" stroke-opacity=".8"/>')
    svg.extend([text(25,530,'Lines connect the same seed; no confidence intervals. Bit counts exclude the final query marker. Dashed = pass floor.',12),'</svg>'])
    out.write_text('\n'.join(svg),encoding='utf-8'); ET.parse(out)
    with cp.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['metric','seed','bits','accuracy']); writer.writerows(rows)
    manifest=dict(report_sha256=digest(rp),audit_sha256=digest(ap),source_sha256=digest(Path(__file__)),
                  figure_sha256=digest(out),csv_sha256=digest(cp),rows=len(rows))
    mp.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest))


if __name__=='__main__': main()
