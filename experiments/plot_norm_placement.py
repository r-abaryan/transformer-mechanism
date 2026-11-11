"""Generate auditable SVG learning/gradient curves from factorial results."""
import csv
import html
import json
import math
import statistics as st
from pathlib import Path
import xml.etree.ElementTree as ET
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/norm_placement.json'; r=json.loads(rp.read_text()); assert r['phase']=='complete'
    out=ROOT/'docs/figures/norm-placement.svg'; cp=ROOT/'results/norm_placement_plot_data.csv'; assert not out.exists() and not cp.exists()
    groups=[(p,w) for p in ('pre','post') for w in ('constant','warmup')]; colors=['#0072b2','#009e73','#d55e00','#8b47a8']
    rows=[]
    def txt(x,y,t,size=14,color='#243449'): return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(str(t))}</text>'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="780" viewBox="0 0 1120 780" role="img"><title>Normalization placement and warmup</title><desc>Thin curves are individual seeds; thick curves are five-seed arithmetic means. Validation accuracy and fixed-probe block-2 parameter gradient RMS.</desc><rect width="1120" height="780" fill="white"/>',txt(30,32,'Normalization placement x warmup',24),txt(30,58,'Thin = seed; thick = mean. Validation offset 0. No model selected from these curves.')]
    for j,((p,w),color) in enumerate(zip(groups,colors)): svg.append(txt(55+j*260,87,p+' / '+w,14,color))
    for panel in range(2):
        top=130+panel*320; X=lambda t:100+t/3000*940
        if panel==0:
            values=[v['accuracy'] for a in r['runs'] for v in a['curve']]; lo,hi=0.,1.
            Y=lambda v:top+220-v*220
            ticks=[0,.25,.5,.75,1]; label='Validation accuracy'
        else:
            values=[v['parameter_gradient_rms']['blocks.1'] for a in r['runs'] for v in a['probes']]
            lo=math.floor(math.log10(min(values))); hi=math.ceil(math.log10(max(values)))
            Y=lambda v:top+220-(math.log10(v)-lo)/(hi-lo)*220
            ticks=[10**i for i in range(lo,hi+1)]; label='Block 2 parameter gradient RMS (fixed 128-example probe; log scale)'
        svg.append(txt(40,top-15,label,16))
        for v in ticks:
            yy=Y(v); svg.append(f'<path d="M100 {yy}H1040" stroke="#dce2e8"/>'); svg.append(txt(25,yy+5,f'{v:.0%}' if panel==0 else f'{v:.0e}',12))
        for t in (0,500,1500,3000): svg.append(txt(X(t)-15,top+244,t,12))
        for (p,w),color in zip(groups,colors):
            selected=[a for a in r['runs'] if a['placement']==p and a['schedule']==w]
            def points(a): return [(v['step'],v['accuracy'] if panel==0 else v['parameter_gradient_rms']['blocks.1']) for v in a['curve' if panel==0 else 'probes']]
            for a in selected:
                svg.append(f'<polyline points="{" ".join(f"{X(t):.2f},{Y(v):.2f}" for t,v in points(a))}" fill="none" stroke="{color}" stroke-opacity=".22" stroke-width="1.3"/>')
            times=sorted(set(t for a in selected for t,v in points(a)))
            means=[(t,st.mean(v for a in selected for step,v in points(a) if step==t)) for t in times]
            svg.append(f'<polyline points="{" ".join(f"{X(t):.2f},{Y(v):.2f}" for t,v in means)}" fill="none" stroke="{color}" stroke-width="3"/>')
            for t,v in means:
                rows.append([label,p,w,t,v]); svg.append(f'<circle cx="{X(t)}" cy="{Y(v)}" r="3" fill="{color}"><title>{p} {w}, update {t}: {v:.8g}</title></circle>')
    svg.extend([txt(450,740,'Optimizer updates',15),txt(30,769,'Gradient size alone is not training quality. Lines connect sampled checkpoints; intermediate behavior is unmeasured.',13),'</svg>'])
    out.write_text('\n'.join(svg),encoding='utf-8'); ET.parse(out)
    with cp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['quantity','placement','schedule','step','mean']); w.writerows(rows)
    manifest=dict(report_sha256=digest(rp),source_sha256=digest(Path(__file__)),files={str(p.relative_to(ROOT)):digest(p) for p in (out,cp)})
    (ROOT/'results/norm_placement_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest))


if __name__=='__main__': main()
