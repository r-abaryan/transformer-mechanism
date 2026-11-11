"""Export paired per-seed length-transfer results as a standalone SVG."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/length_exposure.json'; r=json.loads(rp.read_text()); assert r['phase']=='complete'
    rows=[dict(seed=x['seed'],group=x['group'],pairs=int(n),accuracy=v['accuracy'],loss=v['loss']) for x in r['runs'] for n,v in x['test'].items()]
    cp=ROOT/'results/length_exposure_plot_data.csv'
    with cp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="460" viewBox="0 0 960 460"><title>Paired six-pair transfer accuracy</title><desc>Each line joins baseline and mixed-length training for the same seed. Horizontal line is the predeclared 90 percent candidate criterion.</desc><rect width="960" height="460" fill="white"/>']
    def text(x,y,s,size=14): svg.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" fill="#243449">{s}</text>')
    text(30,30,'Six-pair transfer: paired seeds, fixed architecture',22)
    text(30,55,'Training only on 3 pairs versus alternating 3/4 pairs. No six-pair training.')
    Y=lambda v:360-260*v
    for a in (0,.25,.5,.75,.9,1):
        yy=Y(a); svg.append(f'<path d="M90 {yy}H880" stroke="{"#b3261e" if a==.9 else "#dce2e8"}" stroke-dasharray="4 4"/>'); text(30,yy+5,f'{a:.0%}',12)
    colors=['#0072b2','#009e73','#d55e00','#8b47a8','#6d4c41']
    for i,(seed,color) in enumerate(zip((761,787,809,839,863),colors)):
        x0=130+150*i; x1=x0+60
        b=next(v['accuracy'] for v in rows if v['seed']==seed and v['group']=='baseline' and v['pairs']==6)
        m=next(v['accuracy'] for v in rows if v['seed']==seed and v['group']=='mixed' and v['pairs']==6)
        svg.append(f'<path d="M{x0} {Y(b)}L{x1} {Y(m)}" stroke="{color}" stroke-width="2"/>')
        for x,v in ((x0,b),(x1,m)):
            svg.append(f'<circle cx="{x}" cy="{Y(v)}" r="5" fill="{color}"/>'); text(x-23,Y(v)-12,f'{100*v:.1f}%',12)
        text(x0-12,385,'3 only',12); text(x1-9,385,'3/4',12); text(x0+10,408,seed,12)
    text(30,442,'Red dashed line = 90% criterion. Equal updates/examples; mixed training uses more tokens and compute.',13)
    svg.append('</svg>'); fp=ROOT/'docs/figures/length-exposure.svg'; fp.write_text('\n'.join(svg)); ET.parse(fp)
    manifest=dict(report_sha256=digest(rp),source_sha256=digest(Path(__file__)),files={str(p.relative_to(ROOT)):digest(p) for p in (cp,fp)})
    (ROOT/'results/length_exposure_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Paired SVG and plot data saved; XML parsed.')


if __name__=='__main__': main()
