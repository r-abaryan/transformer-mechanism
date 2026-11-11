"""Dependency-free SVG plots of recorded gradients; no invented measurements."""
import csv
import html
import json
import math
import statistics
from pathlib import Path
import xml.etree.ElementTree as ET
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'results/gradient_flow.json'; report=json.loads(source.read_text()); assert report['phase']=='complete'
    folder=ROOT/'docs/figures'; folder.mkdir(exist_ok=True)
    steps=[0,500,1500,3000]; runs=report['runs']; colors=['#0072b2','#d55e00','#009e73']
    nodes=['input','b1_after_ffn','after_final_ln']; labels=['Block 1 input','After block 1','After final LayerNorm']
    out=folder/'gradient-flow-training.svg'; heat=folder/'gradient-flow-ratios.svg'; csvpath=ROOT/'results/gradient_flow_plot_data.csv'
    assert not out.exists() and not heat.exists() and not csvpath.exists()
    def text(x,y,t,size=15,fill='#243449',anchor='start'):
        return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(t))}</text>'
    def start(w,h,title,desc): return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{html.escape(title)}</title><desc>{html.escape(desc)}</desc><rect width="100%" height="100%" fill="white"/>']
    svg=start(1080,560,'Activation gradients during training','Fixed 128-example validation probe at offset +4. Thin lines are five seeds; thick lines are arithmetic means. RMS activation gradients, logarithmic vertical axis.')
    svg+=[text(35,34,'Measured gradient flow during training',24),text(35,60,'Fixed 128-example +4 probe; thin = individual seed, thick = five-seed mean',14)]
    lo=math.floor(math.log10(min(s['gradient_rms'][n] for r in runs for s in r['snapshots'] for n in nodes)))
    hi=math.ceil(math.log10(max(s['gradient_rms'][n] for r in runs for s in r['snapshots'] for n in nodes)))
    X=lambda step:100+step/3000*890
    Y=lambda value:435-(math.log10(value)-lo)/(hi-lo)*320
    for exponent in range(lo,hi+1):
        yy=Y(10**exponent); svg.append(f'<path d="M100 {yy}H990" stroke="#dce2e8"/>'); svg.append(text(87,yy+5,f'1e{exponent}',13,anchor='end'))
    for step in steps: svg.append(text(X(step),460,step,14,anchor='middle'))
    svg+=[text(540,490,'Optimizer updates',16,anchor='middle'),text(100,95,'RMS of d(mean cross-entropy)/d(activation)',14),text(35,530,'Smaller gradients can accompany a smaller loss. These curves do not establish vanishing-gradient failure.',14)]
    rows=[]
    for node,label,color in zip(nodes,labels,colors):
        for run in runs:
            points=' '.join(f'{X(s["step"]):.2f},{Y(s["gradient_rms"][node]):.2f}' for s in run['snapshots'])
            svg.append(f'<polyline points="{points}" stroke="{color}" stroke-opacity=".22" fill="none" stroke-width="1.5"/>')
        means=[statistics.mean(r['snapshots'][i]['gradient_rms'][node] for r in runs) for i in range(4)]
        svg.append(f'<polyline points="{" ".join(f"{X(t):.2f},{Y(v):.2f}" for t,v in zip(steps,means))}" stroke="{color}" fill="none" stroke-width="3"/>')
        for step,v in zip(steps,means):
            svg.append(f'<circle cx="{X(step)}" cy="{Y(v)}" r="4" fill="{color}"><title>{label}, step {step}: {v:.8g}</title></circle>'); rows.append(['gradient_rms',node,step,v])
        j=nodes.index(node); svg.append(text(390+j*225,82,label,13,color))
    svg.append('</svg>'); out.write_text('\n'.join(svg),encoding='utf-8'); ET.parse(out)
    sites=[('residuals',n) for n in ('b1_attention','b1_ffn','b2_attention','b2_ffn')]+[('normalizations',n) for n in ('b1_attention_ln','b1_ffn_ln','b2_attention_ln','b2_ffn_ln','final_ln')]
    svg=start(1080,720,'Local gradient norm ratios','Five-seed means. Residual ratios compare total with skip gradients. LayerNorm ratios compare its local input gradient with output gradient. These are different quantities, not ablated training outcomes.')
    svg+=[text(35,35,'Where local gradient norms grow or shrink',24),text(35,65,'Five-seed means; green > 1, orange < 1. Local ratios, not parameter-update sizes.',14)]
    for j,step in enumerate(steps): svg.append(text(460+j*150,103,f'Step {step}',15,anchor='middle'))
    for i,(group,site) in enumerate(sites):
        yy=125+i*53; label=site.replace('_',' ')
        svg.append(text(35,yy+29,label,16))
        for j,step in enumerate(steps):
            key='total_over_skip' if group=='residuals' else 'input_over_output'
            value=statistics.mean(r['snapshots'][j][group][site][key] for r in runs); fill='#d4eee4' if value>=1 else '#fae1c9'
            svg.append(f'<rect x="{390+j*150}" y="{yy}" width="140" height="43" rx="4" fill="{fill}"/>'); svg.append(text(460+j*150,yy+28,f'{value:.3f}',16,anchor='middle'))
            rows.append([group,site,step,value])
    svg+=[text(35,640,'Residual rows: ||skip + branch|| / ||skip||. LayerNorm rows: ||local input gradient|| / ||output gradient||.',14),text(35,672,'Identity skips can still cancel with branch gradients. LayerNorm is not necessarily a gradient amplifier.',14),'</svg>']
    heat.write_text('\n'.join(svg),encoding='utf-8'); ET.parse(heat)
    with csvpath.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['quantity','site','step','five_seed_mean']); writer.writerows(rows)
    manifest=dict(report_sha256=digest(source),plot_source_sha256=digest(Path(__file__)),files={str(p.relative_to(ROOT)):digest(p) for p in (out,heat,csvpath)})
    (ROOT/'results/gradient_flow_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__': main()
