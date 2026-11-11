"""Standalone final validation/test plot, only from completed audited evidence."""
import csv
import html
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from position_distractors import digest
ROOT=Path(__file__).resolve().parents[1]


def main():
    rp=ROOT/'results/swiglu_training.json';ap=ROOT/'results/swiglu_training_audit.json'
    report=json.loads(rp.read_text());audit=json.loads(ap.read_text())
    assert report['phase']=='complete' and audit['passed'] and audit['report_sha256']==digest(rp)
    out=ROOT/'docs/figures/swiglu-training.svg';cp=ROOT/'results/swiglu_training_plot_data.csv';mp=ROOT/'results/swiglu_training_plot_manifest.json'
    assert not any(p.exists() for p in (out,cp,mp))
    colors={'relu':'#0072b2','swiglu':'#d55e00'};rows=[]
    def text(x,y,t,size=14,color='#243449'):
        return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(str(t))}</text>'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="760" viewBox="0 0 1050 760" role="img">',
         '<title>ReLU and SwiGLU: final paired perplexity</title>',
         '<desc>Two panels show final validation and reused test perplexity. Lower is better. Both FFNs have 1056 weights and projection MACs per token per block. Validation determines adoption; test is descriptive.</desc>',
         '<rect width="1050" height="760" fill="white"/>',
         text(30,32,'ReLU33 versus SwiGLU22: equal FFN projection cost',23),
         text(30,59,'MQA; width16; two blocks; bias-free FFNs; three paired seeds; 6,000 updates; context64.',14),
         text(30,87,'ReLU',14,colors['relu']),text(160,87,'SwiGLU',14,colors['swiglu'])]
    left,width,height=85,900,215
    for split,top,title in (('validation',143,'Final validation: primary decision metric'),('test',457,'Reused test: descriptive, cannot change adoption')):
        key='final_validation' if split=='validation' else 'test'
        baseline=report['unigram_'+split]['perplexity']
        maximum=math.ceil(max([r[key]['perplexity'] for r in report['runs']]+[baseline])/100)*100
        yy=lambda p:top+height*(1-p/maximum)
        svg.append(text(30,top-19,title,17))
        for i in range(6):
            v=maximum*i/5;y=yy(v)
            svg.append(f'<path d="M{left} {y}h{width}" stroke="#dce2e8"/>');svg.append(text(28,y+4,f'{v:.0f}',12))
        svg.append(f'<path d="M{left} {yy(baseline)}h{width}" stroke="#777" stroke-dasharray="5 4"/>')
        svg.append(text(735,yy(baseline)-8,f'Unigram {baseline:.2f}',12))
        for index,seed in enumerate(sorted({r['seed'] for r in report['runs']})):
            x=left+width*(index+.5)/3
            for arm,shift in (('relu',-32),('swiglu',32)):
                run=next(r for r in report['runs'] if (r['seed'],r['arm'])==(seed,arm))
                value=run[key]['perplexity'];y=yy(value)
                svg.append(f'<rect x="{x+shift-24}" y="{y}" width="48" height="{top+height-y}" fill="{colors[arm]}"/>')
                svg.append(text(x+shift-25,y-10,f'{value:.2f}',11,colors[arm]))
                rows.append([split,arm,seed,run[key]['loss'],value])
            svg.append(text(x-20,top+height+25,seed,12))
    svg.extend([text(270,719,'Seed; perplexity = exp(mean next-word loss); lower is better',14),
                text(30,747,'All validation/test token losses independently audited. Projection matching does not establish equal runtime.',13),'</svg>'])
    out.write_text('\n'.join(svg),encoding='utf-8');ET.parse(out)
    with cp.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['split','arm','seed','loss','perplexity']);writer.writerows(rows)
    manifest=dict(report_sha256=digest(rp),audit_sha256=digest(ap),source_sha256=digest(Path(__file__)),
                  figure_sha256=digest(out),csv_sha256=digest(cp),rows=len(rows))
    mp.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
