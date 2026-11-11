"""Independent label and saved-prediction audit for key swaps."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/key_content.json'; r=json.loads(rp.read_text())
    assert digest(ROOT/'experiments/key_content.py')==r['source_sha256']
    assert digest(ROOT/'docs/key-content-protocol.md')==r['protocol_sha256']
    assert digest(ROOT/'results/head_ablation.json')==r['prior_sha256']
    ip=ROOT/'results/key_content_inputs.json'; assert digest(ip)==r['input_sha256']
    d=json.loads(ip.read_text()); y=torch.tensor(d['original_labels']); cy=torch.tensor(d['changed_labels'])
    for i,(a,b,c) in enumerate(zip(d['recipient'],d['changed'],d['control'])):
        for seq,label in ((a,y[i]),(b,cy[i]),(c,y[i])):
            assert dict(zip(seq[:6:2],seq[1:6:2]))[seq[-1]]==label.item()
        assert a[1::2]==b[1::2]==c[1::2] and a[-2:]==b[-2:]==c[-2:]
        assert sorted(d['rows'][i])==sorted(j+1 for j in (0,2,4) if a[j]!=b[j])
        assert sorted(d['control_rows'][i])==sorted(j+1 for j in (0,2,4) if a[j]!=c[j])
    sp=ROOT/'results/key_content_scores.pt'; assert digest(sp)==r['scores_sha256']
    saved=torch.load(sp,weights_only=True); count=0
    assert len(saved)==len(r['runs'])==5
    for run,detail in zip(r['runs'],saved):
        assert run['seed']==detail['seed'] and digest(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        scores=detail['scores']; mask=(scores['recipient'].argmax(-1)==y)&(scores['changed_donor'].argmax(-1)==cy)
        assert len(scores)==7
        for name,s in scores.items():
            pred=s.argmax(-1); m=run['conditions'][name]
            assert m['eligible_count']==int(mask.sum())
            for k,v in dict(original_accuracy=(pred==y).double().mean().item(),changed_accuracy=(pred==cy).double().mean().item(),original_loss=F.cross_entropy(s,y).item(),changed_loss=F.cross_entropy(s,cy).item(),eligible_changed=(pred[mask]==cy[mask]).double().mean().item()).items(): assert m[k]==v
            count+=1
    result=dict(passed=True,conditions=count,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/key_content_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
