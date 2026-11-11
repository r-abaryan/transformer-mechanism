"""Alternative probability construction and final-block replay on saved evidence."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from compare_depth import StackedDecoder
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/trained_competition.json'; r=json.loads(rp.read_text()); assert r['complete']
    for p,h in r['hashes'].items(): assert digest(ROOT/p)==h
    assert digest(ROOT/'results/depth_confirmation.json')==r['prior_sha256']
    assert digest(ROOT/r['split_file'])==r['split_sha256']
    tp=ROOT/'results/trained_competition_tensors.pt'; assert digest(tp)==r['tensors_sha256']
    data=torch.load(tp,weights_only=True); y=data['labels']; x=data['ids']; target=data['target']; wrong=data['wrong']
    rows=json.loads((ROOT/r['split_file']).read_text())['six_pairs']
    assert x.tolist()==[a for a,b in rows] and y.tolist()==[b for a,b in rows]
    rank=(x[:,:-2:2]==x[:,-1,None]).long().argmax(-1)
    assert torch.equal(target,2*rank+1) and torch.equal(wrong,2*((rank+1)%6)+1)
    assert int((x.gather(1,wrong[:,None]).squeeze(1)==y).sum())==r['control_same_answer']
    assert [a['seed'] for a in r['runs']]==[137,163,191,223,257]
    max_error=0.; summaries={}; strata={}; gains=[]
    with torch.no_grad():
        for run,part in zip(r['runs'],data['runs'],strict=True):
            assert run['seed']==part['seed']
            cp=ROOT/run['checkpoint']; assert digest(cp)==run['checkpoint_sha256']
            model=StackedDecoder(16,32,2).double().eval(); model.load_state_dict(torch.load(cp,weights_only=True))
            block=model.blocks[1]; scores=part['scores']; v=part['values']; residual=part['residual']
            base=part['outputs']['ordinary']; before=base.argmax(-1)==y
            top=(scores.argmax(-1)==target[:,None]).all(-1)
            assert set(part['outputs'])=={'ordinary','sharpen2','target_floor','wrong_slot_floor'}
            a=scores.softmax(-1)
            assert torch.allclose(a.gather(-1,target[:,None,None].expand(-1,2,1)).squeeze(-1).mean(0),torch.tensor(run['target_mass_by_head'],dtype=torch.float64),atol=1e-12,rtol=0)
            for name,original in part['outputs'].items():
                if name=='sharpen2':
                    weights=a.square(); weights=weights/weights.sum(-1,keepdim=True)
                elif name.endswith('floor'):
                    index=(wrong if name=='wrong_slot_floor' else target)[:,None,None].expand(-1,2,1)
                    selected=scores.gather(-1,index)
                    others=scores.clone().scatter(-1,index,float('-inf'))
                    # logit giving odds 9:1 against all other positions combined.
                    desired=torch.maximum(selected,others.logsumexp(-1,keepdim=True)+math.log(9))
                    weights=scores.clone().scatter(-1,index,desired).softmax(-1)
                else: weights=a
                out=block.self_attn.out_proj
                # Project heads separately rather than concatenating before W_O.
                heads=(weights[:,:,None]@v).squeeze(-2)
                attn=sum(F.linear(heads[:,h],out.weight[:,h*8:(h+1)*8]) for h in range(2))+out.bias
                u=residual+attn
                z=u+block.linear2(block.activation(block.linear1(block.norm2(u))))
                replay=model.output(model.final_norm(z))
                error=(replay-original).abs().max().item(); max_error=max(max_error,error); assert error<1e-10
                assert torch.isfinite(original).all()
                now=original.argmax(-1)==y
                m=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(original,y).item(),rescued=int((now&~before).sum()),harmed=int((~now&before).sum()))
                for key,value in m.items(): assert abs(value-run['conditions'][name][key])<1e-12
                summaries.setdefault(name,[]).append(m)
                for entry,mask in zip(run['strata'],[top,~top]):
                    assert entry['count']==int(mask.sum()) and entry['original_errors']==int((mask&~before).sum())
                    assert entry['rescued'][name]==int((mask&~before&now).sum())
            for entry in run['strata']:
                bucket=strata.setdefault(entry['name'],dict(count=0,original_errors=0,rescued={name:0 for name in part['outputs']}))
                bucket['count']+=entry['count']; bucket['original_errors']+=entry['original_errors']
                for name,value in entry['rescued'].items(): bucket['rescued'][name]+=value
            gains.append(run['conditions']['sharpen2']['accuracy']-run['conditions']['ordinary']['accuracy'])
    assert r['sharpen2_keep']==(sum(gains)/5>=.05 and min(gains)>0)
    summary={name:dict(mean_accuracy=sum(a['accuracy'] for a in items)/5,mean_loss=sum(a['loss'] for a in items)/5,
                       rescued=sum(a['rescued'] for a in items),harmed=sum(a['harmed'] for a in items)) for name,items in summaries.items()}
    result=dict(passed=True,max_logit_error=max_error,conditions=20,summary=summary,strata=strata,
                report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    (ROOT/'results/trained_competition_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
