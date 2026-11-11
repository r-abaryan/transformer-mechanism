"""Value-source competition with fixed compact final Q, both directions."""
import itertools
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from activation_paths import forward
from compare_depth import StackedDecoder
from qkv_paths import project,attention
from position_distractors import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=ROOT/'results/target_distractor_keys.json'
    assert not output.exists()
    priorpath=ROOT/'results/head_token_paths.json'
    prior=json.loads(priorpath.read_text())
    splitpath=ROOT/'results/subset_confirmation_splits.json'
    assert digest(splitpath)==prior['split_sha256']
    crops,labels=[],[]
    for ids,target in json.loads(splitpath.read_text())['six_pairs']:
        q=ids[:-2:2].index(ids[-1])
        for others in itertools.combinations([i for i in range(6) if i!=q],2):
            crop=[ids[p] for i in sorted((q,*others)) for p in (2*i,2*i+1)]+ids[-2:]
            assert dict(zip(crop[:-2:2],crop[1:-2:2]))[crop[-1]]==target
            crops.append(crop); labels.append(target)
    x=torch.tensor(crops); y=torch.tensor(labels)
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/target-distractor-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('target_distractor_keys.py','head_token_paths.py','qkv_paths.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(path,weights_only=True))
        pieces={f'{d}_{g}':[] for d in ('forward','reverse') for g in ('none','target','others','all')}
        routing={n:[] for n in pieces}; compact_scores=[]
        with torch.no_grad():
            for chunk in x.split(256):
                c,cc=forward(model,chunk,0); _,sc=forward(model,chunk,4)
                compact_scores.append(c)
                compact=project(model,cc['r1']); shifted=project(model,sc['r1'])
                target=(chunk[:,:6:2]==chunk[:,-1,None]).long().argmax(1)*2+1
                targetmask=torch.arange(8)[None,:]==target[:,None]
                values=torch.zeros_like(targetmask); values[:,[1,3,5]]=True
                others=values&~targetmask
                for direction,recipient,donor,shift in [('forward',shifted,compact,4),('reverse',compact,shifted,0)]:
                    for group,mask in [('none',torch.zeros_like(values)),('target',targetmask),('others',others),('all',values)]:
                        name=f'{direction}_{group}'
                        q,k,v=[t.clone() for t in recipient]
                        q[:,7]=compact[0][:,7]
                        k=torch.where(mask[:,:,None],donor[1],k)
                        a=attention(model,q,k,v)
                        s=forward(model,chunk,shift,{'a1':a},('a1',))[0]
                        pieces[name].append(s)
                        qh=q[:,7].reshape(-1,2,8)
                        kh=k.reshape(-1,8,2,8).transpose(1,2)
                        scores=(qh[:,:,None,:]*kh).sum(-1)/math.sqrt(8)
                        probs=scores.softmax(-1)
                        assert (probs.sum(-1)-1).abs().max()<1e-12
                        index=target[:,None,None].expand(-1,2,1)
                        targetscore=scores.gather(-1,index).squeeze(-1)
                        othermax=scores.masked_fill(~others[:,None,:],float('-inf')).max(-1).values
                        routing[name].append(torch.stack((probs.gather(-1,index).squeeze(-1),
                            (probs*others[:,None,:]).sum(-1),targetscore-othermax),-1))
        scores={n:torch.cat(v) for n,v in pieces.items()}
        conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            reference=name.split('_')[0]+'_none'
            before=scores[reference].argmax(-1)==y; now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                rescued=int((now&~before).sum()),harmed=int((~now&before).sum()),reference=reference,
                routing_per_head=torch.cat(routing[name]).mean(0).tolist())
        for name,old in [('forward_none','Q_final'),('forward_target','target_value'),('forward_all','value_slots')]:
            for metric in ('accuracy','loss'): assert abs(conditions[name][metric]-run['conditions'][old][metric])<1e-10
        error=(scores['reverse_none']-torch.cat(compact_scores)).abs().max().item(); assert error<1e-10
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),reverse_sham_error=error,conditions=conditions))
    report['forward_all_seeds_90']={g:all(r['conditions'][f'forward_{g}']['accuracy']>=.9 for r in report['runs']) for g in ('none','target','others','all')}
    report['reverse_drop_10_every_seed']={g:all(r['conditions']['reverse_none']['accuracy']-r['conditions'][f'reverse_{g}']['accuracy']>=.1 for r in report['runs']) for g in ('target','others','all')}
    report['routing_columns']=['target_probability','other_value_probability_sum','target_minus_max_other_value_score']
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('forward_all_seeds_90','reverse_drop_10_every_seed')}))


if __name__=='__main__':
    main()
