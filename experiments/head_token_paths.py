"""Localize matched routing replacement by head and Q/K position."""
import itertools
import json
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
    output=ROOT/'results/head_token_paths.json'
    assert not output.exists()
    priorpath=ROOT/'results/qkv_paths.json'
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
    # Each spec is head feature slice, Q rows and K source selector.
    specs={'head0':(slice(0,8),'final','all'),'head1':(slice(8,16),'final','all'),
           'both':(slice(None),'final','all'),'Q_final':(slice(None),'final','none'),
           'QK_all':(slice(None),'all','all'),'Q_nonfinal_K_all':(slice(None),'nonfinal','all'),
           'K_all':(slice(None),'none','all')}
    for group in ('key_slots','value_slots','tail','target_key','target_value','pairs'):
        specs[group]=(slice(None),'final',group)
    report=dict(torch=torch.__version__,device='cpu',dtype='float64',prior_sha256=digest(priorpath),split_sha256=digest(splitpath),
        protocol_sha256=digest(ROOT/'docs/head-token-protocol.md'),
        source_sha256={n:digest(ROOT/'experiments'/n) for n in ('head_token_paths.py','qkv_paths.py','activation_paths.py','compare_depth.py','position_distractors.py')},runs=[])
    for run in prior['runs']:
        path=ROOT/run['checkpoint']; assert digest(path)==run['checkpoint_sha256']
        model=StackedDecoder(16,32,2).double().eval()
        model.load_state_dict(torch.load(path,weights_only=True))
        pieces={name:[] for name in specs}; originals=[]
        with torch.no_grad():
            for chunk in x.split(256):
                _,compact=forward(model,chunk,0)
                baseline,shifted=forward(model,chunk,4)
                originals.append(baseline)
                donor=project(model,compact['r1']); recipient=project(model,shifted['r1'])
                target_key=(chunk[:,:6:2]==chunk[:,-1,None]).long().argmax(1)*2
                for name,(features,qrows,group) in specs.items():
                    q,k,v=[t.clone() for t in recipient]
                    qi={'final':[7],'all':list(range(8)),'nonfinal':list(range(7)),'none':[]}[qrows]
                    q[:,qi,features]=donor[0][:,qi,features]
                    fixed={'all':list(range(8)),'none':[],'key_slots':[0,2,4],'value_slots':[1,3,5],'tail':[6,7],'pairs':list(range(6))}
                    if group in fixed:
                        ki=fixed[group]; k[:,ki,features]=donor[1][:,ki,features]
                    else:
                        ki=target_key+(group=='target_value')
                        batch=torch.arange(len(chunk))
                        k[batch,ki,features]=donor[1][batch,ki,features]
                    a=attention(model,q,k,v)
                    pieces[name].append(forward(model,chunk,4,{'a1':a},('a1',))[0])
        scores={n:torch.cat(s) for n,s in pieces.items()}
        baseline=torch.cat(originals); correct=baseline.argmax(-1)==y
        errors={}
        for a,b in [('both','QK_all'),('Q_nonfinal_K_all','K_all')]:
            errors[f'{a}={b}']=(scores[a]-scores[b]).abs().max().item()
            assert errors[f'{a}={b}']<1e-10
        conditions={}
        for name,s in scores.items():
            assert torch.isfinite(s).all()
            now=s.argmax(-1)==y
            conditions[name]=dict(accuracy=now.double().mean().item(),loss=F.cross_entropy(s,y).item(),
                                 rescued=int((now&~correct).sum()),harmed=int((~now&correct).sum()))
        for name,old in [('both','matched_QK'),('K_all','matched_K'),('Q_final','matched_Q')]:
            for metric in ('accuracy','loss'): assert abs(conditions[name][metric]-run['conditions'][old][metric])<1e-10
        assert digest(path)==run['checkpoint_sha256']
        report['runs'].append(dict(seed=run['seed'],checkpoint=run['checkpoint'],checkpoint_sha256=digest(path),errors=errors,
                                  baseline_accuracy=correct.double().mean().item(),conditions=conditions))
    report['all_seeds_accuracy_90']={n:all(r['conditions'][n]['accuracy']>=.9 for r in report['runs']) for n in specs}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['all_seeds_accuracy_90']))


if __name__=='__main__':
    main()
