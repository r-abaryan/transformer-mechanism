"""Preserve the raw study; resolve discrete CE counts at a roundoff boundary."""
import copy
import decimal
import json
import math

import torch
import audit_readout_scale_control as original

study=original.study


def precise_change(final,control,label,precision=60):
    with decimal.localcontext() as ctx:
        ctx.prec=precision
        def ce(z):
            values=[decimal.Decimal.from_float(float(v)) for v in z];high=max(values)
            return sum((v-high).exp() for v in values).ln()+high-values[label]
        return ce(final)-ce(control)


def reconstruct(old,final,control):
    expected=original.reconstruct(old,final,control);boundary=[]
    for i,(b,c,y) in enumerate(zip(final['logits'].reshape(-1,6).tolist(),control['logits'].reshape(-1,6).tolist(),final['labels'].flatten().tolist())):
        delta=original.prior.scalar(b,y)['ce']-original.prior.scalar(c,y)['ce']
        if abs(abs(delta)-1e-12)>1e-14:continue
        exact=precise_change(b,c,y);threshold=decimal.Decimal('1e-12')
        old_pred=int(old['logits'].reshape(-1,6)[i].argmax());final_pred=b.index(max(b))
        name=('correct_both' if old_pred==y else 'rescued') if final_pred==y else ('newly_wrong' if old_pred==y else 'wrong_both')
        repeat=bool(old['repeat'].flatten()[i])
        for group in ('all','repeat' if repeat else 'unique'):
            p=expected[group]['transitions'][name]
            p['loss_increased']+=int(exact < -threshold)-int(delta < -1e-12)
            p['loss_decreased']+=int(exact > threshold)-int(delta>1e-12)
        boundary.append(dict(flat_index=i,scalar_ce_removed=delta,precise_ce_removed=str(exact),transition=name,repeat=repeat))
    return expected,boundary


def reconcile(actual,expected):
    corrected=copy.deepcopy(actual);changes=[]
    for group in ('all','unique','repeat'):
        for transition in ('correct_both','wrong_both','rescued','newly_wrong'):
            for metric in ('loss_increased','loss_decreased'):
                raw=actual[group]['transitions'][transition][metric];effective=expected[group]['transitions'][transition][metric]
                if raw!=effective:
                    changes.append(dict(group=group,transition=transition,metric=metric,raw=raw,effective=effective))
                    corrected[group]['transitions'][transition][metric]=effective
    error=original.prior.compare(corrected,expected)
    return changes,error


def main():
    torch.set_num_threads(1);out=study.OUT;root=study.ROOT;replay=study.training.replay
    report=json.loads((out/'report.json').read_text());lock=json.loads((out/'lock.json').read_text())
    replay.check_files(lock['files']);assert report['status']=='complete' and report['training_updates']==0
    assert report['lock_sha256']==replay.digest(out/'lock.json')
    assert report['scores_sha256']==replay.digest(out/'scores.pt') and report['scales_sha256']==replay.digest(out/'scales.json')
    assert report['scales']==json.loads((out/'scales.json').read_text())
    pairs,old_report,final_report=study.previous.inputs();scores=torch.load(out/'scores.pt',weights_only=True)
    ids=[(s,f'{split}_{m}') for s in (3461,3463) for split in ('train','validation','test') for m in (2,8,16,32)]
    assert [(c['seed'],c['pool']) for c in report['cells']]==ids==[(c['seed'],c['pool']) for c in scores]
    errors=[];forwards=[];corrections=[];boundaries=[]
    for (a,b,rows),c,record in zip(pairs,scores,report['cells']):
        assert (a['seed'],a['pool'])==(b['seed'],b['pool'])==(c['seed'],c['pool'])==(record['seed'],record['pool'])
        assert record['old_checkpoint']==a['checkpoint'] and record['final_checkpoint']==b['checkpoint']
        assert record['controlled_checkpoint']==c['checkpoint']
        for cell,source in ((a,old_report),(b,final_report)):
            entry=next(x for x in source['cells'] if (x['seed'],x['arm'],x['step'],x['pool'])==(cell['seed'],cell['arm'],cell['step'],cell['pool']))
            assert entry=={k:cell[k] for k in ('seed','arm','step','pool','checkpoint','metrics')}
        cp=torch.load(root/c['checkpoint'],weights_only=True)
        assert cp['step']==12000 and report['checkpoint_hashes'][str(c['seed'])]==replay.digest(root/c['checkpoint'])
        errors.append(original.verify_head(torch.load(root/a['checkpoint'],weights_only=True)['model'],
            torch.load(root/b['checkpoint'],weights_only=True)['model'],cp['model'],report['scales'][str(c['seed'])]))
        _,_,labels,repeat=original.prior.previous.independent.inputs(rows,'all_queries')
        assert torch.equal(a['labels'],labels) and torch.equal(a['repeat'],repeat),'Wrong labels/order'
        expected,near=reconstruct(a,b,c);changes,error=reconcile(record['groups'],expected);errors.append(error)
        # Each changed count must have an observed arithmetic-boundary justification.
        assert not changes or near,'Unjustified count correction'
        corrections.extend(dict(seed=a['seed'],pool=a['pool'],**v) for v in changes)
        boundaries.extend(dict(seed=a['seed'],pool=a['pool'],**v) for v in near)
        metrics=original.prior.previous.metrics(c['logits'],labels,repeat)
        original.prior.previous.check_metrics(c['metrics'],metrics)
        assert record['complete_maps']==metrics['complete_maps']==b['metrics']['complete_maps']
        forwards.append(original.prior.previous.audit_cell(root/c['checkpoint'],rows,c)['max_logit_error'])
    assert report['answers']==sum(len(rows)*a['labels'].shape[1] for a,b,rows in pairs)==155904
    assert report['cells_count']==24 and report['max_affine_logit_error']<=1e-9
    assert len(corrections)==2 and all((v['seed'],v['pool'],v['transition'],v['metric'],v['raw'],v['effective'])==
        (3463,'test_8','correct_both','loss_increased',2244,2243) for v in corrections)
    assert {v['group'] for v in corrections}=={'all','repeat'}
    replay.check_files(lock['files'])
    recovery_sources=('experiments/recover_readout_scale_audit.py','docs/protocols/readout-scale-threshold-recovery.md',
        'tests/test_readout_scale_threshold.py')
    preflight=json.loads((out/'recovery_preflight.json').read_text())
    assert preflight['passed'] and preflight['sources']=={p:replay.digest(root/p) for p in recovery_sources}
    overlay=dict(raw_report_sha256=replay.digest(out/'report.json'),corrections=corrections,boundary_cases=boundaries,
        sources=preflight['sources'])
    assert not (out/'threshold-corrections.json').exists();replay.write(out/'threshold-corrections.json',overlay)
    result=dict(passed=True,report_sha256=replay.digest(out/'report.json'),corrections_sha256=replay.digest(out/'threshold-corrections.json'),
        raw_count_audit_passed=False,recovered_count_fields=2,cells=24,answers=155904,training_updates=0,changed_winners=0,
        max_scalar_error=max(errors),max_matrix_logit_error=max(forwards),protected_files=len(lock['files']),
        original_forward_audits_reused_by_hash=True,recovery_sources=preflight['sources'])
    assert not (out/'audit.json').exists();replay.write(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
