"""Extend comparison to ordered lists without modifying executed evidence."""
import argparse
import importlib.util
import json
import unittest

import audit_memory_first_moment_reset as original

study=original.study
SCALAR_COMPARE=original.prior.compare
SOURCES=('experiments/recover_memory_first_moment_audit.py','tests/test_memory_first_moment_audit_recovery.py')
PROTOCOL='docs/protocols/memory-first-moment-audit-recovery.md'


def compare(actual,expected):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(), 'Dictionary coverage differs'
        return max((compare(actual[k],expected[k]) for k in expected),default=0.)
    if isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected), 'List coverage differs'
        return max((compare(a,b) for a,b in zip(actual,expected)),default=0.)
    return SCALAR_COMPARE(actual,expected)


def preflight():
    spec=importlib.util.spec_from_file_location('test_memory_first_moment_audit_recovery',study.ROOT/SOURCES[-1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    record=dict(passed=result.wasSuccessful(),methods=result.testsRun,sources={n:study.training.replay.digest(study.ROOT/n) for n in SOURCES})
    p=study.ROOT/'results/memory_first_moment_audit_recovery_preflight.json';assert not p.exists();study.training.replay.write(p,record);assert record['passed']


def run():
    out=study.OUT;replay=study.training.replay;p=study.ROOT/'results/memory_first_moment_audit_recovery_preflight.json'
    ready=json.loads(p.read_text());assert ready['passed'] and ready['sources']=={n:replay.digest(study.ROOT/n) for n in SOURCES}
    failure=json.loads((out/'audit_failure.json').read_text());assert failure['report_sha256']==replay.digest(out/'report.json')
    assert failure['auditor_sha256']==replay.digest(study.ROOT/study.SOURCES[1])
    files={n:replay.digest(study.ROOT/n) for n in (*SOURCES,PROTOCOL,p.relative_to(study.ROOT).as_posix())}
    replay.check_files(files)
    previous=original.prior.compare;original.prior.compare=compare
    try:original.main()
    finally:original.prior.compare=previous
    replay.check_files(files);assert failure['report_sha256']==replay.digest(out/'report.json')
    record=dict(passed=True,audit_sha256=replay.digest(out/'audit.json'),report_sha256=failure['report_sha256'],
        audit_failure_sha256=replay.digest(out/'audit_failure.json'),files=files,training_rerun=False,tolerance_changed=False)
    p=out/'audit_recovery.json';assert not p.exists();replay.write(p,record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    if p.parse_args().preflight:preflight()
    else:run()
