"""Review harness: isolated fixtures, no training or historical artifact writes."""
import ast
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths={}
    for name in ('copy_marker_training','copy_marker_alignment'):
        report=json.loads((ROOT/'results'/name/'report.json').read_text())
        for p,h in report['source_dependencies'].items():
            assert sha(ROOT/p)==h
            paths[p]=h
        for r in report.get('runs',[]):
            for k in ('checkpoint','scores'):
                assert sha(ROOT/r[k])==r[k+'_sha256'];paths[r[k]]=r[k+'_sha256']
    modules=[];suite=unittest.TestSuite()
    for name in ('test_copy_rope','test_copy_marker','test_copy_marker_alignment'):
        spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.DIRECTORY=OUT/name;mod.DIRECTORY.mkdir(exist_ok=False)
        if name=='test_copy_rope':
            # Retain uniquely named fixtures instead of exercising the known
            # Windows TemporaryDirectory cleanup problem in historical results.
            class RetainedFixture:
                counter=0
                def __init__(self,**kwargs):
                    assert kwargs['dir']==ROOT/'results'
                    self.path=OUT/'test_copy_rope'/f'fixture_{RetainedFixture.counter}'
                    RetainedFixture.counter+=1
                def __enter__(self):
                    self.path.mkdir();return str(self.path)
                def __exit__(self,*args):return False
            mod.tempfile=types.SimpleNamespace(TemporaryDirectory=RetainedFixture)
        modules.append(mod);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()

    # Exercise the original standalone main block with a simulated successful
    # test runner. It must not touch real preflight files. The sentinel gives
    # a concrete overwrite witness, independent of the static code reading.
    probe=OUT/'overwrite_witness';probe.mkdir();target=probe/'preflight.json'
    target.write_text('{"sentinel":"existing evidence"}\n',encoding='utf-8');before=sha(target)
    source=ROOT/'tests/test_copy_marker_alignment.py';tree=ast.parse(source.read_text())
    block=next(node for node in tree.body if isinstance(node,ast.If) and ast.unparse(node.test)=="__name__ == '__main__'")
    fake=types.SimpleNamespace(testsRun=4,wasSuccessful=lambda:True)
    ut=types.SimpleNamespace(defaultTestLoader=types.SimpleNamespace(loadTestsFromTestCase=lambda _:None),
        TextTestRunner=lambda **_:types.SimpleNamespace(run=lambda _:fake))
    def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
    env=dict(__file__=str(source),__name__='__main__',DIRECTORY=probe,unittest=ut,AlignmentTests=object,
        CASES=[{}]*27,SEEDS=(3163,3167,3169),MODES=('native','aligned','shifted'),write=write,
        digest=sha,Path=Path,sys=types.SimpleNamespace(exit=lambda _:None))
    exec(compile(ast.Module(body=block.body,type_ignores=[]),str(source),'exec'),env)
    overwrite=sha(target)!=before;assert overwrite
    assert all(sha(ROOT/p)==h for p,h in paths.items())
    out=dict(passed=True,test_methods=result.testsRun,initialized_equation_gradient_cases=sum(len(getattr(m,'CASES',getattr(m,'ERRORS',[]))) for m in modules),
        original_dependencies_and_checkpoints_unchanged=True,hashes_checked=len(paths),training_updates=0,
        fixture_harness='Isolated review folders; retained fixtures replace old TemporaryDirectory cleanup only.',
        preflight_overwrite_reproduced=True,overwrite_probe='Original main block, simulated successful runner, isolated sentinel file.',
        checks_source_sha256=sha(Path(__file__)))
    (OUT/'runtime.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out))


if __name__=='__main__':main()
