"""Run unchanged source-marker cases with the corrected final case-count literal."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import torch

import check_copy_source_end_marker as producer
from position_distractors import digest
from recurrent_retrieval import write


class CountCorrection(ast.NodeTransformer):
    changes=0

    def visit_Compare(self, node):
        expected=ast.dump(ast.parse('len(cells)').body[0].value)
        if ast.dump(node.left)==expected and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq):
            if len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==192:
                node.comparators[0]=ast.copy_location(ast.Constant(value=384),node.comparators[0]);self.changes+=1
        return self.generic_visit(node)


def main():
    root=producer.ROOT;old_dir=producer.DIRECTORY;directory=old_dir/'count_recovery'
    assert (old_dir/'design.json').exists() and not (old_dir/'report.json').exists() and not directory.exists()
    correction=root/'docs/protocols/copy-source-end-marker-count-correction.md'
    assert correction.exists()
    paths=[Path(__file__),Path(producer.__file__),producer.PROTOCOL,old_dir/'design.json',correction]
    hashes={p.relative_to(root).as_posix():digest(p) for p in paths}
    original=inspect.getsource(producer.main);tree=ast.parse(original);transform=CountCorrection()
    tree=transform.visit(tree);ast.fix_missing_locations(tree);assert transform.changes==1
    corrected=ast.unparse(tree)
    producer.DIRECTORY=directory;producer.PROTOCOL=correction
    exec(compile(tree,producer.__file__,'exec'),producer.__dict__)
    producer.main()
    assert all(digest(root/p)==h for p,h in hashes.items())
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    assert report['cells_checked']==384 and report['source_prefix_examples']==768
    write(directory/'count_correction.json',dict(passed=True,only_ast_literal_changes=transform.changes,
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        corrected_function_sha256=hashlib.sha256(corrected.encode()).hexdigest(),
        dependencies=hashes,prior_hashes_unchanged=True,report_sha256=digest(directory/'report.json'),
        correction='192 -> 384; same predeclared Cartesian product',training_updates=0))
    print(json.dumps(dict(correction_passed=True,cells=384,examples=768)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);main()
