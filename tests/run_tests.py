"""Run archived copy test cases with fresh, retained fixtures, never preflights."""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import types
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results' / 'test_runs'
ARCHIVED = ('test_copy_rope', 'test_copy_marker', 'test_copy_marker_alignment')
FROZEN = ('copy_rope_training', 'copy_marker_training', 'copy_marker_alignment')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_run_directory(base, name):
    # Reject path syntax before creating even the parent directory.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Run name must be a single alphanumeric, underscore or hyphen component.')
    base = base.resolve()
    target = base / name
    if target.resolve().parent != base:
        raise ValueError('Run directory must stay inside the fixture root.')
    base.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def protected_hashes():
    paths = [ROOT / 'tests' / f'{name}.py' for name in ARCHIVED]
    paths.extend(ROOT / 'docs' / 'protocols' / f'{name}.md' for name in
                 ('copy-rope-training', 'copy-marker-training', 'copy-marker-alignment'))
    for name in FROZEN:
        paths.extend(p for p in (ROOT / 'results' / name).iterdir() if p.is_file())
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}


def load_archived(name, directory):
    if name not in ARCHIVED:
        raise ValueError('Unknown archived test module.')
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    # Importing does not execute the historical __main__ recording blocks.
    spec.loader.exec_module(module)
    module.DIRECTORY = directory
    directory.mkdir(exist_ok=False)
    if name == 'test_copy_rope':
        class RetainedFixture:
            def __init__(self, *, dir):
                if Path(dir).resolve() != (ROOT / 'results').resolve():
                    raise ValueError('Unexpected archived temporary-directory root.')
                self.path = directory / ('fixture_' + uuid.uuid4().hex)

            def __enter__(self):
                self.path.mkdir(exist_ok=False)
                return str(self.path)

            def __exit__(self, *args):
                return False

        # Only the archived module's temporary-file helper changes. Its forward,
        # gradient, evaluator, audit and negative-control cases stay unchanged.
        module.tempfile = types.SimpleNamespace(TemporaryDirectory=RetainedFixture)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='run_' + uuid.uuid4().hex,
                        help='Optional fresh folder name inside results/test_runs.')
    args = parser.parse_args(argv)
    try:
        directory = create_run_directory(RUNS, args.run_name)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    before = protected_hashes()
    sys.path.insert(0, str(ROOT / 'experiments'))
    sys.path.insert(0, str(ROOT / 'tests'))
    modules = [load_archived(name, directory / name) for name in ARCHIVED]
    safety = __import__('test_runner_safety')
    safety.SAFETY_ROOT = directory / 'runner_safety'
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
                               for m in (*modules, safety))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = before == protected_hashes()
    cases = {m.__name__: len(getattr(m, 'CASES', getattr(m, 'ERRORS', []))) for m in modules}
    summary = dict(passed=result.wasSuccessful() and unchanged, methods=result.testsRun,
                   initialized_equation_gradient_cases=cases,
                   frozen_files_unchanged=unchanged, frozen_files_checked=len(before),
                   training_updates=0, source_sha256={
                       str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                       for p in [Path(__file__), ROOT / 'tests/test_runner_safety.py',
                                 *[ROOT / 'tests' / f'{name}.py' for name in ARCHIVED]]},
                   fixture_policy='Fresh retained folders; historical main blocks are not invoked.')
    with (directory / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    print(f'Results: {directory / "summary.json"}')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
