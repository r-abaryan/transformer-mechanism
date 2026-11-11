"""Safety regression cases for the current entry point, no training."""
import unittest
import uuid

from run_tests import ARCHIVED, RUNS, create_run_directory, load_archived

SAFETY_ROOT = RUNS


class RunnerSafetyTests(unittest.TestCase):
    def test_existing_run_rejected_without_overwrite(self):
        base = create_run_directory(SAFETY_ROOT, 'safety_' + uuid.uuid4().hex)
        child = create_run_directory(base, 'occupied')
        sentinel = child / 'preflight.json'
        sentinel.write_bytes(b'previous evidence\n')
        with self.assertRaises(FileExistsError):
            create_run_directory(base, 'occupied')
        self.assertEqual(sentinel.read_bytes(), b'previous evidence\n')

    def test_path_syntax_rejected_before_any_write(self):
        base = SAFETY_ROOT / ('uncreated_' + uuid.uuid4().hex)
        for name in ('../escape', '..', 'a/b', 'a\\b', 'C:\\escape', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_run_directory(base, name)
        self.assertFalse(base.exists())

    def test_imports_do_not_record_preflights(self):
        base = create_run_directory(SAFETY_ROOT, 'imports_' + uuid.uuid4().hex)
        for name in ARCHIVED:
            directory = base / name
            module = load_archived(name, directory)
            self.assertEqual(module.DIRECTORY, directory)
            self.assertEqual(list(directory.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
