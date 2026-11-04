"""Contratto della CLI Snake: risultati reali e pesi esportabili."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SnakeCliTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / 'snake.py'), *map(str, args)],
            cwd=ROOT, text=True, capture_output=True, timeout=60,
        )

    def test_default_weights_complete_a_board(self):
        result = self.cli('--headless', '--seed', 7, '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['games'], 1)
        self.assertEqual(report['wins'], 1)
        self.assertEqual(report['deaths'], 0)
        self.assertEqual(report['timeouts'], 0)
        self.assertEqual(report['min_score'], 248)
        self.assertEqual(report['max_score'], 248)
        self.assertGreater(report['total_steps'], 248)

    def test_timeout_is_reported_as_incomplete(self):
        result = self.cli('--headless', '--max-steps', 1, '--json')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['wins'], 0)
        self.assertEqual(report['timeouts'], 1)
        self.assertEqual(report['total_steps'], 1)

    def test_exported_weights_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            weights = Path(temp) / 'weights.json'
            export = self.cli('--export-weights', weights)
            self.assertEqual(export.returncode, 0, export.stderr)
            self.assertTrue(weights.is_file())
            loaded = self.cli('--weights', weights, '--test')
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            self.assertIn('PASS', loaded.stdout)

    def test_invalid_arguments_are_rejected(self):
        for args in (('--seeds', '0'), ('--max-steps', '-1'),
                     ('--speed', 'nan'), ('--speed', '0'),
                     ('--speed', '121')):
            with self.subTest(args=args):
                result = self.cli('--headless', *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(args[0], result.stderr)

    def test_bad_weights_report_a_useful_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.json'
            path.write_text('{"invalid": true}', encoding='utf-8')
            result = self.cli('--headless', '--weights', path)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Traceback', result.stderr)
            self.assertIn('pesi', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
