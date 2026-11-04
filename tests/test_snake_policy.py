"""Integration contracts for state-aware control and baseline comparisons."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from snake_game import SnakeNet, SnakeSimulation
from snake_game.network import create_network, load_network

ROOT = Path(__file__).resolve().parents[1]


class PolicyIntegrationTests(unittest.TestCase):
    def test_shortcuts_are_the_default_but_cycle_remains_selectable(self):
        self.assertEqual(create_network().policy_name, 'shortcuts')
        self.assertEqual(SnakeSimulation().net.policy_name, 'shortcuts')
        self.assertEqual(create_network('cycle').policy_name, 'cycle')
        self.assertEqual(create_network('cycle').parameter_count, 150)

    def test_legacy_weight_file_keeps_its_original_policy(self):
        net = load_network(ROOT / 'weights' / 'snake_classic.json')
        self.assertEqual(net.policy_name, 'cycle')
        expected = SnakeNet()
        for x, y in ((0, 0), (4, 7), (17, 0), (16, 13)):
            np.testing.assert_array_equal(net.forward(x, y), expected.forward(x, y))

    def test_stateful_logits_are_executed_without_a_safety_override(self):
        net = create_network()
        net.layers[-1].weights[:] = 0
        net.layers[-1].bias[:] = [0, 0, 0, 99]
        game = SnakeSimulation(net=net)
        self.assertEqual(game.step(), 'dead')  # LEFT is the occupied neck.
        self.assertEqual(game.direction, 3)
        np.testing.assert_array_equal(game.last_logits, [0, 0, 0, 99])

    def test_default_policy_takes_real_shortcuts_and_finishes(self):
        game = SnakeSimulation(seed=7)
        for _ in range(100000):
            if game.step() != 'running':
                break
        self.assertEqual(game.status, 'won')
        self.assertGreater(game.shortcuts, 0)
        self.assertLess(game.steps, 16230)  # Measured original cycle, seed 7.

    def test_complete_network_roundtrip_keeps_trajectory(self):
        net = create_network()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'complete.json'
            net.save(path)
            first = SnakeSimulation(seed=3, net=net)
            second = SnakeSimulation(seed=3, net=load_network(path))
            for _ in range(100):
                first.step()
                second.step()
                self.assertEqual((first.body, first.food), (second.body, second.food))


class CompareCliTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'snake.py'), *map(str, args)],
                              cwd=ROOT, text=True, capture_output=True, timeout=90)

    def test_compare_reports_both_policies_on_the_same_seed(self):
        result = self.cli('--compare', '--seed', 7, '--seeds', 1, '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['baseline']['policy'], 'cycle')
        self.assertEqual(report['candidate']['policy'], 'shortcuts')
        for model in ('baseline', 'candidate'):
            self.assertEqual(report[model]['first_seed'], 7)
            self.assertEqual(report[model]['wins'], 1)
            self.assertEqual(report[model]['games'], 1)
        self.assertGreater(report['move_reduction_pct'], 0)

    def test_cycle_flag_preserves_measured_baseline(self):
        result = self.cli('--headless', '--policy', 'cycle', '--seed', 7, '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['total_steps'], 16230)
        self.assertEqual(report['parameters'], 150)
        self.assertEqual(report['shortcuts'], 0)

    def test_incomplete_comparison_does_not_claim_an_improvement(self):
        result = self.cli('--compare', '--max-steps', 1, '--json')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report['all_completed'])
        self.assertIsNone(report['move_reduction_pct'])
        self.assertEqual(report['candidate']['timeouts'], 1)


if __name__ == '__main__':
    unittest.main()
