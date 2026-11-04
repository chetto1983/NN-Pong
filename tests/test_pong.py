"""Regressioni della rete, della fisica e dell'interfaccia da terminale."""

import subprocess
import sys
import unittest
from pathlib import Path

import pong

ROOT = Path(__file__).resolve().parents[1]


class NetworkTests(unittest.TestCase):
    def test_fold_accounts_for_ball_radius(self):
        net = pong.PongNet()
        # Coordinate del centro palla e risultati calcolati a mano.
        for raw, expected in ((16, 16), (634, 634), (644, 624),
                              (6, 26), (1000, 268), (1252, 16),
                              (-1220, 16)):
            with self.subTest(raw=raw):
                _, info = net.forward(325, pong.wrap_period(raw))
                self.assertAlmostEqual(info['fold'], expected)
                self.assertAlmostEqual(pong.reflect_y(raw), expected)

    def test_network_has_fifteen_parameters(self):
        self.assertEqual(pong.PongNet().parameter_count, 15)

    def test_matches_reference_controller_on_whole_domain(self):
        net = pong.PongNet()
        for paddle in range(0, pong.HEIGHT + 1, 13):
            for raw in range(-1300, 1900, 7):
                y = pong.wrap_period(raw)
                error = pong.reflect_y(raw) - paddle
                expected = max(0, error - pong.DEADZONE) - max(0, -error - pong.DEADZONE)
                output, info = net.forward(paddle, y)
                self.assertAlmostEqual(output, expected, places=9)
                # The first hidden unit never switches off on the domain.
                self.assertGreater(info['layer1'][0], 0)

    def test_period_endpoint_folds_to_top_wall(self):
        _, info = pong.PongNet().forward(325, pong.BALL_MIN_Y + 2 * pong.SPAN)
        self.assertAlmostEqual(info['fold'], pong.BALL_MIN_Y)

    def test_inputs_outside_domain_are_rejected(self):
        net = pong.PongNet()
        for paddle, y in ((325, pong.BALL_MIN_Y - 1), (325, 1300), (-1, 300), (pong.HEIGHT + 1, 300)):
            with self.subTest(paddle=paddle, y=y), self.assertRaises(ValueError):
                net.forward(paddle, y)

    def test_deadzone_and_direction(self):
        net = pong.PongNet()
        for target, expected in ((180, -16), (195, -1), (196, 0),
                                 (200, 0), (204, 0), (205, 1), (220, 16)):
            with self.subTest(target=target):
                output, _ = net.forward(200, target)
                self.assertAlmostEqual(output, expected)

    def test_negative_deadzone_is_rejected(self):
        with self.assertRaises(ValueError):
            pong.PongNet(delta=-1)


class SimulationTests(unittest.TestCase):
    def test_predicted_intercept_matches_physical_bounce(self):
        sim = pong.PongSimulation()
        sim.ball_x, sim.ball_y = 500.0, 600.0
        sim.ball_vx, sim.ball_vy = -360.0, 210.0
        raw = sim.predict_player_intercept()
        _, info = sim.net.forward(sim.player_y, raw)
        # Impatto a x=57 dopo 443/360 secondi; centro rimbalza a y=634.
        self.assertAlmostEqual(info['fold'], 409.5833333333333)

    def test_wall_bounces_keep_ball_inside_field(self):
        for y, velocity, expected_y, expected_velocity in (
            (17, -180, 18, 180), (633, 180, 632, -180)
        ):
            with self.subTest(y=y):
                sim = pong.PongSimulation()
                sim.ball_y, sim.ball_vy = y, velocity
                sim.step()
                self.assertAlmostEqual(sim.ball_y, expected_y)
                self.assertAlmostEqual(sim.ball_vy, expected_velocity)

    def test_same_seed_reproduces_game(self):
        first, second = pong.PongSimulation(seed=17), pong.PongSimulation(seed=17)
        for _ in range(2000):
            first.step()
            second.step()
        self.assertEqual(first.stats, second.stats)
        self.assertEqual((first.ball_x, first.ball_y, first.player_y),
                         (second.ball_x, second.ball_y, second.player_y))
        self.assertGreater(first.stats.hits, 0)

    def test_moving_away_or_stationary_ball_targets_center(self):
        for vx in (0, 360):
            with self.subTest(vx=vx):
                sim = pong.PongSimulation()
                sim.ball_y, sim.ball_vx = 100, vx
                self.assertEqual(sim.predict_player_intercept(), 325)

    def test_missed_ball_updates_score_and_serves(self):
        for x, vx, expected_player, expected_opponent in (
            (-10, -360, 0, 1), (1010, 360, 1, 0)
        ):
            with self.subTest(x=x):
                sim = pong.PongSimulation()
                sim.ball_x, sim.ball_y = x, 100
                sim.ball_vx, sim.ball_vy = vx, 0
                sim.player_y = sim.opponent_y = 550
                sim.step()
                self.assertEqual(sim.stats.player_score, expected_player)
                self.assertEqual(sim.stats.opponent_score, expected_opponent)
                self.assertEqual(sim.stats.misses, expected_opponent)
                self.assertEqual((sim.ball_x, sim.ball_y), (500, 325))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / 'pong.py'), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )

    def test_headless_command_runs(self):
        result = self.run_cli('--headless', '--seeds', '2', '--frames', '20')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('STRESS 2 seed x 20 frame = 40 frame', result.stdout)

    def test_nonpositive_counts_are_rejected(self):
        for option in ('--frames', '--seeds'):
            for value in ('0', '-1'):
                with self.subTest(option=option, value=value):
                    result = self.run_cli('--headless', option, value)
                    self.assertEqual(result.returncode, 2, result.stdout)
                    self.assertIn(option, result.stderr)


if __name__ == '__main__':
    unittest.main()
