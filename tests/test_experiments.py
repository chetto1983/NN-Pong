"""Fast checks of the experiment code; the experiments themselves run separately."""

import unittest

import numpy as np

from experiments.e1_pong_folds import MultiFoldNet, reference
from experiments.mlp import MLP, mse
from experiments.policies import greedy_action, shortcut_logits
from pong_game.config import HEIGHT
from pong_game.network import PongNet
from snake_game import SnakeSimulation
from snake_game.network import create_network


class PongFoldTests(unittest.TestCase):
    def test_smallest_multifold_matches_shipped_size(self):
        self.assertEqual(MultiFoldNet(0, 2).parameter_count, PongNet().parameter_count)

    def test_each_extra_reflection_costs_one_hinge(self):
        for left, right in ((0, 2), (0, 3), (2, 3), (4, 6)):
            net = MultiFoldNet(left, right)
            self.assertEqual(net.hidden, left + right)
            self.assertEqual(net.parameter_count, 5 * net.hidden + 5)

    def test_multifold_is_exact_on_raw_domain(self):
        rng = np.random.default_rng(3)
        for left, right in ((0, 2), (2, 3), (4, 6)):
            net = MultiFoldNet(left, right)
            for p, y in zip(rng.uniform(0, HEIGHT, 2000), rng.uniform(net.lo, net.hi, 2000)):
                self.assertAlmostEqual(net.forward(p, y)[0], reference(p, y), places=8)

    def test_left_must_be_even(self):
        with self.assertRaises(ValueError):
            MultiFoldNet(1, 2)


class SnakePolicyTests(unittest.TestCase):
    def test_python_rule_matches_network_logits(self):
        net = create_network('shortcuts')
        game = SnakeSimulation(seed=5, net=net)
        for _ in range(1500):
            expected = shortcut_logits(game.body, game.food, 18, 14, net.cycle_ranks)
            np.testing.assert_array_equal(net.forward_state(game.body, game.food), expected)
            game.step()

    def test_greedy_returns_a_legal_move(self):
        game = SnakeSimulation(seed=0)
        for _ in range(200):
            action = greedy_action(game.body, game.food, game.cols, game.rows)
            self.assertIn(action, range(4))
            self.assertEqual(game.step(action), 'running')


class MLPTests(unittest.TestCase):
    def test_regression_loss_decreases(self):
        rng = np.random.default_rng(0)
        mlp = MLP([2, 16, 1], rng)
        x = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        t = (x[:, :1] - x[:, 1:]).astype(np.float32)
        first = mlp.step(x, mse(t), 1e-2)
        for _ in range(300):
            last = mlp.step(x, mse(t), 1e-2)
        self.assertLess(last, first / 10)


if __name__ == '__main__':
    unittest.main()
