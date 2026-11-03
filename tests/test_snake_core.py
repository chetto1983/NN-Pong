"""The analytic controller and the game rules, without a display or training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from snake_game import DOWN, LEFT, RIGHT, UP, SnakeNet, SnakeSimulation


ROOT = Path(__file__).resolve().parents[1]
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class NetworkTests(unittest.TestCase):
    def test_default_network_uses_150_parameters(self):
        net = SnakeNet()
        shapes = ((18, 2), (18,), (4, 18), (4,), (4, 4), (4,))
        arrays = (net.W1, net.b1, net.W2, net.b2, net.W3, net.b3)
        self.assertEqual(tuple(a.shape for a in arrays), shapes)
        self.assertEqual(sum(a.size for a in arrays), 150)

    def test_known_turns_and_straights(self):
        net = SnakeNet()
        for position, direction in (
            ((0, 0), DOWN), ((4, 0), LEFT), ((17, 0), LEFT),
            ((0, 1), RIGHT), ((4, 7), RIGHT), ((16, 7), DOWN),
            ((16, 8), LEFT), ((0, 8), DOWN), ((16, 13), RIGHT),
            ((17, 13), UP), ((17, 8), UP), ((17, 1), UP),
        ):
            with self.subTest(position=position):
                logits = net.forward(*position)
                self.assertEqual(logits.shape, (4,))
                self.assertEqual(int(np.argmax(logits)), direction)

    def test_every_cell_belongs_to_one_legal_cycle(self):
        for cols, rows in ((6, 4), (7, 4), (8, 6), (18, 14), (31, 20)):
            with self.subTest(cols=cols, rows=rows):
                net = SnakeNet(cols, rows)
                position = (0, 0)
                visited = set()
                for _ in range(cols * rows):
                    self.assertNotIn(position, visited)
                    visited.add(position)
                    dx, dy = DELTAS[int(np.argmax(net.forward(*position)))]
                    position = (position[0] + dx, position[1] + dy)
                    self.assertTrue(0 <= position[0] < cols)
                    self.assertTrue(0 <= position[1] < rows)
                self.assertEqual(position, (0, 0))
                self.assertEqual(len(visited), cols * rows)

    def test_output_is_controlled_by_matrices_even_at_a_corner(self):
        net = SnakeNet()
        net.W3[:] = 0
        net.b3[:] = [-4, 7, 2, 0]
        np.testing.assert_array_equal(net.forward(17, 0), [-4, 7, 2, 0])

    def test_invalid_board_dimensions_are_rejected(self):
        for cols, rows in ((5, 4), (6, 2), (6, 5), (0, 4),
                           (6.0, 4), (6, 4.0), (True, 4), (6, True)):
            with self.subTest(cols=cols, rows=rows):
                with self.assertRaises(ValueError):
                    SnakeNet(cols, rows)

    def test_invalid_head_coordinates_are_rejected(self):
        net = SnakeNet()
        for x, y in ((-1, 0), (18, 0), (0, 14), (0, -1),
                     (1.5, 1), (1, np.nan), (np.inf, 1), (True, 1)):
            with self.subTest(x=x, y=y):
                with self.assertRaises(ValueError):
                    net.forward(x, y)

    def test_weights_round_trip_without_changing_logits(self):
        for cols, rows in ((6, 4), (18, 14)):
            with self.subTest(cols=cols, rows=rows), tempfile.TemporaryDirectory() as folder:
                net = SnakeNet(cols, rows)
                net.b3[1] = 0.125
                path = Path(folder) / 'weights.json'
                net.save(path)
                loaded = SnakeNet.load(path)
                self.assertEqual((loaded.cols, loaded.rows), (cols, rows))
                for y in range(rows):
                    for x in range(cols):
                        np.testing.assert_array_equal(loaded.forward(x, y), net.forward(x, y))

    def test_malformed_weight_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.json'
            SnakeNet().save(path)
            original = json.loads(path.read_text(encoding='utf-8'))
            bad_payloads = [[], {}, {'version': 999}, None]
            for key, value in (('version', 999), ('cols', 5), ('rows', 5),
                               ('W1', [[0, 0]]), ('b2', [0, 0, 0]),
                               ('W3', [[0] * 4] * 3), ('b3', [0, 0, 0, float('inf')]),
                               ('b1', [float('nan')] * 18)):
                payload = copy.deepcopy(original)
                payload[key] = value
                bad_payloads.append(payload)
            del original['W2']
            bad_payloads.append(original)
            for payload in bad_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        SnakeNet.load(path)
            path.write_text('{invalid JSON', encoding='utf-8')
            with self.assertRaises(ValueError):
                SnakeNet.load(path)

    def test_bundled_weights_match_the_analytic_policy(self):
        loaded = SnakeNet.load(ROOT / 'weights' / 'snake_classic.json')
        expected = SnakeNet()
        for y in range(14):
            for x in range(18):
                np.testing.assert_array_equal(loaded.forward(x, y), expected.forward(x, y))


class SimulationTests(unittest.TestCase):
    def test_default_start_matches_classic_layout(self):
        sim = SnakeSimulation()
        self.assertEqual(sim.body, [(4, 7), (3, 7), (2, 7), (1, 7)])
        self.assertEqual(sim.food, (13, 7))
        self.assertEqual((sim.score, sim.steps, sim.status, sim.direction), (0, 0, 'running', RIGHT))
        self.assertEqual(int(np.argmax(sim.last_logits)), RIGHT)

    def test_normal_step_moves_the_head_and_vacates_the_tail(self):
        sim = SnakeSimulation()
        self.assertEqual(sim.step(), 'running')
        self.assertEqual(sim.body, [(5, 7), (4, 7), (3, 7), (2, 7)])
        self.assertEqual((sim.steps, sim.score, sim.direction), (1, 0, RIGHT))

    def test_eating_grows_and_spawns_on_an_empty_cell(self):
        sim = SnakeSimulation(seed=9)
        sim.food = (5, 7)
        self.assertEqual(sim.step(), 'running')
        self.assertEqual(len(sim.body), 5)
        self.assertEqual(sim.body[-1], (1, 7))
        self.assertEqual(sim.score, 1)
        self.assertNotIn(sim.food, sim.body)

    def test_entering_a_vacating_tail_is_legal(self):
        sim = SnakeSimulation(cols=6, rows=4)
        sim.body = [(1, 1), (1, 2), (2, 2), (2, 1)]
        sim.food = (5, 3)
        self.assertEqual(sim.step(RIGHT), 'running')
        self.assertEqual(sim.body, [(2, 1), (1, 1), (1, 2), (2, 2)])

    def test_wall_and_body_collisions_keep_a_consistent_body(self):
        for body, direction in (([(0, 1), (1, 1), (2, 1), (3, 1)], LEFT),
                                ([(4, 7), (3, 7), (2, 7), (1, 7)], LEFT)):
            with self.subTest(body=body):
                sim = SnakeSimulation()
                sim.body = body.copy()
                food = sim.food
                self.assertEqual(sim.step(direction), 'dead')
                self.assertEqual(sim.body, body)
                self.assertEqual(sim.food, food)
                self.assertEqual((sim.score, sim.steps), (0, 1))

    def test_terminal_state_does_not_advance(self):
        sim = SnakeSimulation()
        sim.step(LEFT)
        state = (sim.body.copy(), sim.food, sim.score, sim.steps, sim.direction)
        for _ in range(5):
            self.assertEqual(sim.step(), 'dead')
        self.assertEqual((sim.body, sim.food, sim.score, sim.steps, sim.direction), state)

    def test_invalid_actions_do_not_mutate_state(self):
        sim = SnakeSimulation()
        for action in (-1, 4, 1.0, True, 'right', np.nan):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    sim.step(action)
                self.assertEqual((sim.steps, sim.score, sim.status), (0, 0, 'running'))
                self.assertEqual(sim.body[0], (4, 7))

    def test_custom_network_must_match_board_dimensions(self):
        with self.assertRaises(ValueError):
            SnakeSimulation(net=SnakeNet(6, 4))
        sim = SnakeSimulation(cols=6, rows=4, net=SnakeNet(6, 4))
        self.assertEqual(sim.step(), 'running')

    def test_default_action_comes_from_network_output(self):
        net = SnakeNet()
        net.W3[:] = 0
        net.b3[:] = [10, 0, 0, 0]
        sim = SnakeSimulation(net=net)
        sim.step()
        self.assertEqual((sim.body[0], sim.direction), ((4, 6), UP))
        np.testing.assert_array_equal(sim.last_logits, [10, 0, 0, 0])

    def test_filling_the_last_cell_wins_and_removes_food(self):
        sim = SnakeSimulation(cols=6, rows=4)
        cycle = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                 (4, 2), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3),
                 (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2),
                 (5, 1), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0)]
        sim.body = list(reversed(cycle[:-1]))
        sim.score = 19
        sim.food = (1, 0)
        self.assertEqual(sim.step(), 'won')
        self.assertEqual(len(sim.body), 24)
        self.assertEqual(len(set(sim.body)), 24)
        self.assertEqual(sim.score, 20)
        self.assertIsNone(sim.food)
        self.assertEqual(sim.step(), 'won')
        self.assertEqual(sim.steps, 1)

    def test_complete_games_on_multiple_sizes_and_seeds(self):
        for cols, rows in ((6, 4), (7, 6), (18, 14)):
            for seed in (0, 1, 17):
                with self.subTest(cols=cols, rows=rows, seed=seed):
                    sim = SnakeSimulation(seed=seed, cols=cols, rows=rows)
                    capacity = cols * rows
                    for _ in range(capacity * capacity):
                        if sim.status != 'running':
                            break
                        sim.step()
                        self.assertEqual(len(sim.body), len(set(sim.body)))
                        self.assertNotIn(sim.food, sim.body)
                    self.assertEqual(sim.status, 'won')
                    self.assertEqual(sim.score, capacity - 4)
                    self.assertEqual(len(sim.body), capacity)

    def test_same_seed_reproduces_food_and_state(self):
        first, second = SnakeSimulation(seed=17), SnakeSimulation(seed=17)
        for _ in range(1000):
            first.step()
            second.step()
            self.assertEqual((first.body, first.food, first.score, first.status),
                             (second.body, second.food, second.score, second.status))


if __name__ == '__main__':
    unittest.main()
