"""Independent policy, validation and persistence checks for sparse Snake."""

import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from snake_game.network import SnakeNet
from snake_game.simulation import SnakeSimulation

if importlib.util.find_spec('snake_game.shortcut_network') is not None:
    from snake_game.shortcut_network import ShortcutSnakeNet, SparseLayer
else:
    ShortcutSnakeNet = SparseLayer = None

ROOT = Path(__file__).resolve().parents[1]
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def reference_cycle(cols, rows):
    """Enumerate the path geometrically, independently of the rank compiler."""
    cells = [(cols - 1, 0)]
    for y in range(rows):
        xs = range(cols - 2, -1, -1) if y % 2 == 0 else range(cols - 1)
        cells.extend((x, y) for x in xs)
    cells.extend((cols - 1, y) for y in range(rows - 1, 0, -1))
    return cells


def reference_logits(body, food, cols, rows):
    """A deliberately non-neural oracle for the specified move preferences."""
    cycle = reference_cycle(cols, rows)
    rank = {point: i for i, point in enumerate(cycle)}
    capacity = len(cycle)
    head = body[0]

    def distance(point):
        delta = (rank[point] - rank[head]) % capacity
        return delta if delta else capacity

    food_distance, tail_distance = distance(food), distance(body[-1])
    result = []
    for dx, dy in DELTAS:
        target = head[0] + dx, head[1] + dy
        outside = target not in rank
        d = capacity if outside else distance(target)
        occupied = target in body[:-1]
        density_violation = max(d - 1, 0) if len(body) >= capacity // 2 else 0
        violations = (max(d - food_distance, 0) + max(d - tail_distance, 0)
                      + density_violation + int(outside) + int(occupied))
        result.append(d - (capacity + 1) * violations)
    return np.array(result, dtype=np.float64)


class ShortcutNetworkTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ShortcutSnakeNet, 'The sparse shortcut network is not implemented.')

    def test_encoding_has_raw_head_food_tail_and_full_body_planes(self):
        net = ShortcutSnakeNet(6, 4)
        body, food = [(2, 1), (1, 1), (0, 1)], (4, 3)
        expected = np.zeros((4, 24))
        expected[0, 8] = expected[1, 22] = expected[2, 6] = 1
        expected[3, [6, 7, 8]] = 1
        encoded = net.encode_state(body, food)
        np.testing.assert_array_equal(encoded, expected.ravel())
        np.testing.assert_array_equal(net.forward_state(body, food),
                                      net.forward_observation(encoded))

    def test_parameter_counts_include_biases_and_dense_zeros(self):
        net = ShortcutSnakeNet()
        self.assertEqual(net.policy_name, 'shortcuts')
        self.assertEqual([(layer.input_size, layer.output_size) for layer in net.layers],
                         [(1008, 955), (955, 28), (28, 11), (11, 20), (20, 4)])
        self.assertEqual(net.parameter_count,
                         sum(layer.weights.size + layer.bias.size for layer in net.layers))
        self.assertLess(net.parameter_count, 7200)
        self.assertEqual(net.dense_parameter_count, 991006)

    def test_compiled_ranks_follow_the_existing_baseline_cycle(self):
        for cols, rows in ((6, 4), (7, 6), (18, 14)):
            net, baseline = ShortcutSnakeNet(cols, rows), SnakeNet(cols, rows)
            cycle = reference_cycle(cols, rows)
            for rank, (x, y) in enumerate(cycle):
                self.assertEqual(net.cycle_ranks[y * cols + x], rank)
                dx, dy = DELTAS[int(np.argmax(baseline.forward(x, y)))]
                self.assertEqual((x + dx, y + dy), cycle[(rank + 1) % len(cycle)])

    def test_every_head_position_matches_oracle_including_walls_and_wrap(self):
        for cols, rows in ((6, 4), (7, 6)):
            net = ShortcutSnakeNet(cols, rows)
            cycle = reference_cycle(cols, rows)
            for i, head in enumerate(cycle):
                for length in (1, 4, len(cycle) // 2, len(cycle) - 1):
                    body = [cycle[(i - offset) % len(cycle)] for offset in range(length)]
                    food = cycle[(i + 1) % len(cycle)]
                    with self.subTest(cols=cols, rows=rows, head=head, length=length):
                        np.testing.assert_array_equal(net.forward_state(body, food),
                                                      reference_logits(body, food, cols, rows))

    def test_vacating_tail_is_legal_but_non_tail_body_is_penalized(self):
        net = ShortcutSnakeNet(6, 4)
        body, food = [(1, 1), (1, 2), (2, 2), (2, 1)], (4, 1)
        logits = net.forward_state(body, food)
        self.assertEqual(int(np.argmax(logits)), 1)
        self.assertEqual(logits[1], 1)
        blocked = [(2, 1), (2, 2), (3, 2), (3, 1), (4, 1)]
        blocked_logits = net.forward_state(blocked, (5, 0))
        self.assertEqual(blocked_logits[1], -24)
        np.testing.assert_array_equal(blocked_logits,
                                      reference_logits(blocked, (5, 0), 6, 4))

    def test_shortcuts_stop_at_half_capacity(self):
        net = ShortcutSnakeNet(6, 4)
        cycle = reference_cycle(6, 4)
        head_rank = cycle.index((2, 1))
        body = [cycle[(head_rank - offset) % 24] for offset in range(11)]
        food = (2, 2)
        self.assertEqual(int(np.argmax(net.forward_state(body, food))), 2)
        body.append(cycle[(head_rank - 11) % 24])
        self.assertEqual(int(np.argmax(net.forward_state(body, food))), 1)

    def test_logits_match_oracle_through_seeded_rollouts(self):
        rng = random.Random(124)
        for cols, rows in ((6, 4), (18, 14)):
            net = ShortcutSnakeNet(cols, rows)
            for seed in (3, 19):
                sim = SnakeSimulation(seed=seed, cols=cols, rows=rows,
                                      net=SnakeNet(cols, rows))
                for step in range(600):
                    if sim.status != 'running':
                        break
                    logits = net.forward_state(sim.body, sim.food)
                    if step % 13 == 0 or rng.random() < 0.15:
                        np.testing.assert_array_equal(logits,
                            reference_logits(sim.body, sim.food, cols, rows))
                    sim.step(int(np.argmax(logits)))
                self.assertNotEqual(sim.status, 'dead')

    def test_complete_games_and_last_cell_growth(self):
        for cols, rows, seed in ((6, 4, 0), (7, 6, 17), (18, 14, 23)):
            net = ShortcutSnakeNet(cols, rows)
            sim = SnakeSimulation(seed=seed, cols=cols, rows=rows,
                                  net=SnakeNet(cols, rows))
            while sim.status == 'running' and sim.steps < (cols * rows) ** 2:
                sim.step(int(np.argmax(net.forward_state(sim.body, sim.food))))
            with self.subTest(cols=cols, rows=rows):
                self.assertEqual(sim.status, 'won')
                self.assertEqual(len(set(sim.body)), cols * rows)
                self.assertIsNone(sim.food)

    def test_stored_readout_can_override_the_policy_without_a_fallback(self):
        net = ShortcutSnakeNet(6, 4)
        net.layers[-1].weights[:] = 0
        net.layers[-1].bias[:] = [8, -2, 0, 1]
        net.cycle_ranks[:] = -123
        # UP is a wall: the exact requested logits must still come through.
        np.testing.assert_array_equal(net.forward_state([(0, 0)], (3, 2)), [8, -2, 0, 1])

    def test_sparse_layers_equal_explicit_dense_matrix_evaluation(self):
        net = ShortcutSnakeNet(6, 4)
        values = net.encode_state([(2, 1), (1, 1)], (3, 2))
        original = values.copy()
        for index, layer in enumerate(net.layers):
            dense = np.zeros((layer.output_size, layer.input_size))
            np.add.at(dense, (layer.dst, layer.src), layer.weights)
            values = dense @ values + layer.bias
            if index < len(net.layers) - 1:
                values = np.maximum(values, 0)
        np.testing.assert_array_equal(net.forward_observation(original), values)

    def test_invalid_board_and_state_inputs_are_rejected(self):
        for dims in ((5, 4), (6, 3), (6.0, 4), (True, 4), (6, np.bool_(True))):
            with self.subTest(dimensions=dims), self.assertRaises(ValueError):
                ShortcutSnakeNet(*dims)
        net = ShortcutSnakeNet(6, 4)
        for body, food in (([], (1, 1)), (None, (1, 1)),
                           ([(1, 1), (1, 1)], (2, 2)),
                           ([(1, 1), (3, 1)], (2, 2)),
                           ([(1.0, 1)], (2, 2)), ([(True, 1)], (2, 2)),
                           ([(1, 1), (np.bool_(False), 1)], (2, 2)),
                           ([(-1, 1)], (2, 2)), ([(6, 1)], (2, 2)),
                           ([(1, 4)], (2, 2)), ([(1, 1, 1)], (2, 2)),
                           ([(1, 1)], None), ([(1, 1)], (1, 1)),
                           ([(1, 1), (2, 1)], (2, 1)),
                           ([(1, 1)], (2.0, 2)), ([(1, 1)], (True, 2)),
                           ([(1, 1)], (6, 2)), ([(1, 1)], (1, -1))):
            with self.subTest(body=body, food=food), self.assertRaises(ValueError):
                net.encode_state(body, food)

    def test_invalid_raw_observations_are_rejected(self):
        net = ShortcutSnakeNet(6, 4)
        valid = net.encode_state([(2, 1), (1, 1)], (3, 2))
        invalid = [valid[:-1], valid.reshape(4, 24), ['1'] * 96, None]
        for index, value in ((0, np.nan), (0, np.inf), (0, 0.5), (0, -1),
                             (0, 1), (8, 0), (39, 0), (55, 0), (79, 0), (87, 1)):
            vector = valid.copy()
            vector[index] = value
            invalid.append(vector)
        for vector in invalid:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                net.forward_observation(vector)


class ShortcutPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ShortcutSnakeNet, 'The sparse shortcut network is not implemented.')

    def test_v2_roundtrip_preserves_edited_weights_and_indices(self):
        net = ShortcutSnakeNet(6, 4)
        net.layers[-1].weights[:] = 0
        net.layers[-1].bias[:] = [8, -2, 0.125, 1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.json'
            net.save(path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual((payload['kind'], payload['version']), ('shortcuts', 2))
            self.assertEqual(payload['architecture'], [96, 87, 28, 11, 20, 4])
            loaded = ShortcutSnakeNet.load(path)
            self.assertEqual(loaded.parameter_count, net.parameter_count)
            for old, new in zip(net.layers, loaded.layers):
                for name in ('src', 'dst', 'weights', 'bias'):
                    np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
            np.testing.assert_array_equal(loaded.forward_state([(0, 0)], (3, 2)),
                                          [8, -2, 0.125, 1])

    def test_malformed_sparse_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.json'
            ShortcutSnakeNet(6, 4).save(path)
            original = json.loads(path.read_text(encoding='utf-8'))
            bad = [None, [], {}, {'version': 2}, {'kind': 'shortcuts', 'version': True}]
            for key, value in (('kind', 'cycle'), ('version', 1), ('cols', True),
                               ('rows', 5), ('architecture', [96, 87, 4]),
                               ('architecture', [96.0, 87, 28, 11, 20, 4]),
                               ('layers', []), ('layers', {})):
                payload = copy.deepcopy(original)
                payload[key] = value
                bad.append(payload)
            for key, value in (('src', [True]), ('src', [0.0]), ('src', [-1]),
                               ('src', [96]), ('dst', [87]), ('dst', [-1]),
                               ('dst', [False]), ('weights', [float('nan')]),
                               ('weights', [float('inf')]), ('weights', ['1']),
                               ('weights', [True]), ('weights', [[1]]),
                               ('bias', [0]), ('bias', [float('inf')] * 87),
                               ('input_size', 97), ('output_size', True)):
                payload = copy.deepcopy(original)
                layer = payload['layers'][0]
                if key in ('src', 'dst', 'weights'):
                    layer['src'], layer['dst'], layer['weights'] = [0], [0], [1]
                layer[key] = value
                bad.append(payload)
            for key in ('weights', 'src', 'dst', 'bias', 'input_size'):
                payload = copy.deepcopy(original)
                del payload['layers'][0][key]
                bad.append(payload)
            payload = copy.deepcopy(original)
            payload['layers'][0]['src'].append(0)
            bad.append(payload)
            for i, payload in enumerate(bad):
                with self.subTest(case=i):
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        ShortcutSnakeNet.load(path)
            path.write_text('{not JSON', encoding='utf-8')
            with self.assertRaises(ValueError):
                ShortcutSnakeNet.load(path)

    def test_bundled_weights_match_the_analytic_model(self):
        loaded = ShortcutSnakeNet.load(ROOT / 'weights' / 'snake_shortcuts.json')
        net = ShortcutSnakeNet()
        body, food = [(4, 7), (3, 7), (2, 7), (1, 7)], (13, 7)
        np.testing.assert_array_equal(loaded.forward_state(body, food),
                                      net.forward_state(body, food))
        for old, new in zip(net.layers, loaded.layers):
            for name in ('src', 'dst', 'weights', 'bias'):
                np.testing.assert_array_equal(getattr(old, name), getattr(new, name))


if __name__ == '__main__':
    unittest.main()
