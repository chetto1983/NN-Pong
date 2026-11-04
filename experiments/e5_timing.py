"""E5 - Cost of the neural form: the same decision as network versus plain Python.

Single-decision latency (what the games do) and, for Pong, a batched NumPy
version. Before timing, the Snake network and its Python rule are checked to
give identical logits on every state used.
"""

from __future__ import annotations

import argparse
import timeit

import numpy as np

from pong_game.config import BALL_MIN_Y, HEIGHT, SPAN
from pong_game.network import PongNet
from snake_game import SnakeSimulation
from snake_game.network import create_network

from .common import save
from .e1_pong_folds import reference
from .policies import shortcut_logits


def per_call(fn, items, repeat):
    best = min(timeit.repeat(lambda: [fn(*item) for item in items], number=1, repeat=repeat))
    return best / len(items) * 1e6


def pong(repeat):
    rng = np.random.default_rng(0)
    items = list(zip(rng.uniform(0, HEIGHT, 5000), rng.uniform(BALL_MIN_Y, BALL_MIN_Y + 2 * SPAN, 5000)))
    net = PongNet()
    network_us = per_call(lambda p, y: net.forward(p, y)[0], items, repeat)
    python_us = per_call(reference, items, repeat)
    p = np.array([i[0] for i in items])
    y = np.array([i[1] for i in items])
    x = np.stack((p, y))

    def batched_net():
        u = np.maximum(net.W1 @ x + net.b1[:, None], 0)
        v = np.maximum(net.W2 @ u + net.b2[:, None], 0)
        return net.W3 @ v + net.b3[:, None]

    def batched_python():
        phase = (y - BALL_MIN_Y) % (2 * SPAN)
        e = BALL_MIN_Y + np.where(phase <= SPAN, phase, 2 * SPAN - phase) - p
        return np.maximum(e - 4, 0) - np.maximum(-e - 4, 0)

    assert np.allclose(batched_net()[0], batched_python())
    batch_net = min(timeit.repeat(batched_net, number=20, repeat=repeat)) / 20 / len(items) * 1e6
    batch_py = min(timeit.repeat(batched_python, number=20, repeat=repeat)) / 20 / len(items) * 1e6
    return {'single_us': {'network': network_us, 'python': python_us,
                          'ratio': network_us / python_us},
            'batched_numpy_us_per_item': {'network': batch_net, 'python': batch_py,
                                          'ratio': batch_net / batch_py}}


def snake(repeat):
    net = create_network('shortcuts')
    cycle = create_network('cycle')
    states = []
    for seed in (0, 1):
        game = SnakeSimulation(seed=seed, net=net)
        while game.status == 'running':
            if game.steps % 10 == 0:
                states.append((list(game.body), game.food))
            game.step()
    for body, food in states:
        assert np.array_equal(net.forward_state(body, food),
                              shortcut_logits(body, food, 18, 14, net.cycle_ranks))
    items = states[:1500]
    shortcut_net = per_call(net.forward_state, items, repeat)
    encode_only = per_call(net.encode_state, items, repeat)
    shortcut_py = per_call(lambda b, f: shortcut_logits(b, f, 18, 14, net.cycle_ranks), items, repeat)
    heads = [(b[0][0], b[0][1]) for b, _ in items]
    table = {(x, y): int(np.argmax(cycle.forward(x, y))) for x in range(18) for y in range(14)}
    cycle_net = per_call(cycle.forward, heads, repeat)
    cycle_lookup = per_call(lambda x, y: table[(x, y)], heads, repeat)
    return {'states_checked_identical': len(states),
            'shortcuts_us': {'network_total': shortcut_net, 'of_which_encode_and_validate': encode_only,
                             'python_rule': shortcut_py, 'ratio': shortcut_net / shortcut_py},
            'cycle_us': {'network': cycle_net, 'python_lookup_table': cycle_lookup,
                         'ratio': cycle_net / cycle_lookup}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args(argv)
    payload = {'pong': pong(args.repeat), 'snake': snake(args.repeat)}
    print(payload)
    save('e5_timing', payload)


if __name__ == '__main__':
    main()
