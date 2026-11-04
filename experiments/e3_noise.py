"""E3 - Robustness of analytic and trained weights to perturbation and precision.

Relative noise: every stored coefficient w becomes w * (1 + sigma * N(0, 1)),
so structural zeros stay zero and sigma reads as a relative precision. For each
sigma we measure decision agreement with the clean network on fixed states and
closed-loop play. A precision test rounds weights *and* every layer's output to
float16 or bfloat16, as a low-precision accelerator would.

Pong is played at three times the serve speed: at normal speed even a paddle
with no prediction never misses (see E1), so misses there measure nothing.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from pong_game.config import BALL_MIN_Y, HEIGHT, SPAN
from pong_game.network import PongNet
from snake_game import SnakeSimulation
from snake_game.network import create_network
from snake_game.shortcut_network import SparseLayer

from .e1_pong_folds import reference
from .e2_trained import TrainedPongNet, pong_data, pong_misses
from .mlp import MLP, mse

SIGMAS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def jitter(array, sigma, rng):
    return array * (1 + sigma * rng.standard_normal(array.shape))


def bf16(array):
    bits = np.asarray(array, dtype=np.float32).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


ROUND = {'float64': lambda a: np.asarray(a, dtype=np.float64),
         'float32': lambda a: np.asarray(a, dtype=np.float32).astype(np.float64),
         'float16': lambda a: np.asarray(a, dtype=np.float16).astype(np.float64),
         'bfloat16': bf16}


# ---------------------------------------------------------------- Pong

class LowPrecisionPongNet(PongNet):
    def __init__(self, rounding):
        super().__init__()
        self.r = rounding
        for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            setattr(self, name, self.r(getattr(self, name)))

    def forward(self, paddle_y, y_pred):
        r = self.r
        x = r(np.array([paddle_y, y_pred]))
        u = r(np.maximum(r(self.W1 @ x + self.b1), 0))
        v = r(np.maximum(r(self.W2 @ u + self.b2), 0))
        return float(r(self.W3 @ v + self.b3)[0]), {'fold': float('nan')}


def noisy_pong(sigma, rng):
    net = PongNet()
    net.forward = PongNet.forward.__get__(net)
    for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        setattr(net, name, jitter(getattr(net, name), sigma, rng))
    return net


def pong_grid_error(net):
    ps = np.linspace(0, HEIGHT, 60)
    ys = np.linspace(BALL_MIN_Y, BALL_MIN_Y + 2 * SPAN, 120)
    return max(abs(net.forward(p, y)[0] - reference(p, y)) for p in ps for y in ys)


def trained_pong(steps):
    rng = np.random.default_rng(100)
    mlp = MLP([2, 32, 32, 1], rng)
    for it in range(steps):
        x, t, _, _ = pong_data(256, rng)
        mlp.step(x, mse(t), 3e-3 * 0.5 * (1 + np.cos(np.pi * it / steps)))
    return mlp


def noisy_mlp(mlp, sigma, rng):
    clone = copy.deepcopy(mlp)
    clone.W = [jitter(w, sigma, rng).astype(np.float32) for w in mlp.W]
    clone.b = [jitter(b, sigma, rng).astype(np.float32) for b in mlp.b]
    return clone


def run_pong(args):
    mlp = trained_pong(args.pong_train_steps)
    rows = []
    for sigma in SIGMAS:
        for kind in ('analytic_15', 'trained_1185'):
            errors, misses, hits = [], 0, 0
            for draw in range(args.draws):
                rng = np.random.default_rng(1000 * draw + 7)
                net = noisy_pong(sigma, rng) if kind == 'analytic_15' else TrainedPongNet(
                    noisy_mlp(mlp, sigma, rng))
                errors.append(pong_grid_error(net))
                h, m = pong_misses(net, args.pong_seeds, args.pong_frames, 3.0)
                hits, misses = hits + h, misses + m
            row = {'sigma': sigma, 'net': kind, 'max_abs_output_error': float(max(errors)),
                   'hits_x3': hits, 'misses_x3': misses}
            rows.append(row)
            print('pong', row)
    precision = {}
    for name, rounding in ROUND.items():
        net = LowPrecisionPongNet(rounding)
        precision[name] = {'max_abs_output_error': float(pong_grid_error(net)),
                           'misses_x3': pong_misses(net, args.pong_seeds, args.pong_frames, 3.0)[1]}
        print('pong precision', name, precision[name])
    return {'noise': rows, 'precision': precision}


# ---------------------------------------------------------------- Snake

def noisy_shortcut(base, sigma, rng):
    net = copy.copy(base)
    net.layers = [SparseLayer(layer.src, layer.dst, jitter(layer.weights, sigma, rng),
                              jitter(layer.bias, sigma, rng), layer.input_size, layer.output_size)
                  for layer in base.layers]
    return net


def noisy_cycle(base, sigma, rng):
    net = copy.deepcopy(base)
    for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        setattr(net, name, jitter(getattr(base, name), sigma, rng))
    return net


class LowPrecisionShortcut:
    def __init__(self, base, rounding):
        self.base, self.r = base, rounding
        self.cols, self.rows = base.cols, base.rows
        self.layers = [SparseLayer(layer.src, layer.dst, rounding(layer.weights), rounding(layer.bias),
                                   layer.input_size, layer.output_size) for layer in base.layers]

    def forward_state(self, body, food):
        values = self.base.encode_state(body, food)
        for layer in self.layers[:-1]:
            values = self.r(np.maximum(self.r(layer.forward(values)), 0))
        return self.r(self.layers[-1].forward(values))


def agreement(net, reference_net, states):
    same = 0
    for body, food in states:
        same += int(np.argmax(net.forward_state(body, food))
                    == np.argmax(reference_net.forward_state(body, food)))
    return same / len(states)


def sample_states(net, seeds, stride):
    states = []
    for seed in seeds:
        game = SnakeSimulation(seed=seed, net=net)
        while game.status == 'running':
            if game.steps % stride == 0:
                states.append((list(game.body), game.food))
            game.step()
    return states


def play(net, seeds, max_steps):
    wins = deaths = 0
    apples = []
    for seed in seeds:
        game = SnakeSimulation(seed=seed, net=net)
        for _ in range(max_steps):
            if game.step() != 'running':
                break
        wins += game.status == 'won'
        deaths += game.status == 'dead'
        apples.append(game.score)
    return {'games': len(seeds), 'wins': int(wins), 'deaths': int(deaths),
            'mean_apples': float(np.mean(apples))}


def run_snake(args):
    rows = []
    for policy, perturb, max_steps in (('shortcuts', noisy_shortcut, 30000),
                                       ('cycle', noisy_cycle, 40000)):
        base = create_network(policy)
        states = sample_states(create_network('shortcuts'), [1000, 1001], 25)
        for sigma in SIGMAS:
            rng = np.random.default_rng(int(sigma * 1e9) % 2**32 + 3)
            net = perturb(base, sigma, rng)
            row = {'policy': policy, 'sigma': sigma,
                   'decision_agreement': agreement(net, base, states),
                   'play': play(net, range(3000, 3000 + args.snake_games), max_steps)}
            rows.append(row)
            print('snake', row)
    shortcut = create_network('shortcuts')
    states = sample_states(shortcut, [1000, 1001], 25)
    precision = {}
    for name, rounding in ROUND.items():
        net = LowPrecisionShortcut(shortcut, rounding)
        precision[name] = {'decision_agreement': agreement(net, shortcut, states),
                           'play': play(net, range(3000, 3000 + args.snake_games), 30000)}
        print('snake precision', name, precision[name])
    return {'states': len(states), 'noise': rows, 'precision_shortcuts': precision}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--pong-seeds', type=int, default=10)
    parser.add_argument('--pong-frames', type=int, default=3000)
    parser.add_argument('--pong-train-steps', type=int, default=20000)
    parser.add_argument('--snake-games', type=int, default=5)
    parser.add_argument('--only', choices=('pong', 'snake'))
    args = parser.parse_args(argv)
    from .common import save
    payload = {'sigmas': SIGMAS}
    if args.only != 'snake':
        payload['pong'] = run_pong(args)
    if args.only != 'pong':
        payload['snake'] = run_snake(args)
    save('e3_noise' + (f'_{args.only}' if args.only else ''), payload)


if __name__ == '__main__':
    main()
