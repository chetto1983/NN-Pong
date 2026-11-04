"""E1 - Moving the bounce reduction from Python into the Pong network.

The shipped network receives y already reduced modulo 2*SPAN. Here a family
of exact networks folds the *raw* intercept itself: each wall reflection the
network must undo costs one ReLU hinge (5 parameters). We measure

* how many reflections real rallies need (distribution of raw intercepts);
* parameters versus coverage of those intercepts;
* in-game misses when the Python modulo is removed and the network sees the
  raw intercept (clamped to its domain when it falls outside).
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from pong_game.config import BALL_MIN_Y, DEADZONE, HEIGHT, SPAN
from pong_game.network import reflect_y
from pong_game.simulation import PongSimulation

from .common import save


class MultiFoldNet:
    """Exact 2 -> H -> 2 -> 1 fold of raw y on [BALL_MIN_Y - left*SPAN, BALL_MIN_Y + right*SPAN].

    ``left`` must be even (the triangle wave rises at that point). Hidden units:
    one always-active carrier of y - paddle and one hinge per interior breakpoint.
    """

    def __init__(self, left, right, delta=DEADZONE):
        if left % 2 or left < 0 or right < 1:
            raise ValueError('left must be even and nonnegative, right positive')
        self.lo = BALL_MIN_Y - left * SPAN
        self.hi = BALL_MIN_Y + right * SPAN
        self.carrier = float(HEIGHT) - self.lo
        ks = list(range(-left + 1, right))
        hidden = 1 + len(ks)
        self.W1 = np.zeros((hidden, 2))
        self.b1 = np.zeros(hidden)
        self.W1[0] = (-1.0, 1.0)
        self.b1[0] = self.carrier
        coeff = np.zeros(hidden)
        coeff[0] = 1.0
        for i, k in enumerate(ks, start=1):
            self.W1[i] = (0.0, 1.0)
            self.b1[i] = -(BALL_MIN_Y + k * SPAN)
            coeff[i] = -2.0 if k % 2 else 2.0
        # e = fold(y) - paddle = carrier_unit + sum(coeff * hinges) - carrier - (lo - BALL_MIN_Y)
        offset = -self.carrier - (self.lo - BALL_MIN_Y)
        self.W2 = np.vstack((coeff, -coeff))
        self.b2 = np.array([offset - delta, -offset - delta])
        self.W3 = np.array([[1.0, -1.0]])
        self.b3 = np.array([0.0])
        self.hidden = hidden

    @property
    def parameter_count(self):
        return sum(getattr(self, n).size for n in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))

    def covers(self, y):
        return self.lo <= y <= self.hi

    def forward(self, paddle_y, y):
        x = np.array([paddle_y, min(self.hi, max(self.lo, y))])
        u = np.maximum(self.W1 @ x + self.b1, 0)
        v = np.maximum(self.W2 @ u + self.b2, 0)
        out = float((self.W3 @ v + self.b3)[0])
        return out, {'fold': out}


class RawInputSimulation(PongSimulation):
    """The Python modulo is removed: the network sees the raw intercept."""

    def predict_player_intercept(self):
        super().predict_player_intercept()
        return self.raw_prediction


class TrackingSimulation(PongSimulation):
    """Control: no prediction at all, the paddle follows the current ball_y."""

    def predict_player_intercept(self):
        super().predict_player_intercept()
        return self.ball_y if self.ball_vx < 0 else HEIGHT / 2.0


def faster(simulation, factor):
    """Same controller, every serve multiplied in speed by ``factor``."""

    class Faster(simulation):
        def serve(self, direction):
            super().serve(direction)
            self.ball_vx *= factor
            self.ball_vy *= factor

    return Faster


def reference(paddle, raw):
    e = reflect_y(raw) - paddle
    return max(0, e - DEADZONE) - max(0, -e - DEADZONE)


def reflections(raw):
    """Wall reflections between the ball and the paddle plane for a raw intercept."""
    return int(math.floor((raw - BALL_MIN_Y) / SPAN)) if raw >= BALL_MIN_Y else int(
        math.floor((BALL_MIN_Y - raw) / SPAN)) + 1


def collect_raw(seeds, frames):
    raws = []
    for seed in range(seeds):
        sim = PongSimulation(seed=seed)
        for _ in range(frames):
            sim.step()
            if sim.ball_vx < 0:
                raws.append(sim.raw_prediction)
    return np.array(raws)


def stress(net, seeds, frames, simulation=RawInputSimulation):
    hits = misses = 0
    for seed in range(seeds):
        sim = simulation(seed=seed)
        sim.net = net
        for _ in range(frames):
            sim.step()
        hits += sim.stats.hits
        misses += sim.stats.misses
    return hits, misses


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=50)
    parser.add_argument('--frames', type=int, default=10000)
    parser.add_argument('--sweep-seeds', type=int, default=30)
    parser.add_argument('--sweep-frames', type=int, default=5000)
    parser.add_argument('--speed-factors', type=float, nargs='+', default=[1.0, 2.0, 3.0, 4.0, 5.0])
    args = parser.parse_args(argv)

    raws = collect_raw(args.seeds, args.frames)
    counts = np.array([reflections(r) for r in raws])
    hist = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    print(f'{len(raws)} control steps with incoming ball; reflections histogram {hist}')
    print(f'raw intercept range [{raws.min():.0f}, {raws.max():.0f}]')

    rng = np.random.default_rng(0)
    rows = []
    for left, right in ((0, 2), (0, 3), (2, 3), (2, 4), (4, 5), (4, 6)):
        net = MultiFoldNet(left, right)
        # Exactness on the whole covered domain.
        ys = rng.uniform(net.lo, net.hi, 20000)
        ps = rng.uniform(0, HEIGHT, 20000)
        err = max(abs(net.forward(p, y)[0] - reference(p, y)) for p, y in zip(ps, ys))
        coverage = float(np.mean((raws >= net.lo) & (raws <= net.hi)))
        hits, misses = stress(net, args.seeds, args.frames)
        row = {'left_periods': left, 'right_periods': right, 'hidden': net.hidden,
               'parameters': net.parameter_count, 'max_abs_error_on_domain': float(err),
               'coverage_of_real_intercepts': coverage, 'hits': hits, 'misses': misses}
        rows.append(row)
        print(row)
    # Difficulty sweep: where does prediction (and the modulo) start to matter?
    shipped = PongSimulation().net
    controllers = (
        ('tracking_ball_y_no_prediction', shipped, TrackingSimulation),
        ('raw_intercept_15_params', MultiFoldNet(0, 2), RawInputSimulation),
        ('raw_intercept_55_params', MultiFoldNet(4, 6), RawInputSimulation),
        ('python_modulo_15_params_shipped', shipped, PongSimulation),
    )
    sweep = []
    for factor in args.speed_factors:
        for name, net, simulation in controllers:
            hits, misses = stress(net, args.sweep_seeds, args.sweep_frames, faster(simulation, factor))
            row = {'speed_factor': factor, 'controller': name, 'hits': hits, 'misses': misses,
                   'miss_rate': misses / max(1, hits + misses)}
            sweep.append(row)
            print(row)
    save('e1_pong_folds', {
        'sweep_seeds': args.sweep_seeds, 'sweep_frames': args.sweep_frames,
        'difficulty_sweep': sweep,
        'seeds': args.seeds, 'frames': args.frames,
        'incoming_control_steps': int(len(raws)),
        'raw_intercept_min': float(raws.min()), 'raw_intercept_max': float(raws.max()),
        'reflections_histogram': hist,
        'networks': rows,
    })


if __name__ == '__main__':
    main()
