"""E2 - Trained networks versus analytic weights on the same tasks.

Supervised imitation (behaviour cloning) of the analytic policies with a
NumPy MLP and Adam. This is the most favourable training setting: unlimited
noiseless labels from the exact teacher. We report parameters, held-out error
and closed-loop play.

Pong: regression of the controller output on the reduced domain (same inputs
as the analytic net), including the analytic 2-2-2-1 architecture itself.
Snake: classification of the shortcut network's action from the same 1008
binary planes, then the learned network plays on unseen seeds.
"""

from __future__ import annotations

import argparse

import numpy as np

from pong_game.config import BALL_MIN_Y, HEIGHT, SPAN
from pong_game.simulation import PongSimulation
from snake_game import SnakeSimulation
from snake_game.network import create_network

from .common import Timer, save
from .e1_pong_folds import faster, reference
from .mlp import MLP, cross_entropy, mse

OUT_SCALE = 100.0


class TrainedPongNet:
    def __init__(self, mlp):
        self.mlp = mlp

    def forward(self, paddle_y, y):
        x = np.array([[paddle_y / HEIGHT, (y - BALL_MIN_Y) / (2 * SPAN)]], dtype=np.float32)
        return float(self.mlp.forward(x)[0, 0]) * OUT_SCALE, {'fold': float('nan')}


class TrainedSnakeNet:
    def __init__(self, mlp, encoder):
        self.mlp, self.encoder = mlp, encoder
        self.cols, self.rows = encoder.cols, encoder.rows

    def forward_state(self, body, food):
        x = self.encoder.encode_state(body, food).astype(np.float32)[None]
        return self.mlp.forward(x)[0]


def pong_data(n, rng):
    p = rng.uniform(0, HEIGHT, n)
    y = rng.uniform(BALL_MIN_Y, BALL_MIN_Y + 2 * SPAN, n)
    x = np.stack((p / HEIGHT, (y - BALL_MIN_Y) / (2 * SPAN)), 1).astype(np.float32)
    t = np.array([reference(a, b) for a, b in zip(p, y)], dtype=np.float32)[:, None] / OUT_SCALE
    return x, t, p, y


def pong_misses(net, seeds, frames, factor=1.0):
    misses = hits = 0
    simulation = faster(PongSimulation, factor)
    for seed in range(seeds):
        sim = simulation(seed=seed)
        sim.net = net
        for _ in range(frames):
            sim.step()
        misses += sim.stats.misses
        hits += sim.stats.hits
    return hits, misses


def run_pong(args):
    rng = np.random.default_rng(0)
    x_test, t_test, p_test, y_test = pong_data(40000, rng)
    rows = []
    for sizes in ([2, 2, 2, 1], [2, 8, 1], [2, 32, 1], [2, 32, 32, 1]):
        for init in range(args.inits):
            r = np.random.default_rng(100 + init)
            mlp = MLP(sizes, r)
            with Timer() as timer:
                for it in range(args.pong_steps):
                    x, t, _, _ = pong_data(256, r)
                    lr = 3e-3 * 0.5 * (1 + np.cos(np.pi * it / args.pong_steps))
                    mlp.step(x, mse(t), lr)
            pred = mlp.forward(x_test) * OUT_SCALE
            err = np.abs(pred - t_test * OUT_SCALE)[:, 0]
            ref = t_test[:, 0] * OUT_SCALE
            moving = np.abs(ref) > 1.0
            wrong_dir = float(np.mean(np.sign(pred[moving, 0]) != np.sign(ref[moving])))
            games = {f'speed_x{f:g}': dict(zip(('hits', 'misses'), pong_misses(
                TrainedPongNet(mlp), args.pong_seeds, args.pong_frames, f))) for f in (1.0, 3.0)}
            row = {'sizes': sizes, 'init': init, 'parameters': mlp.parameter_count,
                   'train_seconds': round(timer.seconds, 1),
                   'max_abs_error': float(err.max()), 'rmse': float(np.sqrt(np.mean(err ** 2))),
                   'wrong_direction_rate': wrong_dir, 'games': games}
            rows.append(row)
            print('pong', row)
    analytic = {f'speed_x{f:g}': dict(zip(('hits', 'misses'), pong_misses(
        PongSimulation().net, args.pong_seeds, args.pong_frames, f))) for f in (1.0, 3.0)}
    return {'steps': args.pong_steps, 'seeds': args.pong_seeds, 'frames': args.pong_frames,
            'analytic': {'parameters': 15, 'max_abs_error': 0.0, 'games': analytic},
            'trained': rows}


def snake_states(net, seeds, stride):
    xs, ys = [], []
    for seed in seeds:
        game = SnakeSimulation(seed=seed, net=net)
        while game.status == 'running':
            if game.steps % stride == 0:
                xs.append(net.encode_state(game.body, game.food).astype(np.uint8))
                ys.append(int(np.argmax(game.net.forward_state(game.body, game.food))))
            game.step()
    return np.array(xs), np.array(ys)


def play(net, seeds, max_steps):
    wins = deaths = timeouts = 0
    apples, win_steps = [], []
    for seed in seeds:
        game = SnakeSimulation(seed=seed, net=net)
        for _ in range(max_steps):
            if game.step() != 'running':
                break
        apples.append(game.score)
        if game.status == 'won':
            wins += 1
            win_steps.append(game.steps)
        elif game.status == 'dead':
            deaths += 1
        else:
            timeouts += 1
    return {'games': len(seeds), 'wins': wins, 'deaths': deaths, 'timeouts': timeouts,
            'mean_apples': float(np.mean(apples)), 'max_apples': int(max(apples)),
            'mean_steps_when_won': float(np.mean(win_steps)) if win_steps else None}


def run_snake(args):
    teacher = create_network()
    with Timer() as timer:
        x_train, y_train = snake_states(teacher, range(args.snake_train_games), args.stride)
        x_test, y_test = snake_states(teacher, range(1000, 1010), args.stride)
    print(f'snake data {x_train.shape} train, {x_test.shape} test in {timer.seconds:.0f}s')
    rows = []
    for sizes in ([1008, 64, 4], [1008, 256, 4], [1008, 256, 256, 4]):
        r = np.random.default_rng(7)
        mlp = MLP(sizes, r)
        total = args.snake_epochs * (len(x_train) // 256)
        it = 0
        with Timer() as timer:
            for _ in range(args.snake_epochs):
                order = r.permutation(len(x_train))
                for start in range(0, len(order) - 255, 256):
                    idx = order[start:start + 256]
                    lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * it / total))
                    mlp.step(x_train[idx].astype(np.float32), cross_entropy(y_train[idx]), lr)
                    it += 1
        acc_train = float(np.mean(np.argmax(mlp.forward(x_train.astype(np.float32)), 1) == y_train))
        acc_test = float(np.mean(np.argmax(mlp.forward(x_test.astype(np.float32)), 1) == y_test))
        result = play(TrainedSnakeNet(mlp, teacher), range(2000, 2000 + args.snake_play_games),
                      args.max_steps)
        row = {'sizes': sizes, 'parameters': mlp.parameter_count, 'train_seconds': round(timer.seconds, 1),
               'train_accuracy': acc_train, 'test_accuracy': acc_test, 'play': result}
        rows.append(row)
        print('snake', row)
    analytic = play(teacher, range(2000, 2000 + args.snake_play_games), args.max_steps)
    return {'train_states': int(len(x_train)), 'test_states': int(len(x_test)),
            'epochs': args.snake_epochs, 'stride': args.stride,
            'analytic': {'parameters_stored': teacher.parameter_count,
                         'parameters_dense': teacher.dense_parameter_count, 'play': analytic},
            'trained': rows}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inits', type=int, default=3)
    parser.add_argument('--pong-steps', type=int, default=20000)
    parser.add_argument('--pong-seeds', type=int, default=10)
    parser.add_argument('--pong-frames', type=int, default=5000)
    parser.add_argument('--snake-train-games', type=int, default=40)
    parser.add_argument('--snake-epochs', type=int, default=15)
    parser.add_argument('--snake-play-games', type=int, default=20)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=30000)
    parser.add_argument('--only', choices=('pong', 'snake'))
    args = parser.parse_args(argv)
    payload = {}
    if args.only != 'snake':
        payload['pong'] = run_pong(args)
    if args.only != 'pong':
        payload['snake'] = run_snake(args)
    save('e2_trained' + (f'_{args.only}' if args.only else ''), payload)


if __name__ == '__main__':
    main()
