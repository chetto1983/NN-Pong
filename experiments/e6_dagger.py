"""E6 - DAgger for Snake: does fixing behaviour cloning's distribution shift close the gap?

E2 cloned the analytic shortcut network on states from *its own* games and the
clone never finished a game: small errors lead it to states the teacher never
visits. DAgger (Ross, Gordon & Bagnell, 2011) iterates:

1. play with a mixture of teacher and learner (beta_i = 0.5**i teacher share,
   chosen per step), cap and deduplicate the visited states;
2. label every visited state with the teacher's action;
3. aggregate into the dataset and keep training the same learner.

Inputs, teacher and test games are the same as in E2. States are stored
bit-packed (126 bytes each). Each iteration is evaluated on 20 unseen seeds
with the pure learner, plus its on-policy agreement with the teacher.
"""

from __future__ import annotations

import argparse

import numpy as np

from snake_game import SnakeSimulation
from snake_game.network import create_network

from .common import Timer, save
from .e2_trained import TrainedSnakeNet, play, snake_states
from .mlp import MLP, cross_entropy


def pack(states):
    return np.packbits(states.astype(np.uint8), axis=1)


def unpack(packed, size):
    return np.unpackbits(packed, axis=1, count=size).astype(np.float32)


def rollouts(teacher, learner, seeds, beta, max_steps, rng):
    """Visited states (deduplicated) labelled by the teacher, plus on-policy agreement."""
    seen, xs, ys = set(), [], []
    agree = total = 0
    for seed in seeds:
        game = SnakeSimulation(seed=seed, net=teacher)
        loop_guard, score = set(), 0
        for _ in range(max_steps):
            if game.score != score:
                loop_guard.clear()
                score = game.score
            key = (tuple(game.body), game.food)
            if key in loop_guard:
                break
            loop_guard.add(key)
            x = teacher.encode_state(game.body, game.food)
            label = int(np.argmax(teacher.forward_state(game.body, game.food)))
            guess = int(np.argmax(learner.mlp.forward(x.astype(np.float32)[None])[0]))
            agree += guess == label
            total += 1
            if key not in seen:
                seen.add(key)
                xs.append(x)
                ys.append(label)
            action = label if rng.random() < beta else guess
            if game.step(action) != 'running':
                break
    return pack(np.array(xs)), np.array(ys), agree / max(1, total)


def train(mlp, packed, labels, size, epochs, lr, rng):
    for epoch in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(order) - 255, 256):
            idx = order[start:start + 256]
            mlp.step(unpack(packed[idx], size), cross_entropy(labels[idx]), lr * (0.7 ** epoch))


def accuracy(mlp, packed, labels, size):
    correct = 0
    for start in range(0, len(labels), 4096):
        out = mlp.forward(unpack(packed[start:start + 4096], size))
        correct += int(np.sum(np.argmax(out, 1) == labels[start:start + 4096]))
    return correct / len(labels)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--initial-games', type=int, default=40)
    parser.add_argument('--games-per-iteration', type=int, default=20)
    parser.add_argument('--rollout-steps', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--play-games', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=30000)
    args = parser.parse_args(argv)

    teacher = create_network()
    size = 4 * teacher.cols * teacher.rows
    rng = np.random.default_rng(11)
    with Timer() as timer:
        x0, y0 = snake_states(teacher, range(args.initial_games), 3)
        x_test, y_test = snake_states(teacher, range(1000, 1010), 3)
    packed, labels = pack(x0), y0
    test_packed = pack(x_test)
    print(f'initial expert data {len(labels)} states, test {len(y_test)} ({timer.seconds:.0f}s)')

    mlp = MLP([size, *args.hidden, 4], rng)
    learner = TrainedSnakeNet(mlp, teacher)
    history = []
    next_seed = 5000
    for iteration in range(args.iterations + 1):
        with Timer() as fit:
            train(mlp, packed, labels, size, args.epochs if iteration else 3 * args.epochs, 1e-3, rng)
        result = play(learner, range(2000, 2000 + args.play_games), args.max_steps)
        row = {'iteration': iteration, 'dataset_states': int(len(labels)),
               'train_seconds': round(fit.seconds, 1),
               'expert_state_test_accuracy': accuracy(mlp, test_packed, y_test, size), 'play': result}
        if iteration < args.iterations:
            beta = 0.5 ** (iteration + 1)
            seeds = range(next_seed, next_seed + args.games_per_iteration)
            next_seed += args.games_per_iteration
            with Timer() as roll:
                new_x, new_y, on_policy = rollouts(teacher, learner, seeds, beta,
                                                   args.rollout_steps, rng)
            packed = np.concatenate((packed, new_x))
            labels = np.concatenate((labels, new_y))
            row.update({'next_beta': beta, 'new_states': int(len(new_y)),
                        'on_policy_agreement': on_policy, 'rollout_seconds': round(roll.seconds, 1)})
        history.append(row)
        print(row, flush=True)
    save('e6_dagger', {'architecture': [size, *args.hidden, 4], 'parameters': mlp.parameter_count,
                       'teacher_parameters_stored': teacher.parameter_count,
                       'settings': vars(args), 'iterations': history})


if __name__ == '__main__':
    main()
