"""E4 - Snake against external references and on other board sizes.

* Safe greedy BFS (path to food only if the tail stays reachable, else follow
  the tail): a standard strong heuristic with no completion guarantee.
* Manhattan lower bound on the *realized* food sequence: for each apple, the
  distance from the head when it appeared. moves / bound measures detours on
  that sequence; it is not an optimality gap, because a different policy would
  see different apples.
* Moves per apple split at half occupancy, where the shortcut network is
  forced back onto the cycle.
* Completion and move reduction of the shortcut network versus the cycle
  baseline for several board sizes.
"""

from __future__ import annotations

import argparse

import numpy as np

from snake_game import SnakeSimulation
from snake_game.network import create_network

from .common import Timer, save
from .policies import greedy_action


def run_game(game, choose, max_steps):
    bound = abs(game.body[0][0] - game.food[0]) + abs(game.body[0][1] - game.food[1])
    capacity = game.cols * game.rows
    half_step = None
    for _ in range(max_steps):
        score = game.score
        game.step(choose(game) if choose else None)
        if game.status != 'running':
            break
        if game.score != score:
            head = game.body[0]
            bound += abs(head[0] - game.food[0]) + abs(head[1] - game.food[1])
            if half_step is None and len(game.body) >= capacity // 2:
                half_step = (game.steps, game.score)
    return {'status': game.status, 'apples': game.score, 'steps': game.steps,
            'manhattan_bound': bound, 'half': half_step}


def summary(games):
    won = [g for g in games if g['status'] == 'won']
    out = {'games': len(games), 'wins': len(won),
           'deaths': sum(g['status'] == 'dead' for g in games),
           'timeouts': sum(g['status'] == 'running' for g in games),
           'mean_apples': float(np.mean([g['apples'] for g in games])),
           'min_apples': int(min(g['apples'] for g in games)),
           'mean_steps_per_apple': float(np.mean([g['steps'] / max(1, g['apples']) for g in games])),
           'mean_moves_over_manhattan_bound': float(np.mean(
               [g['steps'] / max(1, g['manhattan_bound']) for g in games]))}
    if won:
        out['mean_steps_when_won'] = float(np.mean([g['steps'] for g in won]))
    halves = [g for g in won if g['half']]
    if halves:
        first = [g['half'][0] / g['half'][1] for g in halves]
        second = [(g['steps'] - g['half'][0]) / (g['apples'] - g['half'][1]) for g in halves]
        out['steps_per_apple_first_half'] = float(np.mean(first))
        out['steps_per_apple_second_half'] = float(np.mean(second))
        out['share_of_steps_in_second_half'] = float(np.mean(
            [(g['steps'] - g['half'][0]) / g['steps'] for g in halves]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--size-seeds', type=int, default=10)
    args = parser.parse_args(argv)

    seeds = range(args.seeds)
    policies = {}
    for name in ('shortcuts', 'cycle', 'greedy_bfs'):
        net = create_network('shortcuts' if name == 'greedy_bfs' else name)

        def choose(game, name=name):
            return greedy_action(game.body, game.food, game.cols, game.rows) if name == 'greedy_bfs' else None

        with Timer() as timer:
            limit = 30000 if name == 'greedy_bfs' else 100000
            games = [run_game(SnakeSimulation(seed=s, net=net), choose, limit) for s in seeds]
        policies[name] = {**summary(games), 'seconds': round(timer.seconds, 1)}
        print(name, policies[name])

    sizes = []
    for cols, rows in ((6, 4), (8, 6), (10, 8), (12, 10), (14, 12), (18, 14), (24, 18)):
        capacity = cols * rows
        row = {'cols': cols, 'rows': rows, 'cells': capacity}
        for policy in ('cycle', 'shortcuts'):
            net = create_network(policy, cols, rows)
            games = [run_game(SnakeSimulation(seed=s, cols=cols, rows=rows, net=net), None,
                              2 * capacity * capacity) for s in range(args.size_seeds)]
            row[policy] = summary(games)
            if policy == 'shortcuts':
                row['stored_parameters'] = net.parameter_count
                row['dense_parameters'] = net.dense_parameter_count
        row['move_reduction'] = 1 - (row['shortcuts'].get('mean_steps_when_won', np.nan)
                                     / row['cycle'].get('mean_steps_when_won', np.nan))
        sizes.append(row)
        print({k: row[k] for k in ('cols', 'rows', 'move_reduction')},
              row['shortcuts']['wins'], row['cycle']['wins'])
    save('e4_snake_references', {'seeds': args.seeds, 'size_seeds': args.size_seeds,
                                 'policies_18x14': policies, 'board_sizes': sizes})


if __name__ == '__main__':
    main()
