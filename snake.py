#!/usr/bin/env python3
"""Snake Classic con pesi costruiti analiticamente, senza training.

    python snake.py
    python snake.py --test
    python snake.py --headless --seeds 100
    python snake.py --export-weights weights/snake_classic.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from snake_game import DIRECTIONS, SnakeNet, SnakeSimulation


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('richiede un intero maggiore di zero')
    return result


def speed_value(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 1 <= result <= 120:
        raise argparse.ArgumentTypeError('richiede un numero finito tra 1 e 120')
    return result


def parameter_count(net: SnakeNet) -> int:
    return sum(getattr(net, name).size for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))


def verify_cycle(net: SnakeNet) -> int:
    """Verifica le mosse effettivamente prodotte dalle matrici su ogni cella."""
    current = (0, 0)
    visited = set()
    for _ in range(net.cols * net.rows):
        x, y = current
        if not (0 <= x < net.cols and 0 <= y < net.rows):
            raise ValueError(f'la rete esce dal campo in {current}')
        if current in visited:
            raise ValueError(f'la rete chiude il giro troppo presto in {current}')
        visited.add(current)
        logits = net.forward(x, y)
        action = int(np.argmax(logits))
        dx, dy = DIRECTIONS[action]
        current = (x + dx, y + dy)
    if current != (0, 0):
        raise ValueError('il percorso non si richiude dopo aver visitato il campo')
    return len(visited)


def benchmark(net: SnakeNet, seed: int, seeds: int, max_steps: int) -> dict:
    wins = deaths = timeouts = total_steps = 0
    scores = []
    for game_seed in range(seed, seed + seeds):
        game = SnakeSimulation(seed=game_seed, cols=net.cols, rows=net.rows, net=net)
        for _ in range(max_steps):
            if game.step() != 'running':
                break
        wins += game.status == 'won'
        deaths += game.status == 'dead'
        timeouts += game.status == 'running'
        total_steps += game.steps
        scores.append(game.score)
    return {
        'games': seeds,
        'first_seed': seed,
        'board': [net.cols, net.rows],
        'parameters': parameter_count(net),
        'wins': wins,
        'deaths': deaths,
        'timeouts': timeouts,
        'min_score': min(scores),
        'max_score': max(scores),
        'mean_score': sum(scores) / seeds,
        'total_steps': total_steps,
        'mean_steps': total_steps / seeds,
        'step_limit': max_steps,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Snake: rete ReLU a pesi analitici')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--headless', action='store_true', help='simula senza finestra')
    mode.add_argument('--test', action='store_true', help='verifica ciclo e una partita completa')
    mode.add_argument('--export-weights', type=Path, metavar='FILE', help='salva le matrici in JSON')
    parser.add_argument('--weights', type=Path, metavar='FILE', help='carica matrici da JSON')
    parser.add_argument('--seed', type=int, default=0, help='seed iniziale (default: 0)')
    parser.add_argument('--seeds', type=positive_int, default=1, help='partite headless (default: 1)')
    parser.add_argument('--max-steps', type=positive_int, default=100000,
                        help='limite di mosse per partita (default: 100000)')
    parser.add_argument('--speed', type=speed_value, default=10,
                        help='mosse al secondo nella finestra, 1–120 (default: 10)')
    parser.add_argument('--json', action='store_true', help='risultati headless in JSON')
    args = parser.parse_args(argv)
    if args.json and not args.headless:
        parser.error('--json richiede --headless')

    try:
        net = SnakeNet.load(args.weights) if args.weights else SnakeNet()
    except (OSError, ValueError, TypeError) as error:
        parser.error(f'impossibile caricare i pesi: {error}')

    if args.export_weights:
        try:
            net.save(args.export_weights)
        except OSError as error:
            parser.error(f'impossibile salvare i pesi: {error}')
        print(f'Pesi salvati: {args.export_weights} ({parameter_count(net)} parametri)')
        return 0

    if args.test:
        try:
            cells = verify_cycle(net)
        except ValueError as error:
            print(f'FAIL ciclo: {error}')
            return 1
        print(f'PASS ciclo: {cells} celle, tutte adiacenti, nessuna ripetizione')
        report = benchmark(net, args.seed, 1, args.max_steps)
        if report['wins'] != 1:
            print(f'FAIL partita: {report}')
            return 1
        print(f"PASS partita: campo pieno, {report['max_score']} mele, "
              f"{report['total_steps']} mosse")
        print(f'PASS pesi: {parameter_count(net)} parametri fissi, nessun training')
        return 0

    if args.headless:
        report = benchmark(net, args.seed, args.seeds, args.max_steps)
        if args.json:
            print(json.dumps(report))
        else:
            print(f'SNAKE {args.seeds} seed da {args.seed} | griglia {net.cols}x{net.rows} '
                  f'| {report["parameters"]} parametri')
            print(f"vittorie={report['wins']} collisioni={report['deaths']} "
                  f"timeout={report['timeouts']}")
            print(f"mele min/media/max: {report['min_score']}/"
                  f"{report['mean_score']:.1f}/{report['max_score']}")
            print(f"mosse totali={report['total_steps']:,} media={report['mean_steps']:.1f}")
        return 0 if report['wins'] == args.seeds else 1

    if (net.cols, net.rows) != (18, 14):
        parser.error('la finestra riproduce il campo 18x14; usa --headless per altre dimensioni')
    from snake_game.app import SnakeApp
    record_path = Path.home() / '.nn-pong' / 'snake-record.json'
    SnakeApp(seed=args.seed, net=net, speed=args.speed, record_path=record_path).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
