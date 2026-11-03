#!/usr/bin/env python3
"""Snake Classic con pesi costruiti analiticamente, senza training.

    python snake.py
    python snake.py --test
    python snake.py --headless --seeds 100
    python snake.py --compare --seeds 100
    python snake.py --export-weights weights/snake_shortcuts.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from snake_game import DIRECTIONS, SnakeNet, SnakeSimulation
from snake_game.network import create_network, load_network


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


def parameter_count(net) -> int:
    return net.parameter_count


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


def benchmark(net, seed: int, seeds: int, max_steps: int, validate_order=False) -> dict:
    started = time.perf_counter()
    wins = deaths = timeouts = total_steps = 0
    shortcuts = 0
    scores = []
    for game_seed in range(seed, seed + seeds):
        game = SnakeSimulation(seed=game_seed, cols=net.cols, rows=net.rows, net=net)
        for _ in range(max_steps):
            status = game.step()
            if validate_order and hasattr(net, 'cycle_ranks'):
                indices = np.array([y * net.cols + x for x, y in game.body])
                ranks = net.cycle_ranks[indices]
                backwards = (ranks[0] - ranks) % (net.cols * net.rows)
                if not np.all(np.diff(backwards) > 0):
                    raise ValueError(f'ordine del corpo violato al passo {game.steps}')
            if status != 'running':
                break
        wins += game.status == 'won'
        deaths += game.status == 'dead'
        timeouts += game.status == 'running'
        total_steps += game.steps
        shortcuts += game.shortcuts
        scores.append(game.score)
    return {
        'policy': net.policy_name,
        'games': seeds,
        'first_seed': seed,
        'board': [net.cols, net.rows],
        'parameters': parameter_count(net),
        'dense_parameters': net.dense_parameter_count,
        'wins': wins,
        'deaths': deaths,
        'timeouts': timeouts,
        'min_score': min(scores),
        'max_score': max(scores),
        'mean_score': sum(scores) / seeds,
        'total_steps': total_steps,
        'mean_steps': total_steps / seeds,
        'shortcuts': shortcuts,
        'step_limit': max_steps,
        'runtime_seconds': round(time.perf_counter() - started, 6),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Snake: rete ReLU a pesi analitici')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--headless', action='store_true', help='simula senza finestra')
    mode.add_argument('--test', action='store_true', help='verifica ciclo e una partita completa')
    mode.add_argument('--compare', action='store_true', help='confronta la rete selezionata con il ciclo puro')
    mode.add_argument('--export-weights', type=Path, metavar='FILE', help='salva le matrici in JSON')
    parser.add_argument('--weights', type=Path, metavar='FILE', help='carica matrici da JSON')
    parser.add_argument('--policy', choices=('shortcuts', 'cycle'),
                        help='politica analitica (default: shortcuts)')
    parser.add_argument('--seed', type=int, default=0, help='seed iniziale (default: 0)')
    parser.add_argument('--seeds', type=positive_int, default=1, help='partite headless (default: 1)')
    parser.add_argument('--max-steps', type=positive_int, default=100000,
                        help='limite di mosse per partita (default: 100000)')
    parser.add_argument('--speed', type=speed_value, default=10,
                        help='mosse al secondo nella finestra, 1–120 (default: 10)')
    parser.add_argument('--json', action='store_true', help='risultati headless in JSON')
    args = parser.parse_args(argv)
    if args.json and not (args.headless or args.compare):
        parser.error('--json richiede --headless oppure --compare')
    if args.weights and args.policy:
        parser.error('--weights e --policy sono alternativi')

    try:
        net = load_network(args.weights) if args.weights else create_network(args.policy or 'shortcuts')
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
            reference = net if net.policy_name == 'cycle' else SnakeNet(net.cols, net.rows)
            cells = verify_cycle(reference)
        except ValueError as error:
            print(f'FAIL ciclo: {error}')
            return 1
        print(f'PASS ciclo di riferimento: {cells} celle, tutte adiacenti, nessuna ripetizione')
        try:
            report = benchmark(net, args.seed, 1, args.max_steps, validate_order=True)
        except ValueError as error:
            print(f'FAIL politica: {error}')
            return 1
        if report['wins'] != 1:
            print(f'FAIL partita: {report}')
            return 1
        print(f"PASS partita: campo pieno, {report['max_score']} mele, "
              f"{report['total_steps']} mosse, {report['shortcuts']} scorciatoie")
        print(f'PASS pesi: {parameter_count(net)} coefficienti memorizzati, politica {net.policy_name}')
        return 0

    if args.compare:
        baseline = benchmark(SnakeNet(net.cols, net.rows), args.seed, args.seeds, args.max_steps)
        candidate = benchmark(net, args.seed, args.seeds, args.max_steps)
        complete = baseline['wins'] == candidate['wins'] == args.seeds
        reduction = (100 * (1 - candidate['total_steps'] / baseline['total_steps'])
                     if complete else None)
        comparison = {'baseline': baseline, 'candidate': candidate,
                      'move_reduction_pct': reduction, 'all_completed': complete}
        if args.json:
            print(json.dumps(comparison))
        else:
            print(f"Confronto su {args.seeds} seed da {args.seed}:")
            for report in (baseline, candidate):
                print(f"{report['policy']}: vittorie={report['wins']}/{args.seeds}, "
                      f"collisioni={report['deaths']}, timeout={report['timeouts']}, "
                      f"mosse medie={report['mean_steps']:.2f}")
            if reduction is None:
                print('Riduzione non calcolata: una o più partite non sono state completate.')
            else:
                print(f'Riduzione delle mosse: {reduction:.2f}%')
        return 0 if complete else 1

    if args.headless:
        report = benchmark(net, args.seed, args.seeds, args.max_steps)
        if args.json:
            print(json.dumps(report))
        else:
            print(f'SNAKE {args.seeds} seed da {args.seed} | griglia {net.cols}x{net.rows} '
                  f'| {net.policy_name} | {report["parameters"]} coefficienti memorizzati')
            print(f"vittorie={report['wins']} collisioni={report['deaths']} "
                  f"timeout={report['timeouts']}")
            print(f"mele min/media/max: {report['min_score']}/"
                  f"{report['mean_score']:.1f}/{report['max_score']}")
            print(f"mosse totali={report['total_steps']:,} media={report['mean_steps']:.1f}")
            print(f"scorciatoie={report['shortcuts']} tempo di esecuzione={report['runtime_seconds']:.2f}s")
        return 0 if report['wins'] == args.seeds else 1

    if (net.cols, net.rows) != (18, 14):
        parser.error('la finestra riproduce il campo 18x14; usa --headless per altre dimensioni')
    from snake_game.app import SnakeApp
    record_path = Path.home() / '.nn-pong' / 'snake-record.json'
    SnakeApp(seed=args.seed, net=net, speed=args.speed, record_path=record_path).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
