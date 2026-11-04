#!/usr/bin/env python3
"""NN Pong: gioco Pygame con fisica continua e controllo ReLU analitico.

Avvio: python pong.py
Test: python pong.py --test
Simulazione: python pong.py --headless --seeds 100 --frames 10000

Spazio: pausa; R: nuova partita; G: diagnostica; +/-: velocita; F: schermo intero.
Implementazione in pong_game; i nomi pubblici storici restano importabili qui.
"""

from __future__ import annotations

import argparse
import random

from pong_game.config import *
from pong_game.network import PongNet, PongNetV5, clamp, reflect_y, wrap_period  # noqa: F401  (public names)
from pong_game.simulation import PongSimulation, Stats  # noqa: F401


def __getattr__(name):
    if name == 'PongApp':
        from pong_game.app import PongApp
        return PongApp
    if name == 'pygame':
        try:
            import pygame
        except ImportError:
            return None
        return pygame
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def test_network(samples=50000):
    net = PongNet()
    rng = random.Random(2026)
    max_fold_err = 0.0
    max_out_err = 0.0
    for _ in range(samples):
        y = rng.uniform(-2 * SPAN + 1, 2.9 * SPAN)
        paddle = rng.uniform(0, HEIGHT)
        out, info = net.forward(paddle, wrap_period(y))
        expected_fold = reflect_y(y)
        expected_out = (max(0, expected_fold - paddle - DEADZONE)
                        - max(0, paddle - expected_fold - DEADZONE))
        max_fold_err = max(max_fold_err, abs(info["fold"] - expected_fold))
        max_out_err = max(max_out_err, abs(out - expected_out))
    assert max_fold_err < 1e-7, max_fold_err
    assert max_out_err < 1e-7, max_out_err
    print(f"PASS rete {net.parameter_count} parametri")
    print(f"PASS fold: {samples} campioni, errore max {max_fold_err:.3e}")
    print(f"PASS output: {samples} campioni, errore max {max_out_err:.3e}")

    for e in (-20, -4.001, -4, -3.999, 0, 3.999, 4, 4.001, 20):
        out, _ = net.forward(200, 200 + e)
        expected = max(0, e - DEADZONE) - max(0, -e - DEADZONE)
        assert abs(out - expected) < 1e-7
    print("PASS deadzone: confini e valori interni/esterni")


def stress_test(seeds=100, frames=10000):
    total_hits = total_misses = total_player = total_opponent = 0
    worst_misses = 0
    for seed in range(seeds):
        sim = PongSimulation(seed=seed)
        for _ in range(frames):
            sim.step()
        s = sim.stats
        total_hits += s.hits
        total_misses += s.misses
        total_player += s.player_score
        total_opponent += s.opponent_score
        worst_misses = max(worst_misses, s.misses)
    print(f"STRESS {seeds} seed x {frames} frame = {seeds*frames:,} frame")
    print(f"hits={total_hits} misses={total_misses} worst_misses/seed={worst_misses}")
    print(f"score aggregato P1-P2: {total_player}-{total_opponent}")
    return total_hits, total_misses


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("deve essere un intero maggiore di zero")
    return number

def main():
    parser = argparse.ArgumentParser(description="NN Pong")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true")
    mode.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=positive_int, default=10000)
    parser.add_argument("--seeds", type=positive_int, default=1)
    args = parser.parse_args()
    if args.test:
        test_network()
    elif args.headless:
        stress_test(args.seeds, args.frames)
    else:
        from pong_game.app import PongApp
        PongApp().run()


if __name__ == "__main__":
    main()
