#!/usr/bin/env python3
"""
Pong V5 — Weight-Space Programming
==================================
Gioco Pong Pygame con rete ReLU a pesi fissi (nessun training).

Installazione:
    python -m pip install -r requirements.txt

Avvio:
    python pong.py
    python pong.py --test
    python pong.py --headless --frames 10000
    python pong.py --headless --seeds 100 --frames 10000

Controlli:
    ESC      esci
    R        nuova partita
    SPACE    pausa
    G        mostra/nascondi grafico
    +/-      velocità palla

Nota tecnica:
La rete implementa il comparatore su fold(y_pred) e paddle_y.
Il fold è espresso come combinazione lineare delle attivazioni ReLU
del primo layer; il readout diagnostico non influenza l'azione.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


# ------------------------------- Configurazione -------------------------------

WIDTH, HEIGHT = 1000, 650
FPS = 60
TOP = 7.0
BOTTOM = HEIGHT - 7.0

PADDLE_W, PADDLE_H = 14, 104
PADDLE_MARGIN = 34
BALL_R = 9
BALL_MIN_Y = TOP + BALL_R
BALL_MAX_Y = BOTTOM - BALL_R
SPAN = BALL_MAX_Y - BALL_MIN_Y
PADDLE_SPEED = 7.0
BALL_SPEED = 6.0
MAX_BALL_SPEED = 12.0
DEADZONE = 4.0

BG = (8, 10, 17)
FIELD = (17, 22, 33)
WHITE = (238, 242, 248)
MUTED = (120, 135, 155)
GREEN = (35, 245, 130)
YELLOW = (255, 220, 55)
CYAN = (35, 220, 245)
RED = (255, 75, 90)
PANEL = (15, 20, 31)
GRID = (34, 43, 59)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def reflect_y(y: float) -> float:
    """Ripiega il centro palla tra le pareti, considerando il suo raggio."""
    phase = (y - BALL_MIN_Y) % (2.0 * SPAN)
    return BALL_MIN_Y + (phase if phase <= SPAN else 2.0 * SPAN - phase)


# -------------------------------- Rete neurale --------------------------------

class PongNetV5:
    """
    Rete 2 -> 8 -> 2 -> 1 con pesi fissati analiticamente.

    Input: [paddle_y, y_pred]
    Output positivo: muovi verso il basso.
    Output negativo: muovi verso l'alto.

    Il primo layer costruisce gli hinge ReLU del fold triangolare.
    Il secondo layer calcola ReLU(error-delta) e ReLU(-error-delta).
    Il terzo layer sottrae le due attivazioni.
    """

    def __init__(self, delta: float = DEADZONE):
        if delta < 0:
            raise ValueError("delta deve essere non negativo")
        self.delta = float(delta)

        self.W1 = np.array([
            [0,  1], [0, -1],
            [0,  1], [0, -1],
            [0,  1], [0, -1],
            [1,  0], [-1, 0],
        ], dtype=float)
        self.b1 = np.array([
            -BALL_MIN_Y, BALL_MIN_Y,
            -BALL_MIN_Y - SPAN, BALL_MIN_Y - SPAN,
            -BALL_MIN_Y - 2 * SPAN, BALL_MIN_Y - 2 * SPAN,
            0, 0,
        ], dtype=float)

        # Il fold è BALL_MIN_Y + u0 + u1 -2u2 -2u3 +2u4 +2u5.
        fold_coeff = np.array([1, 1, -2, -2, 2, 2, 0, 0], dtype=float)
        paddle_coeff = np.array([0, 0, 0, 0, 0, 0, -1, 1], dtype=float)
        signed_error = fold_coeff + paddle_coeff

        self.W2 = np.vstack((signed_error, -signed_error))
        self.b2 = np.array([
            BALL_MIN_Y - self.delta,
            -BALL_MIN_Y - self.delta,
        ], dtype=float)
        self.W3 = np.array([[1.0, -1.0]])
        self.b3 = np.array([0.0])

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    def forward(self, paddle_y: float, y_pred: float):
        x = np.array([paddle_y, y_pred], dtype=float)
        u = self.relu(self.W1 @ x + self.b1)
        v = self.relu(self.W2 @ u + self.b2)
        output = float((self.W3 @ v + self.b3)[0])

        # Solo diagnostica: il controllo viene calcolato da W2 e W3.
        fold = float(BALL_MIN_Y + u[0] + u[1] - 2*u[2] - 2*u[3]
                     + 2*u[4] + 2*u[5])
        return output, {
            "layer1": u, "layer2": v,
            "fold": fold, "error": fold - paddle_y,
            "prediction": y_pred,
        }


# --------------------------------- Simulatore ---------------------------------

@dataclass
class Stats:
    frames: int = 0
    hits: int = 0
    misses: int = 0
    player_score: int = 0
    opponent_score: int = 0
    direction_changes: int = 0


class PongSimulation:
    def __init__(self, seed=0, opponent_skill=0.82, trace=False):
        self.rng = random.Random(seed)
        self.net = PongNetV5()
        self.opponent_skill = opponent_skill
        self.trace = trace
        self.stats = Stats()
        self.reset()

    def reset(self):
        self.player_y = HEIGHT / 2
        self.opponent_y = HEIGHT / 2
        self.ball_x = WIDTH / 2
        self.ball_y = HEIGHT / 2
        self.ball_vx = -BALL_SPEED
        self.ball_vy = self.rng.uniform(-3.5, 3.5)
        self.output = 0.0
        self.prediction = HEIGHT / 2
        self.fold = HEIGHT / 2
        self.last_direction = 0

    def serve(self, direction):
        self.ball_x, self.ball_y = WIDTH / 2, HEIGHT / 2
        self.ball_vx = direction * BALL_SPEED
        self.ball_vy = self.rng.uniform(-3.5, 3.5)

    def predict_player_intercept(self):
        if self.ball_vx >= 0:
            return HEIGHT / 2
        impact_x = PADDLE_MARGIN + PADDLE_W + BALL_R
        t = (impact_x - self.ball_x) / self.ball_vx
        if t < 0:
            return HEIGHT / 2
        raw = self.ball_y + self.ball_vy * t
        # Il range corrisponde al dominio coperto dai breakpoint della rete.
        return clamp(raw, -2 * SPAN + 1, 2.9 * SPAN)

    def step(self):
        self.prediction = self.predict_player_intercept()
        self.output, info = self.net.forward(self.player_y, self.prediction)
        self.fold = info["fold"]

        move = clamp(self.output, -PADDLE_SPEED, PADDLE_SPEED)
        if abs(move) > 0.05:
            direction = 1 if move > 0 else -1
            if self.last_direction and direction != self.last_direction:
                self.stats.direction_changes += 1
            self.last_direction = direction
            self.player_y += move

        # Avversario deterministico con una piccola imprecisione controllata.
        if self.ball_vx > 0:
            x_target = WIDTH - PADDLE_MARGIN - PADDLE_W - BALL_R
            t = (x_target - self.ball_x) / self.ball_vx
            target = reflect_y(self.ball_y + self.ball_vy * max(0, t)
                               * self.opponent_skill)
        else:
            target = HEIGHT / 2
        gap = target - self.opponent_y
        self.opponent_y += clamp(gap, -PADDLE_SPEED * 0.85,
                                 PADDLE_SPEED * 0.85)

        half = PADDLE_H / 2
        self.player_y = clamp(self.player_y, half, HEIGHT - half)
        self.opponent_y = clamp(self.opponent_y, half, HEIGHT - half)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y - BALL_R < TOP:
            self.ball_y = TOP + (TOP - (self.ball_y - BALL_R)) + BALL_R
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + BALL_R > BOTTOM:
            self.ball_y = BOTTOM - ((self.ball_y + BALL_R) - BOTTOM) - BALL_R
            self.ball_vy = -abs(self.ball_vy)

        left_x = PADDLE_MARGIN + PADDLE_W
        right_x = WIDTH - PADDLE_MARGIN - PADDLE_W

        if (self.ball_vx < 0 and self.ball_x - BALL_R <= left_x
                and abs(self.ball_y - self.player_y) <= half + BALL_R):
            self.ball_x = left_x + BALL_R
            self.ball_vx = min(MAX_BALL_SPEED, abs(self.ball_vx) * 1.025)
            offset = (self.ball_y - self.player_y) / half
            self.ball_vy = clamp(self.ball_vy + offset * 2.2, -7.5, 7.5)
            self.stats.hits += 1

        if (self.ball_vx > 0 and self.ball_x + BALL_R >= right_x
                and abs(self.ball_y - self.opponent_y) <= half + BALL_R):
            self.ball_x = right_x - BALL_R
            self.ball_vx = -min(MAX_BALL_SPEED, abs(self.ball_vx) * 1.025)
            offset = (self.ball_y - self.opponent_y) / half
            self.ball_vy = clamp(self.ball_vy + offset * 2.2, -7.5, 7.5)

        if self.ball_x < -BALL_R:
            self.stats.opponent_score += 1
            self.stats.misses += 1
            self.serve(+1)
        elif self.ball_x > WIDTH + BALL_R:
            self.stats.player_score += 1
            self.serve(-1)

        self.stats.frames += 1
        if self.trace:
            print(f"frame={self.stats.frames} ball=({self.ball_x:.1f},"
                  f"{self.ball_y:.1f}) pred={self.prediction:.1f} "
                  f"fold={self.fold:.1f} out={self.output:+.2f}")


# ---------------------------------- Grafica ------------------------------------

class PongApp:
    def __init__(self, seed=42):
        if pygame is None:
            raise RuntimeError("Pygame non installato: pip install pygame")
        pygame.init()
        pygame.display.set_caption("Pong V5 — Weight-Space Programming")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 19)
        self.small = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 27, bold=True)
        self.game = PongSimulation(seed=seed)
        self.paused = False
        self.running = True
        self.show_graph = True

    def text(self, value, x, y, color=WHITE, font=None):
        surf = (font or self.font).render(str(value), True, color)
        self.screen.blit(surf, (x, y))

    def draw_graph(self):
        # Mini grafico della funzione di fold per una coordinata grezza.
        rect = pygame.Rect(690, 440, 280, 175)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, MUTED, rect, 1, border_radius=10)
        self.text("FOLD / TRAIETTORIA", 710, 450, WHITE, self.small)
        gx, gy, gw, gh = 710, 480, 240, 105
        pygame.draw.line(self.screen, MUTED, (gx, gy + gh), (gx + gw, gy + gh), 1)
        pygame.draw.line(self.screen, MUTED, (gx, gy), (gx, gy + gh), 1)
        pts = []
        for i in range(101):
            raw = -SPAN + i * (3 * SPAN / 100)
            folded = reflect_y(raw)
            px = gx + i * gw / 100
            py = gy + gh - (folded - BALL_MIN_Y) / SPAN * gh
            pts.append((int(px), int(py)))
        pygame.draw.lines(self.screen, CYAN, False, pts, 2)
        marker_x = int(gx + ((self.pred_plot - (-SPAN)) / (3 * SPAN)) * gw)
        marker_x = int(clamp(marker_x, gx, gx + gw))
        pygame.draw.line(self.screen, YELLOW, (marker_x, gy),
                         (marker_x, gy + gh), 1)
        self.text("fold(y)", 710, 590, CYAN, self.small)
        self.text("predizione", 820, 590, YELLOW, self.small)

    def draw(self):
        g = self.game
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, FIELD, (18, 72, WIDTH - 36, HEIGHT - 150),
                         border_radius=12)
        for y in range(90, HEIGHT - 85, 26):
            pygame.draw.rect(self.screen, MUTED, (WIDTH // 2 - 2, y, 4, 13))

        self.text("PONG V5 — FOLD ReLU NELLA RETE", 245, 18,
                  WHITE, self.title_font)
        self.text("P1", 40, 82, GREEN)
        self.text("P2", WIDTH - 75, 82, YELLOW)
        score = f"{g.stats.player_score}  -  {g.stats.opponent_score}"
        sw = self.title_font.render(score, True, WHITE)
        self.screen.blit(sw, (WIDTH // 2 - sw.get_width() // 2, 80))

        left_x = PADDLE_MARGIN
        right_x = WIDTH - PADDLE_MARGIN - PADDLE_W
        pygame.draw.rect(self.screen, GREEN,
                         (left_x, int(g.player_y - PADDLE_H / 2),
                          PADDLE_W, PADDLE_H), border_radius=5)
        pygame.draw.rect(self.screen, YELLOW,
                         (right_x, int(g.opponent_y - PADDLE_H / 2),
                          PADDLE_W, PADDLE_H), border_radius=5)
        pygame.draw.circle(self.screen, WHITE,
                           (int(g.ball_x), int(g.ball_y)), BALL_R)

        # Linea prevista verso il piano d'impatto e marker del fold.
        impact_x = left_x + PADDLE_W + BALL_R
        pygame.draw.line(self.screen, RED,
                         (int(g.ball_x), int(g.ball_y)),
                         (impact_x, int(g.prediction)), 1)
        pygame.draw.circle(self.screen, RED,
                           (impact_x, int(clamp(g.prediction, 75, HEIGHT - 78))), 5)
        pygame.draw.circle(self.screen, CYAN,
                           (left_x + PADDLE_W + 15,
                            int(clamp(g.fold, 75, HEIGHT - 78))), 5)

        # HUD inferiore.
        pygame.draw.rect(self.screen, PANEL, (25, HEIGHT - 65, 625, 43),
                         border_radius=8)
        pygame.draw.rect(self.screen, MUTED, (25, HEIGHT - 65, 625, 43),
                         1, border_radius=8)
        self.text(f"net={g.output:+6.2f}", 40, HEIGHT - 56, GREEN, self.small)
        self.text(f"fold={g.fold:6.1f}", 190, HEIGHT - 56, CYAN, self.small)
        self.text(f"hits={g.stats.hits}", 340, HEIGHT - 56, YELLOW, self.small)
        self.text(f"miss={g.stats.misses}", 445, HEIGHT - 56, RED, self.small)
        self.text(f"frame={g.stats.frames}", 535, HEIGHT - 56, WHITE, self.small)

        if self.show_graph:
            self.pred_plot = g.prediction
            self.draw_graph()

        if self.paused:
            overlay = self.title_font.render("PAUSA", True, YELLOW)
            self.screen.blit(overlay, (WIDTH // 2 - overlay.get_width() // 2, 330))
        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.game = PongSimulation()
                    elif event.key == pygame.K_g:
                        self.show_graph = not self.show_graph
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.game.ball_vx *= 1.1
                        self.game.ball_vy *= 1.1
                    elif event.key == pygame.K_MINUS:
                        self.game.ball_vx *= 0.9
                        self.game.ball_vy *= 0.9
            if not self.paused:
                self.game.step()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


# ----------------------------------- Test --------------------------------------

def test_network(samples=50000):
    net = PongNetV5()
    rng = random.Random(2026)
    max_fold_err = 0.0
    max_out_err = 0.0
    for _ in range(samples):
        y = rng.uniform(-2 * SPAN + 1, 2.9 * SPAN)
        paddle = rng.uniform(-100, HEIGHT + 100)
        out, info = net.forward(paddle, y)
        expected_fold = reflect_y(y)
        expected_out = (max(0, expected_fold - paddle - DEADZONE)
                        - max(0, paddle - expected_fold - DEADZONE))
        max_fold_err = max(max_fold_err, abs(info["fold"] - expected_fold))
        max_out_err = max(max_out_err, abs(out - expected_out))
    assert max_fold_err < 1e-7, max_fold_err
    assert max_out_err < 1e-7, max_out_err
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
    parser = argparse.ArgumentParser(description="Pong V5")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=positive_int, default=10000)
    parser.add_argument("--seeds", type=positive_int, default=1)
    args = parser.parse_args()

    if args.test:
        test_network()
        return
    if args.headless:
        stress_test(args.seeds, args.frames)
        return
    PongApp().run()


if __name__ == "__main__":
    main()
