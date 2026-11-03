"""Pygame presentation: a clear court, optional diagnostics and fixed-step motion."""

from collections import deque
import math
from pathlib import Path

import pygame
import pygame.gfxdraw

from .config import (
    BALL_MIN_Y, BALL_R, BOTTOM, PADDLE_H, PADDLE_MARGIN,
    PADDLE_W, PHYSICS_DT, SPAN, TOP, WIDTH,
)
from .simulation import PongSimulation


BG = (12, 19, 29)
FIELD = (18, 29, 42)
PANEL = (22, 34, 48)
BORDER = (44, 61, 79)
WHITE = (237, 244, 248)
MUTED = (139, 157, 175)
TEAL = (89, 221, 199)
CORAL = (255, 145, 126)
AMBER = (246, 202, 123)

LOGICAL_SIZE = (1200, 900)
DIAGNOSTIC_SIZE = (1440, 900)
WORLD_OFFSET = (100, 125)
FIXED_DT = PHYSICS_DT


class PongApp:
    """Own the window and visual state without changing simulation time scale."""

    def __init__(self, seed=42):
        pygame.init()
        pygame.display.set_caption('Pong')
        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
        window_scale = min(1.0, max(1, desktop_width - 80) / LOGICAL_SIZE[0],
                           max(1, desktop_height - 100) / LOGICAL_SIZE[1])
        initial_size = tuple(max(1, int(dimension * window_scale))
                             for dimension in LOGICAL_SIZE)
        self.screen = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
        self.canvas = pygame.Surface(LOGICAL_SIZE).convert()
        self.clock = pygame.time.Clock()
        font_path = Path(__file__).resolve().parents[1] / 'snake_assets' / 'Nunito-Bold.ttf'
        font_file = str(font_path) if font_path.is_file() else None
        self.fonts = {size: pygame.font.Font(font_file, size)
                      for size in (12, 13, 14, 16, 18, 22, 24, 30, 60)}
        self.seed = seed
        self.game = PongSimulation(seed=seed)
        self.paused = False
        self.running = True
        self.show_graph = False
        self.fullscreen = False
        self._windowed_size = self.screen.get_size()
        self._accumulator = 0.0
        self._previous = self._current = self._snapshot()
        self.trail = deque(maxlen=7)
        self._impacts = []
        self.court_rect = pygame.Rect(*self.world_to_canvas(0, TOP), WIDTH, BOTTOM - TOP)
        self.diagnostics_rect = None
        self.button_rects = {
            'pause': pygame.Rect(100, 804, 146, 44),
            'restart': pygame.Rect(258, 804, 166, 44),
            'diagnostics': pygame.Rect(923, 804, 177, 44),
        }
        self.viewport_rect = pygame.Rect(0, 0, *LOGICAL_SIZE)
        self.ball_draw_position = self.world_to_canvas(self.game.ball_x, self.game.ball_y)
        self.paddle_rects = self._paddle_rects(self.game.player_y, self.game.opponent_y)
        self._fit_viewport()

    @staticmethod
    def world_to_canvas(x, y):
        """Map physical world coordinates to the unscaled presentation canvas."""
        return (WORLD_OFFSET[0] + x, WORLD_OFFSET[1] + y)

    def window_to_canvas(self, position):
        """Ignore letterbox margins and map input through the current viewport."""
        self._fit_viewport()
        if not self.viewport_rect.collidepoint(position):
            return None
        return ((position[0] - self.viewport_rect.x) * self.canvas.get_width()
                / self.viewport_rect.width,
                (position[1] - self.viewport_rect.y) * self.canvas.get_height()
                / self.viewport_rect.height)

    def _fit_viewport(self):
        width, height = self.screen.get_size()
        canvas_width, canvas_height = self.canvas.get_size()
        scale = min(width / canvas_width, height / canvas_height)
        fitted = (max(1, round(canvas_width * scale)),
                  max(1, round(canvas_height * scale)))
        self.viewport_rect = pygame.Rect((width - fitted[0]) // 2,
                                         (height - fitted[1]) // 2, *fitted)

    def _snapshot(self):
        game = self.game
        return (game.ball_x, game.ball_y, game.player_y, game.opponent_y)

    def _render_state(self):
        current = self._snapshot()
        # Also support callers advancing the simulation directly, e.g. previews.
        if current != self._current:
            return current
        alpha = min(1.0, self._accumulator / FIXED_DT)
        return tuple(before + (after - before) * alpha
                     for before, after in zip(self._previous, current))

    def _restart(self):
        self.game = PongSimulation(seed=self.seed)
        self._accumulator = 0.0
        self._previous = self._current = self._snapshot()
        self.trail.clear()
        self._impacts.clear()
        self.paused = False

    def _toggle_diagnostics(self):
        self.show_graph = not self.show_graph
        size = DIAGNOSTIC_SIZE if self.show_graph else LOGICAL_SIZE
        self.canvas = pygame.Surface(size).convert()
        self.diagnostics_rect = pygame.Rect(1140, 32, 268, 816) if self.show_graph else None
        self._fit_viewport()

    def _toggle_fullscreen(self):
        if self.fullscreen:
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
            # Some SDL drivers leave fullscreen before applying the window size.
            if self.screen.get_size() != self._windowed_size:
                self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        else:
            self._windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.fullscreen = not self.fullscreen
        self._fit_viewport()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self._windowed_size = (max(1, event.w), max(1, event.h))
            self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
            self._fit_viewport()
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key == pygame.K_r:
                self._restart()
            elif key == pygame.K_g:
                self._toggle_diagnostics()
            elif key == pygame.K_f:
                self._toggle_fullscreen()
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.game.scale_ball_speed(1.1)
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.game.scale_ball_speed(1 / 1.1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            position = self.window_to_canvas(event.pos)
            if position is None:
                return
            for action, rect in self.button_rects.items():
                if rect.collidepoint(position):
                    if action == 'pause':
                        self.paused = not self.paused
                    elif action == 'restart':
                        self._restart()
                    else:
                        self._toggle_diagnostics()
                    break

    def update(self, dt):
        """Advance elapsed seconds using 120 Hz physics, retaining partial steps."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative')
        if self.paused or not self.running:
            return
        self._accumulator += dt
        while self._accumulator + 1e-12 >= FIXED_DT:
            game = self.game
            self._previous = self._snapshot()
            score = (game.stats.player_score, game.stats.opponent_score)
            velocity = (game.ball_vx, game.ball_vy)
            self._impacts = [(x, y, age + FIXED_DT, color)
                             for x, y, age, color in self._impacts if age < 0.16]
            game.step(FIXED_DT)
            self._current = self._snapshot()
            if score != (game.stats.player_score, game.stats.opponent_score):
                # A serve is a discontinuity, never draw a trail across the court.
                self._previous = self._current
                self.trail.clear()
                self._impacts.clear()
            elif getattr(game, 'serve_remaining', 0) > 0:
                self.trail.clear()
            else:
                self.trail.append(self._previous[:2])
                if velocity[0] * game.ball_vx < 0:
                    color = TEAL if game.ball_vx > 0 else CORAL
                    self._impacts.append((game.ball_x, game.ball_y, 0.0, color))
                elif velocity[1] * game.ball_vy < 0:
                    self._impacts.append((game.ball_x, game.ball_y, 0.0, MUTED))
            self._accumulator = max(0.0, self._accumulator - FIXED_DT)

    def _text(self, value, x, y, size=16, color=WHITE, align='left'):
        surface = self.fonts[size].render(str(value), True, color)
        rect = surface.get_rect()
        if align == 'center':
            rect.midtop = (round(x), round(y))
        elif align == 'right':
            rect.topright = (round(x), round(y))
        else:
            rect.topleft = (round(x), round(y))
        self.canvas.blit(surface, rect)

    def _circle(self, color, position, radius):
        x, y = (round(position[0]), round(position[1]))
        radius = round(radius)
        pygame.gfxdraw.aacircle(self.canvas, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.canvas, x, y, radius, color)

    def _paddle_rects(self, player_y, opponent_y):
        return tuple(pygame.Rect(*self.world_to_canvas(x, y - PADDLE_H / 2),
                                PADDLE_W, PADDLE_H)
                     for x, y in ((PADDLE_MARGIN, player_y),
                                  (WIDTH - PADDLE_MARGIN - PADDLE_W, opponent_y)))

    def _draw_capsule(self, rect, color):
        # Supersampling rounds the caps without changing the physical dimensions.
        scale = 3
        surface = pygame.Surface((rect.width * scale, rect.height * scale), pygame.SRCALPHA)
        pygame.draw.rect(surface, color, surface.get_rect(), border_radius=rect.width * scale // 2)
        self.canvas.blit(pygame.transform.smoothscale(surface, rect.size), rect)

    def _draw_header(self):
        pygame.draw.rect(self.canvas, TEAL, (100, 48, 4, 24), border_radius=2)
        pygame.draw.rect(self.canvas, CORAL, (111, 53, 4, 14), border_radius=2)
        self._text('PONG', 129, 42, 24)
        self._text('RETE', 508, 25, 13, TEAL, 'center')
        self._text(self.game.stats.player_score, 508, 39, 60, WHITE, 'center')
        self._text('AVVERSARIO', 692, 25, 13, CORAL, 'center')
        self._text(self.game.stats.opponent_score, 692, 39, 60, WHITE, 'center')
        self._circle(BORDER, (600, 67), 2)
        self._circle(BORDER, (600, 77), 2)
        serving = getattr(self.game, 'serve_remaining', 0) > 0
        status = 'IN PAUSA' if self.paused else 'SERVIZIO' if serving else 'IN GIOCO'
        status_color = AMBER if self.paused or serving else TEAL
        pygame.draw.rect(self.canvas, PANEL, (982, 46, 118, 32), border_radius=16)
        self._circle(status_color, (998, 62), 3)
        self._text(status, 1010, 52, 12, status_color)

    def _draw_court(self):
        rect = self.court_rect
        pygame.draw.rect(self.canvas, FIELD, rect)
        # The inner edge of each visible rail is exactly the collision wall.
        pygame.draw.rect(self.canvas, BORDER, (rect.left, rect.top - 3, rect.width, 3))
        pygame.draw.rect(self.canvas, BORDER, (rect.left, rect.bottom, rect.width, 3))
        for y in range(rect.top + 18, rect.bottom - 10, 28):
            pygame.draw.rect(self.canvas, (35, 49, 64), (599, y, 2, 10), border_radius=1)
        # Small colored ends identify the two sides without implying side walls.
        for x, color in ((rect.left, TEAL), (rect.right - 34, CORAL)):
            pygame.draw.rect(self.canvas, color, (x, rect.top - 3, 34, 3))
            pygame.draw.rect(self.canvas, color, (x, rect.bottom, 34, 3))

        state = self._render_state()
        self.ball_draw_position = self.world_to_canvas(*state[:2])
        self.paddle_rects = self._paddle_rects(*state[2:])
        previous_clip = self.canvas.get_clip()
        # The ball remains visible through the open goals until fully outside.
        self.canvas.set_clip(pygame.Rect(80, rect.top, 1040, rect.height))
        for index, point in enumerate(self.trail):
            strength = (index + 1) / len(self.trail)
            color = tuple(round(a + (b - a) * strength * 0.22)
                          for a, b in zip(FIELD, WHITE))
            self._circle(color, self.world_to_canvas(*point), 3 + strength * 3)
        for x, y, age, color in self._impacts:
            strength = max(0.0, 1 - age / 0.16)
            faded = tuple(round(a + (b - a) * strength * 0.4)
                          for a, b in zip(FIELD, color))
            pygame.draw.circle(self.canvas, faded, self.world_to_canvas(x, y),
                               round(11 + age * 45), 1)
        for paddle, color in zip(self.paddle_rects, (TEAL, CORAL)):
            self._draw_capsule(paddle, color)
        self._circle(WHITE, self.ball_draw_position, BALL_R)
        self.canvas.set_clip(previous_clip)

    def _draw_button(self, action, label, shortcut, active=False):
        rect = self.button_rects[action]
        hover = self.window_to_canvas(pygame.mouse.get_pos())
        hovered = hover is not None and rect.collidepoint(hover)
        fill = (34, 52, 65) if active or hovered else PANEL
        pygame.draw.rect(self.canvas, fill, rect, border_radius=9)
        pygame.draw.rect(self.canvas, BORDER, rect, 1, border_radius=9)
        self._text(label, rect.left + 15, rect.top + 11, 16,
                   TEAL if active else WHITE)
        key_width = 22 if len(shortcut) == 1 else 36
        key_rect = pygame.Rect(rect.right - key_width - 11, rect.top + 11, key_width, 22)
        pygame.draw.rect(self.canvas, BG, key_rect, border_radius=4)
        self._text(shortcut, key_rect.centerx, key_rect.y + 2, 12, MUTED, 'center')

    def _draw_footer(self):
        self._draw_button('pause', 'Riprendi' if self.paused else 'Pausa', 'SPC', self.paused)
        self._draw_button('restart', 'Ricomincia', 'R')
        self._draw_button('diagnostics', 'Diagnostica', 'G', self.show_graph)
        self._text('SPAZIO  pausa', 100, 866, 12, MUTED)
        self._text('− / +  velocità', 261, 866, 12, MUTED)
        self._text('F  schermo intero', 431, 866, 12, MUTED)
        self._text('ESC  esci', 1100, 866, 12, MUTED, 'right')

    def _draw_diagnostics(self):
        game = self.game
        x = 1164
        pygame.draw.line(self.canvas, BORDER, (1140, 32), (1140, 848))
        self._text('DIAGNOSTICA', x, 43, 18)
        self._text('Il controllo della rete', x, 72, 13, MUTED)
        speed = math.hypot(game.ball_vx, game.ball_vy)
        metrics = [('VELOCITÀ', f'{speed:.0f}', 'px / s'),
                   ('RESPINTE', str(game.stats.hits), ''),
                   ('TEMPO', f'{game.elapsed:.1f}', 's')]
        for index, (label, value, unit) in enumerate(metrics):
            y = 132 + index * 80
            self._text(label, x, y, 12, MUTED)
            self._text(value, x, y + 18, 30)
            self._text(unit, 1396, y + 32, 13, MUTED, 'right')
            pygame.draw.line(self.canvas, BORDER, (x, y + 69), (1396, y + 69))
        self._text('RIFLESSIONE', x, 403, 13, TEAL)
        self._text('Da previsione a punto sul campo', x, 426, 12, MUTED)
        graph = pygame.Rect(x, 466, 232, 138)
        pygame.draw.rect(self.canvas, FIELD, graph, border_radius=5)
        for fraction in (0.0, 0.5, 1.0):
            y = graph.bottom - 8 - fraction * (graph.height - 16)
            pygame.draw.line(self.canvas, BORDER, (graph.left + 8, y), (graph.right - 8, y))
        points = [(graph.left + 8, graph.bottom - 8),
                  (graph.centerx, graph.top + 8),
                  (graph.right - 8, graph.bottom - 8)]
        pygame.draw.aalines(self.canvas, TEAL, False, points)
        phase = (game.raw_prediction - BALL_MIN_Y) % (2 * SPAN)
        folded = BALL_MIN_Y + (phase if phase <= SPAN else 2 * SPAN - phase)
        marker = (graph.left + 8 + phase / (2 * SPAN) * (graph.width - 16),
                  graph.bottom - 8 - (folded - BALL_MIN_Y) / SPAN * (graph.height - 16))
        self._circle(WHITE, marker, 4)
        self._text('0', graph.left, graph.bottom + 8, 12, MUTED)
        self._text('1 periodo', graph.right, graph.bottom + 8, 12, MUTED, 'right')
        for index, (label, value) in enumerate((
            ('Previsione', f'{game.raw_prediction:.0f} px'),
            ('Punto riflesso', f'{game.fold:.0f} px'),
            ('Uscita rete', f'{game.output:+.1f}'),
        )):
            y = 674 + index * 36
            self._text(label, x, y, 13, MUTED)
            self._text(value, 1396, y, 13, WHITE, 'right')
        self._text('G  chiudi diagnostica', x, 816, 12, MUTED)

    def draw(self):
        self.canvas.fill(BG)
        self._draw_header()
        self._draw_court()
        self._draw_footer()
        if self.show_graph:
            self._draw_diagnostics()
        self._fit_viewport()
        self.screen.fill(BG)
        if self.viewport_rect.size == self.canvas.get_size():
            self.screen.blit(self.canvas, self.viewport_rect)
        else:
            scaled = pygame.transform.smoothscale(self.canvas, self.viewport_rect.size)
            self.screen.blit(scaled, self.viewport_rect)
        pygame.display.flip()

    def run(self):
        try:
            while self.running:
                # Limit suspension catch-up in the interactive loop only. Public
                # update(dt) still accounts for every requested simulation second.
                dt = min(self.clock.tick(144) / 1000.0, 0.25)
                for event in pygame.event.get():
                    self.handle_event(event)
                if self.running:
                    self.update(dt)
                    self.draw()
        finally:
            pygame.quit()
