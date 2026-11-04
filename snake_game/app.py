"""Local Pygame presentation of the Snake board, with procedurally drawn sprites."""

from __future__ import annotations

import json
import math
from array import array
from pathlib import Path

import pygame

from . import sprites
from .network import SnakeNet, create_network
from .simulation import SnakeSimulation

WIDTH, HEIGHT = 908, 801
CELL = 48
BOARD = pygame.Rect(22, 100, 864, 672)
FRAME = (183, 80, 48)
HEADER = (100, 77, 31)
TILES = ((251, 193, 78), (252, 178, 79))
WHITE = (255, 255, 250)
GREEN = (80, 174, 15)
ASSETS = Path(__file__).resolve().parent / 'assets'
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class SnakeApp:
    """Autoplay UI. Arrow keys resume; only the network chooses directions."""

    def __init__(self, seed=0, net=None, speed=10, record_path=None):
        self.seed = seed
        self.net = net if net is not None else create_network()
        if (self.net.cols, self.net.rows) != (18, 14):
            raise ValueError('La finestra Snake richiede una rete 18 x 14; usa --headless per altre dimensioni.')
        self.speed = max(1, min(120, int(speed)))
        self.record_path = Path(record_path) if record_path is not None else None
        self.best_score = self._read_record()
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        label = 'scorciatoie' if self.net.policy_name == 'shortcuts' else 'ciclo'
        pygame.display.set_caption(f'Snake · {label} · pesi analitici')
        self.canvas = pygame.Surface((WIDTH, HEIGHT)).convert()
        self._fonts = {size: pygame.font.Font(str(ASSETS / 'Nunito-Bold.ttf'), size)
                       for size in (16, 18, 20, 22, 26, 32)}
        self._apple = sprites.apple(60)
        self._header_apple = sprites.apple(32)
        self._trophy = sprites.trophy(32)
        self._icons = sprites.icons(30)
        self._header_buttons = {
            name: pygame.Rect(x, 16, 38, 38)
            for name, x in zip(('help', 'settings', 'fullscreen', 'sound'), (748, 786, 824, 862))
        }
        self._speed_buttons = [(pygame.Rect(x, 341, 124, 44), speed)
                               for x, speed in ((258, 5), (392, 10), (526, 20))]
        self._path_button = pygame.Rect(258, 408, 392, 48)
        self._play_button = pygame.Rect(258, 522, 392, 52)
        self._close_button = pygame.Rect(618, 224, 32, 32)
        self._board_surface = self._make_board()
        self._snake_layer = pygame.Surface((WIDTH * 2, HEIGHT * 2), pygame.SRCALPHA)
        self._path_surface = None
        self._beep = self._make_beep()
        self.sound_enabled = True
        self.fullscreen = False
        self._window_size = (WIDTH, HEIGHT)
        self.running = True
        self.paused = False
        self.overlay = None
        self._paused_before_overlay = False
        self.show_path = False
        self._restart()

    def _read_record(self):
        if self.record_path is None:
            return 0
        try:
            value = json.loads(self.record_path.read_text(encoding='utf-8')).get('best_score', 0)
            return max(0, int(value))
        except (OSError, ValueError, TypeError, AttributeError, OverflowError):
            return 0

    def _save_record(self):
        if self.game.score <= self.best_score:
            return
        self.best_score = self.game.score
        if self.record_path is not None:
            try:
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
                self.record_path.write_text(json.dumps({'best_score': self.best_score}) + '\n', encoding='utf-8')
            except OSError:
                # A read-only home folder must not stop a game in progress.
                pass

    def _make_beep(self):
        config = pygame.mixer.get_init()
        if not config or config[1] != -16:
            return None
        frequency, _, channels = config
        duration = int(frequency * 0.085)
        samples = array('h')
        for i in range(duration):
            envelope = min(1.0, i / (frequency * 0.005)) * (1 - i / duration)
            value = int(3500 * envelope * math.sin(2 * math.pi * 740 * i / frequency))
            samples.extend([value] * channels)
        try:
            return pygame.mixer.Sound(buffer=samples)
        except pygame.error:
            return None

    def _make_board(self):
        board = pygame.Surface(BOARD.size, pygame.SRCALPHA)
        for row in range(14):
            for col in range(18):
                pygame.draw.rect(board, TILES[(row + col) % 2], (col * CELL, row * CELL, CELL, CELL))
        shadow = pygame.Surface(BOARD.size, pygame.SRCALPHA)
        for inset in range(12):
            alpha = max(0, 31 - inset * 3)
            pygame.draw.line(shadow, (93, 54, 12, alpha), (0, inset), (BOARD.width, inset))
            pygame.draw.line(shadow, (93, 54, 12, alpha // 2), (inset, 0), (inset, BOARD.height))
        board.blit(shadow, (0, 0))
        mask = pygame.Surface(BOARD.size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=12)
        board.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return board

    def _restart(self):
        self.game = SnakeSimulation(seed=self.seed, net=self.net)
        self._previous_body = list(self.game.body)
        self._previous_direction = self.game.direction
        self._accumulator = 0.0
        self._has_moved = False
        self.paused = False
        self.overlay = None

    def _open_overlay(self, name):
        if self.overlay == name:
            self._close_overlay()
            return
        if self.overlay is None:
            self._paused_before_overlay = self.paused
        self.overlay = name
        self.paused = True

    def _close_overlay(self, play=False):
        self.overlay = None
        self.paused = False if play else self._paused_before_overlay

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self._window_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self._window_size, pygame.RESIZABLE)
            # Some SDL drivers first leave fullscreen, then apply window size.
            if self.screen.get_size() != self._window_size:
                self.screen = pygame.display.set_mode(self._window_size, pygame.RESIZABLE)

    def _viewport(self):
        width, height = self.screen.get_size()
        scale = min(width / WIDTH, height / HEIGHT)
        size = (max(1, round(WIDTH * scale)), max(1, round(HEIGHT * scale)))
        return pygame.Rect((width - size[0]) // 2, (height - size[1]) // 2, *size)

    def _logical_position(self, position):
        rect = self._viewport()
        if not rect.collidepoint(position):
            return None
        return ((position[0] - rect.x) * WIDTH / rect.width,
                (position[1] - rect.y) * HEIGHT / rect.height)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self._window_size = (max(1, event.w), max(1, event.h))
            self.screen = pygame.display.set_mode(self._window_size, pygame.RESIZABLE)
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                if self.overlay:
                    self._close_overlay()
                else:
                    self.running = False
            elif key == pygame.K_h:
                self._open_overlay('help')
            elif key == pygame.K_f:
                self._toggle_fullscreen()
            elif key == pygame.K_r:
                self._save_record()
                self._restart()
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed = min(120, self.speed + 1)
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(1, self.speed - 1)
            elif key == pygame.K_g:
                self.show_path = not self.show_path
            elif key == pygame.K_m:
                self.sound_enabled = not self.sound_enabled
            elif key == pygame.K_SPACE and self.overlay is None:
                self.paused = not self.paused
            elif key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT) and self.overlay is None:
                self.paused = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            position = self._logical_position(event.pos)
            if position is not None:
                self._click(position)

    def _click(self, position):
        for name, rect in self._header_buttons.items():
            if rect.collidepoint(position):
                if name in ('help', 'settings'):
                    self._open_overlay(name)
                elif name == 'fullscreen':
                    self._toggle_fullscreen()
                elif name == 'sound':
                    self.sound_enabled = not self.sound_enabled
                return
        if self.overlay:
            if self._close_button.collidepoint(position):
                self._close_overlay()
            elif self._play_button.collidepoint(position):
                if self.game.status != 'running':
                    self._save_record()
                    self._restart()
                else:
                    self._close_overlay(play=True)
            elif self.overlay == 'settings':
                for rect, speed in self._speed_buttons:
                    if rect.collidepoint(position):
                        self.speed = speed
                if self._path_button.collidepoint(position):
                    self.show_path = not self.show_path
        elif self.game.status != 'running':
            if self._play_button.collidepoint(position):
                self._save_record()
                self._restart()
        elif self.paused:
            if self._play_button.collidepoint(position) or self._close_button.collidepoint(position):
                self.paused = False

    def update(self, dt_seconds):
        if not self.running or self.paused or self.overlay or self.game.status != 'running':
            return
        self._accumulator += max(0.0, dt_seconds)
        interval = 1.0 / self.speed
        while self._accumulator + 1e-12 >= interval and self.game.status == 'running':
            self._accumulator -= interval
            self._previous_body = list(self.game.body)
            self._previous_direction = self.game.direction
            score = self.game.score
            self.game.step()
            self._has_moved = True
            if self.game.score > score:
                self._save_record()
                if self.sound_enabled and self._beep and pygame.mixer.get_init():
                    self._beep.play()

    def _text(self, text, position, size=22, color=WHITE, center=False):
        surface = self._fonts[size].render(text, True, color)
        rect = surface.get_rect(center=position) if center else surface.get_rect(topleft=position)
        self.canvas.blit(surface, rect)

    def _button(self, rect, text, selected=False, green=False):
        color = GREEN if green else (132, 108, 58)
        pygame.draw.rect(self.canvas, color, rect, border_radius=10)
        if selected:
            pygame.draw.rect(self.canvas, WHITE, rect, width=2, border_radius=10)
        self._text(text, rect.center, size=20, center=True)

    @staticmethod
    def _center(cell):
        return (BOARD.x + (cell[0] + 0.5) * CELL, BOARD.y + (cell[1] + 0.5) * CELL)

    def _draw_path(self):
        if self._path_surface is None:
            self._path_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            reference = SnakeNet(self.net.cols, self.net.rows)
            for y in range(14):
                for x in range(18):
                    logits = reference.forward(x, y)
                    action = max(range(4), key=lambda direction: logits[direction])
                    dx, dy = DIRECTIONS[action]
                    destination = (x + dx, y + dy)
                    if 0 <= destination[0] < 18 and 0 <= destination[1] < 14:
                        start, end = self._center((x, y)), self._center(destination)
                        pygame.draw.line(self._path_surface, (116, 84, 32, 70), start, end, 2)
                        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                        pygame.draw.lines(self._path_surface, (116, 84, 32, 95), False,
                                          [(mid[0] - 4 * dx - 3 * dy, mid[1] - 4 * dy + 3 * dx),
                                           mid,
                                           (mid[0] - 4 * dx + 3 * dy, mid[1] - 4 * dy - 3 * dx)], 2)
        self.canvas.blit(self._path_surface, (0, 0))

    def _draw_snake(self):
        """Slide the ends along the actual body path, keeping corners on cells."""
        body = self.game.body
        fraction = min(1.0, max(0.0, self._accumulator * self.speed)) if self._has_moved else 1.0
        if self.game.status != 'running':
            fraction = 1.0
        if self._has_moved and len(body) > 1:
            previous_head = self._center(self._previous_body[0])
            head = self._center(body[0])
            points = [(previous_head[0] + (head[0] - previous_head[0]) * fraction,
                       previous_head[1] + (head[1] - previous_head[1]) * fraction)]
            points.extend(self._center(cell) for cell in body[1:])
            if len(body) == len(self._previous_body):
                old_tail = self._center(self._previous_body[-1])
                new_tail = self._center(body[-1])
                points.append((old_tail[0] + (new_tail[0] - old_tail[0]) * fraction,
                               old_tail[1] + (new_tail[1] - old_tail[1]) * fraction))
        else:
            points = [self._center(cell) for cell in body]
        dx, dy = DIRECTIONS[self.game.direction]
        head = points[0]
        points.insert(0, (head[0] + 12 * dx, head[1] + 12 * dy))
        layer = self._snake_layer
        layer.fill((0, 0, 0, 0))

        def tube(color, radius, offset=(0, 0)):
            scaled = [(round((x + offset[0]) * 2), round((y + offset[1]) * 2)) for x, y in points]
            if len(scaled) > 1:
                pygame.draw.lines(layer, color, False, scaled, round(radius * 4))
            for point in scaled:
                pygame.draw.circle(layer, color, point, round(radius * 2))

        def circle(color, center, radius):
            pygame.draw.circle(layer, color, (round(center[0] * 2), round(center[1] * 2)), round(radius * 2))

        tube((103, 72, 18, 48), 20, (0, 6))
        tube((76, 122, 249), 20)
        tube((58, 107, 222), 17)
        tube((92, 144, 245), 7, (0, -7))
        for side in (-1, 1):
            eye = (head[0] - 7 * dx - 14 * dy * side,
                   head[1] - 7 * dy + 14 * dx * side)
            circle((103, 72, 18, 48), (eye[0], eye[1] + 5), 17)
            circle((76, 122, 249), eye, 16)
            circle(WHITE, eye, 12.5)
            pupil = (eye[0] + 5 * dx, eye[1] + 5 * dy)
            circle((23, 32, 58), pupil, 8)
            circle(WHITE, (pupil[0] - 2, pupil[1] - 3), 2.5)
            nose = (head[0] + 25 * dx - 3 * dy * side,
                    head[1] + 25 * dy + 3 * dx * side)
            circle((28, 74, 156), nose, 1.5)
        self.canvas.blit(pygame.transform.smoothscale(layer, (WIDTH, HEIGHT)), (0, 0))

    def _draw_header(self):
        pygame.draw.rect(self.canvas, HEADER, (0, 0, WIDTH, 70))
        self.canvas.blit(self._header_apple, (12, 19))
        self._text(str(self.game.score), (52, 18))
        if self.best_score:
            score_width = self._fonts[22].size(str(self.game.score))[0]
            trophy_x = max(84, 52 + score_width + 14)
            self.canvas.blit(self._trophy, (trophy_x, 19))
            self._text(str(self.best_score), (trophy_x + 40, 18))
        mouse = self._logical_position(pygame.mouse.get_pos())
        for name, rect in self._header_buttons.items():
            if self.overlay == name or (mouse and rect.collidepoint(mouse)):
                pygame.draw.rect(self.canvas, (127, 105, 64), rect, border_radius=8)
            self.canvas.blit(self._icons[name], (rect.x + 4, rect.y + 4))
        if not self.sound_enabled:
            pygame.draw.line(self.canvas, HEADER, (870, 23), (893, 46), 5)
            pygame.draw.line(self.canvas, WHITE, (870, 23), (893, 46), 3)

    def _draw_overlay(self):
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        self.canvas.blit(shade, (0, 0))
        card = pygame.Rect(238, 208, 432, 296)
        pygame.draw.rect(self.canvas, HEADER, card, border_radius=12)
        pygame.draw.line(self.canvas, (140, 116, 64), (238, 271), (670, 271))
        if self.overlay or (self.paused and self.game.status == 'running'):
            pygame.draw.line(self.canvas, WHITE, (628, 233), (640, 245), 2)
            pygame.draw.line(self.canvas, WHITE, (640, 233), (628, 245), 2)
        if self.overlay == 'settings':
            self._text('Impostazioni', (258, 225), size=26)
            self._text('Snake classico · pesi analitici', (258, 286), size=20)
            for (rect, speed), label in zip(self._speed_buttons, ('Lento', 'Normale', 'Veloce')):
                self._button(rect, label, selected=self.speed == speed)
            self._text(f'Velocità: {self.speed} passi/s   (+ / −)', (258, 388), size=16)
            path_label = 'Ciclo di riferimento: visibile' if self.show_path else 'Mostra ciclo di riferimento'
            self._button(self._path_button, path_label, selected=self.show_path)
            self._text('La rete guida automaticamente il serpente.', (258, 470), size=16)
        elif self.overlay == 'help':
            self._text('Come giocare', (258, 225), size=26)
            for index, line in enumerate((
                'La rete guida il serpente e raccoglie le mele.',
                'Pesi scritti a mano, senza addestramento.',
                ('Usa mela e corpo per scegliere scorciatoie.'
                 if self.net.policy_name == 'shortcuts'
                 else 'Segue un ciclo che attraversa ogni casella.'),
                'Spazio: pausa     Frecce: riprendi',
                'R: ricomincia     + / −: velocità     G: ciclo',
                'F: schermo intero     M: suono     H: aiuto',
                'Esc: chiudi il pannello oppure esci',
            )):
                self._text(line, (258, 289 + index * 28), size=16)
        else:
            title = 'Hai vinto!' if self.game.status == 'won' else 'Partita terminata'
            if self.paused and self.game.status == 'running':
                title = 'In pausa'
            self._text(title, (454, 247), size=32, center=True)
            self._text(f'Mele raccolte: {self.game.score}', (454, 350), size=26, center=True)
            self._text(f'Record: {self.best_score}', (454, 397), size=20, center=True)
            if self.paused and self.game.status == 'running':
                self._text('Premi Spazio per riprendere', (454, 454), size=18, center=True)
        label = 'Rigioca' if self.game.status != 'running' and self.overlay is None else 'Gioca'
        self._button(self._play_button, label, green=True)

    def draw(self):
        self.canvas.fill(FRAME)
        self.canvas.blit(self._board_surface, BOARD)
        if self.show_path:
            self._draw_path()
        if self.game.food is not None:
            center = self._center(self.game.food)
            self.canvas.blit(self._apple, self._apple.get_rect(center=center))
        self._draw_snake()
        self._draw_header()
        if self.overlay or self.paused or self.game.status != 'running':
            self._draw_overlay()
        viewport = self._viewport()
        self.screen.fill((30, 24, 16))
        if viewport.size == (WIDTH, HEIGHT):
            self.screen.blit(self.canvas, viewport)
        else:
            self.screen.blit(pygame.transform.smoothscale(self.canvas, viewport.size), viewport)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        try:
            while self.running:
                elapsed = min(clock.tick(60) / 1000.0, 0.25)
                for event in pygame.event.get():
                    self.handle_event(event)
                if self.running:
                    self.update(elapsed)
                    self.draw()
        finally:
            self._save_record()
            pygame.quit()
