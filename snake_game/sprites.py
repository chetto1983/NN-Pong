"""Procedurally drawn sprites and icons for the Snake window.

Everything is drawn with Pygame primitives at four times the target size and
then smoothed down, so the project ships no third-party image files.
"""

from __future__ import annotations

import math

import pygame

_SCALE = 4



def _arc(surface, color, rect, start, stop, width):
    """Solid thick arc (pygame.draw.arc leaves gaps at large widths).

    Angles follow pygame's convention: radians, counterclockwise, y up.
    """
    rect = pygame.Rect(rect)
    cx, cy = rect.centerx, rect.centery
    rx, ry = rect.width / 2.0, rect.height / 2.0
    steps = max(8, int(abs(stop - start) * 24))
    outer, inner = [], []
    for i in range(steps + 1):
        a = start + (stop - start) * i / steps
        c, s = math.cos(a), -math.sin(a)
        outer.append((cx + c * rx, cy + s * ry))
        inner.append((cx + c * (rx - width), cy + s * (ry - width)))
    pygame.draw.polygon(surface, color, outer + inner[::-1])


def _canvas(size):
    return pygame.Surface((size * _SCALE, size * _SCALE), pygame.SRCALPHA)


def _finish(surface, size):
    return pygame.transform.smoothscale(surface, (size, size))


def apple(size):
    """Red apple with highlight, stem and leaf."""
    s = _canvas(size)
    u = s.get_width() / 100.0
    body = (226, 52, 40)
    shade = (181, 33, 30)
    pygame.draw.circle(s, shade, (round(50 * u), round(60 * u)), round(36 * u))
    pygame.draw.circle(s, body, (round(36 * u), round(57 * u)), round(28 * u))
    pygame.draw.circle(s, body, (round(64 * u), round(57 * u)), round(28 * u))
    pygame.draw.circle(s, body, (round(50 * u), round(66 * u)), round(28 * u))
    pygame.draw.ellipse(s, (255, 150, 130),
                        pygame.Rect(round(28 * u), round(40 * u), round(13 * u), round(20 * u)))
    pygame.draw.line(s, (110, 70, 30), (round(50 * u), round(36 * u)),
                     (round(55 * u), round(14 * u)), max(1, round(6 * u)))
    leaf = [(56 * u, 28 * u), (70 * u, 12 * u), (86 * u, 14 * u), (74 * u, 28 * u)]
    pygame.draw.polygon(s, (80, 170, 40), [(round(x), round(y)) for x, y in leaf])
    return _finish(s, size)


def trophy(size):
    """Gold cup with two handles on a stepped base."""
    s = _canvas(size)
    u = s.get_width() / 100.0
    gold, dark = (255, 200, 40), (214, 150, 20)

    def r(x, y, w, h):
        return pygame.Rect(round(x * u), round(y * u), round(w * u), round(h * u))

    width = max(1, round(7 * u))
    _arc(s, dark, r(8, 16, 30, 34), math.pi / 2, 3 * math.pi / 2, width)
    _arc(s, dark, r(62, 16, 30, 34), -math.pi / 2, math.pi / 2, width)
    cup = [(22, 10), (78, 10), (74, 44), (62, 58), (38, 58), (26, 44)]
    pygame.draw.polygon(s, gold, [(round(x * u), round(y * u)) for x, y in cup])
    pygame.draw.rect(s, (255, 235, 140), r(32, 16, 8, 26), border_radius=round(4 * u))
    pygame.draw.rect(s, dark, r(44, 58, 12, 16))
    pygame.draw.rect(s, gold, r(30, 74, 40, 10), border_radius=round(3 * u))
    pygame.draw.rect(s, dark, r(24, 84, 52, 10), border_radius=round(3 * u))
    return _finish(s, size)


def _white_icon(draw, size):
    s = _canvas(size)
    draw(s, s.get_width() / 24.0)
    return _finish(s, size)


def _help(s, u):
    center = (round(12 * u), round(12 * u))
    pygame.draw.circle(s, (255, 255, 255), center, round(10 * u), max(1, round(2 * u)))
    _arc(s, (255, 255, 255),
                    pygame.Rect(round(8.5 * u), round(6 * u), round(7 * u), round(7 * u)),
                    -math.pi / 2, math.pi, max(1, round(2 * u)))
    pygame.draw.line(s, (255, 255, 255), (round(12 * u), round(12.5 * u)),
                     (round(12 * u), round(14.5 * u)), max(1, round(2 * u)))
    pygame.draw.circle(s, (255, 255, 255), (round(12 * u), round(17.5 * u)), round(1.3 * u))


def _settings(s, u):
    center = 12 * u
    for k in range(8):
        angle = k * math.pi / 4
        dx, dy = math.cos(angle), math.sin(angle)
        px, py = -dy, dx
        corners = []
        for along, across in ((5, -1.6), (10, -1.6), (10, 1.6), (5, 1.6)):
            corners.append((round(center + (dx * along + px * across) * u),
                            round(center + (dy * along + py * across) * u)))
        pygame.draw.polygon(s, (255, 255, 255), corners)
    pygame.draw.circle(s, (255, 255, 255), (round(center), round(center)), round(7.5 * u))
    pygame.draw.circle(s, (0, 0, 0, 0), (round(center), round(center)), round(3.2 * u))


def _fullscreen(s, u):
    width = max(1, round(2 * u))
    for cx, cy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x = 12 - cx * 7
        y = 12 - cy * 7
        corner = (round(x * u), round(y * u))
        pygame.draw.line(s, (255, 255, 255), corner, (round((x + cx * 5) * u), round(y * u)), width)
        pygame.draw.line(s, (255, 255, 255), corner, (round(x * u), round((y + cy * 5) * u)), width)


def _sound(s, u):
    speaker = [(3, 9), (7, 9), (12, 4), (12, 20), (7, 15), (3, 15)]
    pygame.draw.polygon(s, (255, 255, 255), [(round(x * u), round(y * u)) for x, y in speaker])
    width = max(1, round(2 * u))
    for radius in (4, 8):
        rect = pygame.Rect(round((12 - radius) * u), round((12 - radius) * u),
                           round(2 * radius * u), round(2 * radius * u))
        _arc(s, (255, 255, 255), rect, -math.pi / 3.2, math.pi / 3.2, width)


def _replay(s, u):
    rect = pygame.Rect(round(4 * u), round(4 * u), round(16 * u), round(16 * u))
    _arc(s, (255, 255, 255), rect, math.pi / 4, 2 * math.pi - 0.15, max(1, round(2 * u)))
    tip = [(14.5, 2.5), (20.5, 6.5), (14, 10)]
    pygame.draw.polygon(s, (255, 255, 255), [(round(x * u), round(y * u)) for x, y in tip])


_ICONS = {'help': _help, 'settings': _settings, 'fullscreen': _fullscreen,
          'sound': _sound, 'replay': _replay}


def icons(size):
    """Return white UI icons keyed by help, settings, fullscreen, sound, replay."""
    return {name: _white_icon(draw, size) for name, draw in _ICONS.items()}
