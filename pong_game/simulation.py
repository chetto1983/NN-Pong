"""Continuous circle/capsule collision physics, independent of rendering."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import (
    BALL_MAX_Y,
    BALL_MIN_Y,
    BALL_R,
    BALL_SPEED,
    CONTROL_GAIN,
    HEIGHT,
    MAX_BALL_SPEED,
    MIN_BALL_SPEED,
    PADDLE_FRICTION,
    PADDLE_MARGIN,
    PADDLE_MAX_Y,
    PADDLE_MIN_Y,
    PADDLE_RADIUS,
    PADDLE_SEGMENT_HALF,
    PADDLE_SPEED,
    PADDLE_W,
    PHYSICS_DT,
    SERVE_DELAY,
    WIDTH,
)
from .network import PongNet, clamp, reflect_y, wrap_period

_TIME_EPS = 1e-12
_SPACE_EPS = 1e-8
_VELOCITY_EPS = 1e-9


@dataclass
class Stats:
    steps: int = 0
    hits: int = 0
    misses: int = 0
    player_score: int = 0
    opponent_score: int = 0
    direction_changes: int = 0


class PongSimulation:
    def __init__(self, seed=0, opponent_skill=0.82, trace=False):
        self.rng = random.Random(seed)
        self.net = PongNet()
        self.opponent_skill = opponent_skill
        self.trace = trace
        self.reset()

    def reset(self):
        self.stats = Stats()
        self.elapsed = 0.0
        self.player_y = self.opponent_y = HEIGHT / 2.0
        self.player_vy = self.opponent_vy = 0.0
        self.output = 0.0
        self.prediction = self.raw_prediction = self.fold = HEIGHT / 2.0
        self.last_direction = 0
        self.serve(-1)

    def serve(self, direction):
        """Place a fresh ball at center, at a constant total speed."""
        angle = math.radians(self.rng.uniform(-25.0, 25.0))
        self.ball_x, self.ball_y = WIDTH / 2.0, HEIGHT / 2.0
        self.ball_vx = (-1 if direction < 0 else 1) * BALL_SPEED * math.cos(angle)
        self.ball_vy = BALL_SPEED * math.sin(angle)
        self.ball_spin = 0.0
        self.serve_remaining = 0.0

    def scale_ball_speed(self, factor):
        """Explicit user speed control; impact responses have no speed cap."""
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be finite and positive")
        speed = math.hypot(self.ball_vx, self.ball_vy)
        if speed == 0:
            self.ball_vx = -MIN_BALL_SPEED
            return
        target = clamp(speed * factor, MIN_BALL_SPEED, MAX_BALL_SPEED)
        self.ball_vx *= target / speed
        self.ball_vy *= target / speed

    def predict_player_intercept(self):
        """Return a periodic equivalent of the raw flight coordinate.

        The modulo reduction is done here, in Python: it accounts for any
        number of future wall bounces, so the network only has to fold once.
        raw_prediction remains available for diagnostics.
        """
        impact_x = PADDLE_MARGIN + PADDLE_W + BALL_R
        if self.ball_vx >= 0 or self.ball_x < impact_x:
            self.raw_prediction = HEIGHT / 2.0
        else:
            flight = (impact_x - self.ball_x) / self.ball_vx
            self.raw_prediction = self.ball_y + self.ball_vy * flight
        return wrap_period(self.raw_prediction)

    def _control(self, now):
        self.prediction = self.predict_player_intercept()
        self.output, info = self.net.forward(self.player_y, self.prediction)
        self.fold = info['fold']
        self.player_vy = clamp(self.output * CONTROL_GAIN,
                               -PADDLE_SPEED, PADDLE_SPEED)
        if abs(self.player_vy) > 3.0:
            direction = 1 if self.player_vy > 0 else -1
            if self.last_direction and direction != self.last_direction:
                self.stats.direction_changes += 1
            self.last_direction = direction
        if self.ball_vx > 0:
            impact_x = WIDTH - PADDLE_MARGIN - PADDLE_W - BALL_R
            flight = max(0.0, (impact_x - self.ball_x) / self.ball_vx)
            target = reflect_y(self.ball_y + self.ball_vy * flight
                               * self.opponent_skill)
            # A small deterministic stroke supplies real paddle momentum.
            # The collision impulse alone determines the ball's response.
            # Aim from a reachable center before applying the stroke, so a
            # near-wall intercept cannot pin the paddle forever at its stop.
            target = clamp(target, PADDLE_MIN_Y, PADDLE_MAX_Y)
            target += 30.0 * math.sin(2.0 * math.pi * 0.6 * now)
        else:
            target = HEIGHT / 2.0
        self.opponent_vy = clamp((target - self.opponent_y) * CONTROL_GAIN,
                                 -0.85 * PADDLE_SPEED, 0.85 * PADDLE_SPEED)

    def _paddle_contact(self, center_x, center_y, paddle_vy, duration):
        """First approaching contact against the expanded vertical capsule.

        In the paddle's moving frame the ball is a point, the capsule has
        radius paddle_radius + ball_radius, and its straight segment is
        unchanged. Candidate contacts are two side planes and two circles.
        """
        px, py = self.ball_x - center_x, self.ball_y - center_y
        vx, vy = self.ball_vx, self.ball_vy - paddle_vy
        radius = PADDLE_RADIUS + BALL_R
        half = PADDLE_SEGMENT_HALF
        nearest_y = clamp(py, -half, half)
        if math.hypot(px, py - nearest_y) < radius - _SPACE_EPS:
            # An externally placed, already penetrating ball must not be
            # pulled back across a paddle it has already missed.
            return None
        candidates = []

        def add(time, nx, ny):
            if (-_TIME_EPS <= time <= duration + _TIME_EPS
                    and vx * nx + vy * ny < -_VELOCITY_EPS):
                candidates.append((max(0.0, min(time, duration)), nx, ny))

        if abs(vx) > _VELOCITY_EPS:
            for nx in (-1.0, 1.0):
                time = (nx * radius - px) / vx
                if -half - _SPACE_EPS <= py + vy * time <= half + _SPACE_EPS:
                    add(time, nx, 0.0)
        speed_squared = vx * vx + vy * vy
        if speed_squared > _VELOCITY_EPS**2:
            for side in (-1.0, 1.0):
                cap_y = side * half
                dy = py - cap_y
                dot = px * vx + dy * vy
                distance = px * px + dy * dy - radius * radius
                discriminant = dot * dot - speed_squared * distance
                if discriminant < 0:
                    continue
                time = (-dot - math.sqrt(discriminant)) / speed_squared
                qx, qy = px + vx * time, dy + vy * time
                if qy * side >= -_SPACE_EPS:
                    length = math.hypot(qx, qy)
                    if length:
                        add(time, qx / length, qy / length)
        return min(candidates, default=None)

    def _impulse(self, nx, ny, paddle_vy):
        """Elastic normal impulse and Coulomb friction for a unit-mass disk."""
        relative_x, relative_y = self.ball_vx, self.ball_vy - paddle_vy
        normal_speed = relative_x * nx + relative_y * ny
        if normal_speed >= -_VELOCITY_EPS:
            return False
        tx, ty = -ny, nx
        tangent_speed = relative_x * tx + relative_y * ty - self.ball_spin * BALL_R
        normal_impulse = -2.0 * normal_speed
        limit = PADDLE_FRICTION * normal_impulse
        tangent_impulse = clamp(-tangent_speed / 3.0, -limit, limit)
        self.ball_vx += normal_impulse * nx + tangent_impulse * tx
        self.ball_vy += normal_impulse * ny + tangent_impulse * ty
        self.ball_spin -= 2.0 * tangent_impulse / BALL_R
        return True

    def _resolve_wall_pinch(self, name, nx, ny):
        """Yield a closing actuator when its cap traps a disk at a wall.

        Opposing rigid contacts cannot use independent elastic impulses: a
        prescribed moving paddle would add unbounded energy without advancing
        time. In this constrained contact the motor stalls instead. Remove
        blocked vertical motion and reflect incoming horizontal motion outward,
        keeping position and spin unchanged. This exception supplies no energy
        and leaves a feasible velocity without stopping a lateral approach.
        """
        against_top = self.ball_y <= BALL_MIN_Y + _SPACE_EPS and ny < 0
        against_bottom = self.ball_y >= BALL_MAX_Y - _SPACE_EPS and ny > 0
        if not (against_top or against_bottom):
            return False
        paddle_vy = getattr(self, name + '_vy')
        if paddle_vy * ny < 0:
            # A receding cap opens the gap and does not pin the ball.
            return False
        setattr(self, name + '_vy', 0.0)
        self.ball_vy = 0.0
        if self.ball_vx * nx < 0:
            self.ball_vx = -self.ball_vx
            if name == 'player':
                self.stats.hits += 1
        return True

    def _move(self, duration, ball_active):
        self.player_y += self.player_vy * duration
        self.opponent_y += self.opponent_vy * duration
        if ball_active:
            self.ball_x += self.ball_vx * duration
            self.ball_y += self.ball_vy * duration
        else:
            self.serve_remaining = max(0.0, self.serve_remaining - duration)

    def _advance(self, duration):
        """Resolve ordered contacts and paddle stops without dropping time."""
        remaining = duration
        while remaining > 0:
            events = []

            def add(time, kind, data=None):
                if -_TIME_EPS <= time <= remaining + _TIME_EPS:
                    events.append((max(0.0, min(time, remaining)), kind, data))

            # Stops split the trajectory so every sweep uses the actual
            # paddle velocity, including collisions after it reaches a wall.
            for name in ('player', 'opponent'):
                y, velocity = getattr(self, name + '_y'), getattr(self, name + '_vy')
                if velocity < 0:
                    add(max(0.0, (PADDLE_MIN_Y - y) / velocity), 'stop', (name, PADDLE_MIN_Y))
                elif velocity > 0:
                    add(max(0.0, (PADDLE_MAX_Y - y) / velocity), 'stop', (name, PADDLE_MAX_Y))

            ball_active = self.serve_remaining <= 0
            if not ball_active:
                add(self.serve_remaining, 'serve')
            else:
                if self.ball_vy < -_VELOCITY_EPS:
                    add((BALL_MIN_Y - self.ball_y) / self.ball_vy, 'wall', BALL_MIN_Y)
                elif self.ball_vy > _VELOCITY_EPS:
                    add((BALL_MAX_Y - self.ball_y) / self.ball_vy, 'wall', BALL_MAX_Y)
                if self.ball_x < -BALL_R:
                    add(0, 'goal', -1)
                elif self.ball_x > WIDTH + BALL_R:
                    add(0, 'goal', 1)
                elif self.ball_vx < -_VELOCITY_EPS:
                    add((-BALL_R - self.ball_x) / self.ball_vx, 'goal', -1)
                elif self.ball_vx > _VELOCITY_EPS:
                    add((WIDTH + BALL_R - self.ball_x) / self.ball_vx, 'goal', 1)
                for name, center_x in (
                    ('player', PADDLE_MARGIN + PADDLE_RADIUS),
                    ('opponent', WIDTH - PADDLE_MARGIN - PADDLE_RADIUS),
                ):
                    contact = self._paddle_contact(
                        center_x, getattr(self, name + '_y'),
                        getattr(self, name + '_vy'), remaining)
                    if contact is not None:
                        time, nx, ny = contact
                        add(time, 'paddle', (name, nx, ny))

            if not events:
                self._move(remaining, ball_active)
                break
            time = min(event[0] for event in events)
            self._move(time, ball_active)
            remaining = max(0.0, remaining - time)
            # Resolve simultaneous contacts before changing a paddle's
            # incoming velocity at its stop event.
            contacts = [event for event in events if abs(event[0] - time) <= _TIME_EPS]
            contacts.sort(key=lambda event: event[1] == 'stop')
            for _, kind, data in contacts:
                if kind == 'wall':
                    self.ball_y = data
                    self.ball_vy = abs(self.ball_vy) if data == BALL_MIN_Y else -abs(self.ball_vy)
                elif kind == 'paddle':
                    name, nx, ny = data
                    if self._resolve_wall_pinch(name, nx, ny):
                        continue
                    if self._impulse(nx, ny, getattr(self, name + '_vy')) and name == 'player':
                        self.stats.hits += 1
                elif kind == 'stop':
                    name, y = data
                    setattr(self, name + '_y', y)
                    setattr(self, name + '_vy', 0.0)
                elif kind == 'serve':
                    self.serve_remaining = 0.0
                elif kind == 'goal':
                    if data < 0:
                        self.stats.opponent_score += 1
                        self.stats.misses += 1
                    else:
                        self.stats.player_score += 1
                    self.serve(-data)
                    self.serve_remaining = SERVE_DELAY

    def step(self, dt=1.0 / 60.0, control=True):
        """Advance seconds of physics; control=False keeps assigned velocities.

        With control enabled, control updates are bounded to 120 Hz intervals.
        Each interval still uses continuous contact times, so high speeds do
        not tunnel through surfaces. Scoring includes an explicit timed serve
        pause; remaining time advances that pause and then the next rally.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if dt == 0:
            return
        remaining = dt
        while remaining > 0:
            interval = min(remaining, PHYSICS_DT) if control else remaining
            if control:
                self._control(self.elapsed + dt - remaining)
            self._advance(interval)
            remaining = max(0.0, remaining - interval)
            if remaining < _TIME_EPS:
                if remaining:
                    self._advance(remaining)
                break
        self.elapsed += dt
        self.stats.steps += 1
        if self.trace:
            print(f"step={self.stats.steps} t={self.elapsed:.3f} "
                  f"ball=({self.ball_x:.1f},{self.ball_y:.1f}) "
                  f"pred={self.prediction:.1f} fold={self.fold:.1f} "
                  f"out={self.output:+.2f}")
