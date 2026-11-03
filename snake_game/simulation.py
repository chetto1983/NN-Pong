"""Deterministic Snake rules, independent of the Pygame display."""

from numbers import Integral
import random

import numpy as np

from .network import RIGHT, SnakeNet, _dimensions


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class SnakeSimulation:
    """A head-first snake whose default actions are the argmax of network logits.

    Following the analytic cycle keeps the body in cycle order, so each free
    food cell is eventually reached. This guarantees completion on an empty
    rectangular board; external action overrides and edited weights can lose.
    """

    def __init__(self, seed=0, cols=18, rows=14, net=None):
        self.cols, self.rows = _dimensions(cols, rows)
        if net is None:
            net = SnakeNet(self.cols, self.rows)
        if (net.cols, net.rows) != (self.cols, self.rows):
            raise ValueError('Network and simulation board dimensions must match.')
        self.net = net
        self._random = random.Random(seed)
        start_row = self.rows // 2
        if start_row % 2 == 0:
            start_row += 1
        self.body = [(x, start_row) for x in (4, 3, 2, 1)]
        self.food = (min(13, self.cols - 1), start_row)
        self.score = 0
        self.steps = 0
        self.status = 'running'
        self.direction = RIGHT
        self.last_logits = self.net.forward(*self.body[0])

    def _spawn_food(self):
        occupied = set(self.body)
        available = [(x, y) for y in range(self.rows) for x in range(self.cols)
                     if (x, y) not in occupied]
        return self._random.choice(available) if available else None

    def step(self, action=None):
        """Advance once, or leave a terminal game unchanged.

        Explicit actions are useful for manual control and rule tests. Collisions
        count as an attempted step but retain the last valid body and food.
        ``last_logits`` describes the decision at the head before this step.
        """
        if self.status != 'running':
            return self.status
        if action is not None and (isinstance(action, (bool, np.bool_))
                                   or not isinstance(action, Integral)
                                   or not 0 <= action < 4):
            raise ValueError('Action must be an integer: UP=0, RIGHT=1, DOWN=2, LEFT=3.')
        self.last_logits = self.net.forward(*self.body[0])
        if action is None:
            action = int(np.argmax(self.last_logits))
        self.direction = int(action)
        self.steps += 1
        dx, dy = DIRECTIONS[self.direction]
        head_x, head_y = self.body[0]
        new_head = (head_x + dx, head_y + dy)
        growing = new_head == self.food
        blocking_body = self.body if growing else self.body[:-1]
        if (not 0 <= new_head[0] < self.cols or not 0 <= new_head[1] < self.rows
                or new_head in blocking_body):
            self.status = 'dead'
            return self.status
        self.body.insert(0, new_head)
        if growing:
            self.score += 1
            if len(self.body) == self.cols * self.rows:
                self.food = None
                self.status = 'won'
            else:
                self.food = self._spawn_food()
        else:
            self.body.pop()
        return self.status
