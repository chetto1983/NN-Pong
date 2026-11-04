"""Plain-Python reference policies for Snake, used for checks and comparisons.

``shortcut_logits`` re-implements, cell by cell, the rule that the sparse
network's weights encode. It is used to (a) verify the network computes exactly
that rule and (b) time the same decision without the neural form.
``greedy_action`` is an independent heuristic baseline (BFS to the food with
a flood-fill fallback); it gives no completion guarantee.
"""

from __future__ import annotations

from collections import deque

import numpy as np

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def shortcut_logits(body, food, cols, rows, ranks):
    """UP, RIGHT, DOWN, LEFT logits of the shortcut rule, computed directly."""
    capacity = cols * rows
    head_x, head_y = body[0]
    head_rank = ranks[head_y * cols + head_x]

    def distance(cell):
        diff = (ranks[cell[1] * cols + cell[0]] - head_rank) % capacity
        return diff or capacity

    d_food, d_tail = distance(food), distance(body[-1])
    permitted = len(body) < capacity // 2
    occupied = set(body[:-1])
    logits = []
    for dx, dy in DIRECTIONS:
        target = (head_x + dx, head_y + dy)
        inside = 0 <= target[0] < cols and 0 <= target[1] < rows
        d = distance(target) if inside else capacity
        blocked = (not inside) or target in occupied
        penalty = (max(0, d - d_food) + max(0, d - d_tail)
                   + max(0, d - capacity * permitted - 1) + blocked)
        logits.append(d - (capacity + 1) * penalty)
    return np.array(logits, dtype=np.float64)


def shortcut_action(body, food, cols, rows, ranks):
    return int(np.argmax(shortcut_logits(body, food, cols, rows, ranks)))


def _neighbours(cell, cols, rows):
    x, y = cell
    for action, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            yield action, (nx, ny)


def _reachable(start, blocked, cols, rows):
    seen, queue = {start}, deque([start])
    while queue:
        cell = queue.popleft()
        for _, nxt in _neighbours(cell, cols, rows):
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def _path(start, goal, blocked, cols, rows):
    """Shortest path start -> goal (excluding start) avoiding blocked cells, or None."""
    parent, queue = {start: None}, deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell != start:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        for _, nxt in _neighbours(cell, cols, rows):
            if nxt not in parent and (nxt not in blocked or nxt == goal):
                parent[nxt] = cell
                queue.append(nxt)
    return None


def _tail_reachable(body, cols, rows):
    return _path(body[0], body[-1], set(body[1:-1]), cols, rows) is not None


def greedy_action(body, food, cols, rows):
    """Safe greedy: shortest path to food only if the tail stays reachable afterwards.

    Otherwise follow the tail along the longest-shortest route, else take the
    move with the most free space. A standard strong heuristic, no guarantee.
    """
    head = body[0]
    actions = {cell: action for action, cell in _neighbours(head, cols, rows)}
    path = _path(head, food, set(body[:-1]), cols, rows)
    if path:
        virtual = (path[::-1] + body)[:len(body) + 1]
        if len(virtual) == cols * rows or _tail_reachable(virtual, cols, rows):
            return actions[path[0]]
    best, best_len = None, -1
    for cell, action in actions.items():
        if cell in body[:-1]:
            continue
        virtual = [cell] + (body if cell == food else body[:-1])
        if len(virtual) > 1 and _tail_reachable(virtual, cols, rows):
            route = _path(cell, virtual[-1], set(virtual[1:-1]), cols, rows)
            if route is not None and len(route) > best_len:
                best, best_len = action, len(route)
    if best is not None:
        return best
    best, best_space = 0, -1
    blocked = set(body[:-1]) | {head}
    for cell, action in actions.items():
        if cell in blocked:
            continue
        space = _reachable(cell, blocked, cols, rows)
        if space > best_space:
            best, best_space = action, space
    return best
