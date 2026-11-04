"""A small ReLU network encoding a Hamiltonian cycle, without training.

Coordinates are integer grid cells, with (0, 0) at the upper left. The
rightmost column returns upward; the other columns snake through alternating
rows. An even row count joins the path into one cycle through the entire board.
"""

import json
from numbers import Integral
from pathlib import Path

import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)
WEIGHT_VERSION = 1


def _dimensions(cols, rows):
    if (isinstance(cols, (bool, np.bool_)) or not isinstance(cols, Integral)
            or isinstance(rows, (bool, np.bool_)) or not isinstance(rows, Integral)
            or cols < 6 or rows < 4 or rows % 2):
        raise ValueError('Board dimensions must be integers: cols >= 6 and even rows >= 4.')
    return int(cols), int(rows)


class SnakeNet:
    """Analytic 2 -> (rows + 4) -> 4 -> 4 network; 150 parameters at 18 x 14."""

    policy_name = 'cycle'

    @property
    def parameter_count(self):
        return sum(getattr(self, name).size for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))

    @property
    def dense_parameter_count(self):
        return self.parameter_count

    def forward_state(self, body, food):
        """Common state protocol; this baseline deliberately uses only the head."""
        return self.forward(*body[0])

    def __init__(self, cols=18, rows=14):
        self.cols, self.rows = _dimensions(cols, rows)
        width = self.rows + 4
        hinges = self.rows - 1
        right, left, penultimate, top, bottom = range(hinges, width)

        self.W1 = np.zeros((width, 2), dtype=np.float64)
        self.b1 = np.zeros(width, dtype=np.float64)
        # Hinge basis gives exact alternating parity at integer row indices.
        self.W1[:hinges, 1] = 1
        self.b1[:hinges] = -np.arange(hinges)
        self.W1[right, 0], self.b1[right] = 1, -(self.cols - 2)
        self.W1[left, 0], self.b1[left] = -1, 1
        self.W1[penultimate, 0], self.b1[penultimate] = 1, -(self.cols - 3)
        self.W1[top, 1], self.b1[top] = -1, 1
        self.W1[bottom, 1], self.b1[bottom] = 1, -(self.rows - 2)

        odd_row = np.zeros(width, dtype=np.float64)
        odd_row[0] = 1
        for k in range(1, hinges):
            odd_row[k] = 2 * (-1) ** k

        self.W2 = np.zeros((4, width), dtype=np.float64)
        self.b2 = np.array([0, 0, -1, 0], dtype=np.float64)
        # Gates: up = ReLU(R-T), down-left = ReLU(L-O),
        # down-right = ReLU(Q-2R+O-B-1), odd = ReLU(O).
        self.W2[0, right], self.W2[0, top] = 1, -1
        self.W2[1] = -odd_row
        self.W2[1, left] = 1
        self.W2[2] = odd_row
        self.W2[2, penultimate] = 1
        self.W2[2, right] = -2
        self.W2[2, bottom] = -1
        self.W2[3] = odd_row

        self.W3 = np.array([[3, 0, 0, 0], [0, 0, 0, 1],
                            [0, 3, 3, 0], [0, 0, 0, -1]], dtype=np.float64)
        self.b3 = np.array([0, 0, 0, 1], dtype=np.float64)

    def forward(self, x, y):
        """Return UP, RIGHT, DOWN, LEFT logits using only affine/ReLU layers."""
        if (isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
                or isinstance(y, (bool, np.bool_)) or not isinstance(y, Integral)
                or not 0 <= x < self.cols or not 0 <= y < self.rows):
            raise ValueError('Head coordinates must be integer cells inside the board.')
        inputs = np.array([x, y], dtype=np.float64)
        hidden1 = np.maximum(self.W1 @ inputs + self.b1, 0)
        hidden2 = np.maximum(self.W2 @ hidden1 + self.b2, 0)
        return self.W3 @ hidden2 + self.b3

    def save(self, path):
        """Write all weights and board dimensions to a portable JSON file."""
        payload = {'version': WEIGHT_VERSION, 'cols': self.cols, 'rows': self.rows}
        for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            payload[name] = getattr(self, name).tolist()
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        """Load matrices after checking the format, dimensions, shapes and values.

        Shape validation does not certify the policy: edited weights can change
        the chosen actions and may cause collisions.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError('Invalid Snake weight JSON.') from exc
        if (not isinstance(payload, dict) or type(payload.get('version')) is not int
                or payload['version'] != WEIGHT_VERSION):
            raise ValueError('Unsupported Snake weight format version.')
        cols, rows = _dimensions(payload.get('cols'), payload.get('rows'))
        width = rows + 4
        shapes = {'W1': (width, 2), 'b1': (width,), 'W2': (4, width),
                  'b2': (4,), 'W3': (4, 4), 'b3': (4,)}
        arrays = {}
        for name, shape in shapes.items():
            try:
                array = np.asarray(payload[name])
                if array.dtype.kind not in 'iuf' or array.shape != shape:
                    raise ValueError(f'{name} must be a numeric array with shape {shape}.')
                array = array.astype(np.float64)
            except (KeyError, TypeError, OverflowError, ValueError) as exc:
                raise ValueError(f'Invalid Snake weights for {name}; expected shape {shape}.') from exc
            if not np.isfinite(array).all():
                raise ValueError(f'Snake weights for {name} must all be finite.')
            arrays[name] = array
        net = cls.__new__(cls)
        net.cols, net.rows = cols, rows
        for name, array in arrays.items():
            setattr(net, name, array)
        return net


def create_network(policy='shortcuts', cols=18, rows=14):
    """Construct analytic weights for the selected policy, without fitting."""
    if policy == 'cycle':
        return SnakeNet(cols, rows)
    if policy == 'shortcuts':
        from .shortcut_network import ShortcutSnakeNet
        return ShortcutSnakeNet(cols, rows)
    raise ValueError(f'Unknown Snake policy: {policy!r}')


def load_network(path):
    """Dispatch portable v1 dense and v2 sparse weight files."""
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('Invalid Snake weight JSON.') from error
    if not isinstance(payload, dict) or type(payload.get('version')) is not int:
        raise ValueError('Unsupported Snake weight format.')
    if payload['version'] == 1:
        return SnakeNet.load(path)
    if payload['version'] == 2:
        from .shortcut_network import ShortcutSnakeNet
        return ShortcutSnakeNet.load(path)
    raise ValueError('Unsupported Snake weight format version.')
