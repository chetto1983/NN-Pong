"""A sparse ReLU controller whose analytic weights encode safe cycle shortcuts.

The observation is four raw row-major binary planes: head, food, tail, body.
Only construction uses grid geometry or cycle ranks. Inference is sparse affine
layers and ReLUs, including occupancy, modular distances and the density cutoff.
The completion guarantee applies to cycle-ordered states reached from the normal
initial body with these weights; edited weights or arbitrary bodies can lose.
"""

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from .network import _dimensions

WEIGHT_VERSION = 2
_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class SparseLayer:
    """Mutable coordinate-format affine weights; repeated entries are summed."""

    src: np.ndarray
    dst: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    input_size: int
    output_size: int

    def forward(self, values):
        return (np.bincount(self.dst, weights=self.weights * values[self.src],
                            minlength=self.output_size) + self.bias)


def _architecture(cols, rows):
    capacity = cols * rows
    arcs = 2 * (cols * (rows - 1) + rows * (cols - 1))
    return [4 * capacity, arcs + 11, 28, 11, 20, 4]


def _cycle_ranks(cols, rows):
    capacity = cols * rows
    ranks = np.empty(capacity, dtype=np.int64)
    for y in range(rows):
        for x in range(cols):
            rank = (capacity - y if x == cols - 1 else
                    y * (cols - 1) + (x + 1 if y % 2 else cols - 1 - x))
            ranks[y * cols + x] = rank % capacity
    return ranks


def _layer(input_size, output_size, entries, bias=None):
    """Pack analytic (source, destination, coefficient) entries once."""
    return SparseLayer(
        src=np.array([entry[0] for entry in entries], dtype=np.intp),
        dst=np.array([entry[1] for entry in entries], dtype=np.intp),
        weights=np.array([entry[2] for entry in entries], dtype=np.float64),
        bias=np.zeros(output_size) if bias is None else np.asarray(bias, dtype=np.float64),
        input_size=input_size, output_size=output_size,
    )


class ShortcutSnakeNet:
    """1008 -> 955 -> 28 -> 11 -> 20 -> 4 on the default 18 x 14 board."""

    policy_name = 'shortcuts'

    def __init__(self, cols=18, rows=14):
        self.cols, self.rows = _dimensions(cols, rows)
        self.cycle_ranks = _cycle_ranks(self.cols, self.rows)
        self.layers = self._compile_layers()

    def _compile_layers(self):
        """Translate the geometric policy into fixed affine/ReLU coefficients."""
        cols, rows = self.cols, self.rows
        capacity = cols * rows
        sizes = _architecture(cols, rows)
        arcs = sizes[1] - 11
        ranks = self.cycle_ranks
        z_start, wall_start, length_index = arcs, arcs + 6, arcs + 10

        # Layer 1: each directed adjacency has a head AND occupied-not-tail gate.
        # Six shifted differences Z = target_rank - head_rank + N are positive.
        entries, adjacency_actions = [], []
        bias = np.zeros(sizes[1])
        bias[:arcs] = -1
        bias[z_start:z_start + 6] = capacity
        gate = 0
        for source in range(capacity):
            x, y = source % cols, source // cols
            for action, (dx, dy) in enumerate(_DIRECTIONS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    target = ny * cols + nx
                    entries.extend(((source, gate, 1), (3 * capacity + target, gate, 1),
                                    (2 * capacity + target, gate, -1)))
                    entries.append((source, z_start + action, ranks[target] - ranks[source]))
                    adjacency_actions.append(action)
                    gate += 1
                else:
                    # A missing neighbour keeps Z=N and receives a wall penalty.
                    entries.append((source, wall_start + action, 1))
            for plane, offset in ((1, 4), (2, 5)):
                entries.extend(((plane * capacity + source, z_start + offset, ranks[source]),
                                (source, z_start + offset, -ranks[source])))
            entries.append((3 * capacity + source, length_index, 1))
        layers = [_layer(sizes[0], sizes[1], entries, bias)]

        # Layer 2: Z[0:6], wrap hinges[6:18], walls[18:22], occupied[22:26],
        # and two hinges at the half-capacity body-length cutoff[26:28].
        entries, bias = [], np.zeros(28)
        for i in range(6):
            entries.extend(((z_start + i, i, 1), (z_start + i, 6 + i, -1),
                            (z_start + i, 12 + i, -1)))
            bias[6 + i], bias[12 + i] = capacity + 1, capacity
        entries.extend((wall_start + i, 18 + i, 1) for i in range(4))
        entries.extend((index, 22 + action, 1)
                       for index, action in enumerate(adjacency_actions))
        entries.extend(((length_index, 26, -1), (length_index, 27, -1)))
        bias[26], bias[27] = capacity // 2, capacity // 2 - 1
        layers.append(_layer(sizes[1], 28, entries, bias))

        # Layer 3: positive modular distances[0:6], wall/occupancy[6:10],
        # and a binary permission for shortcuts[10]. Equal ranks give distance N.
        entries, bias = [], np.zeros(11)
        for i in range(6):
            entries.extend(((i, i, 1), (6 + i, i, capacity), (12 + i, i, -capacity)))
            bias[i] = -capacity
        for i in range(4):
            entries.extend(((18 + i, 6 + i, 1), (22 + i, 6 + i, 1)))
        entries.extend(((26, 10, 1), (27, 10, -1)))
        layers.append(_layer(28, 11, entries, bias))

        # Layer 4: candidate distances, overshoot of food, overshoot of tail,
        # density violations, and copied wall/occupancy indicators (four each).
        entries, bias = [], np.zeros(20)
        for action in range(4):
            entries.extend(((action, action, 1),
                            (action, 4 + action, 1), (4, 4 + action, -1),
                            (action, 8 + action, 1), (5, 8 + action, -1),
                            (action, 12 + action, 1), (10, 12 + action, -capacity),
                            (6 + action, 16 + action, 1)))
            bias[12 + action] = -1
        layers.append(_layer(11, 20, entries, bias))

        # Layer 5: one unit violation outweighs every valid forward distance.
        entries = []
        for action in range(4):
            entries.append((action, action, 1))
            entries.extend((offset + action, action, -(capacity + 1))
                           for offset in (4, 8, 12, 16))
        layers.append(_layer(20, 4, entries))
        return layers

    @property
    def parameter_count(self):
        """Stored scalar coefficients, including zero coefficients and all biases."""
        return sum(layer.weights.size + layer.bias.size for layer in self.layers)

    @property
    def dense_parameter_count(self):
        """Parameters if each sparse affine map were stored as a dense matrix."""
        return sum(layer.input_size * layer.output_size + layer.output_size
                   for layer in self.layers)

    def _cell(self, point):
        try:
            x, y = point
        except (TypeError, ValueError) as exc:
            raise ValueError('Cells must contain two integer coordinates.') from exc
        if (isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
                or isinstance(y, (bool, np.bool_)) or not isinstance(y, Integral)
                or not 0 <= x < self.cols or not 0 <= y < self.rows):
            raise ValueError('Cells must have integer coordinates inside the board.')
        return int(y) * self.cols + int(x)

    def encode_state(self, body, food):
        """Validate a head-first body and pack raw planes without deriving actions."""
        try:
            cells = [self._cell(point) for point in body]
        except TypeError as exc:
            raise ValueError('Body must be a nonempty sequence of grid cells.') from exc
        capacity = self.cols * self.rows
        if not cells or len(cells) >= capacity or len(set(cells)) != len(cells):
            raise ValueError('An active body must have unique cells and leave room for food.')
        for previous, current in zip(cells, cells[1:]):
            dx = abs(previous % self.cols - current % self.cols)
            dy = abs(previous // self.cols - current // self.cols)
            if dx + dy != 1:
                raise ValueError('Consecutive body cells must share a grid edge.')
        food_cell = self._cell(food)
        if food_cell in cells:
            raise ValueError('An active food cell must be outside the body.')
        values = np.zeros(4 * capacity, dtype=np.float64)
        values[cells[0]] = values[capacity + food_cell] = values[2 * capacity + cells[-1]] = 1
        values[3 * capacity + np.asarray(cells)] = 1
        return values

    def _evaluate(self, values):
        for layer in self.layers[:-1]:
            values = np.maximum(layer.forward(values), 0)
        return self.layers[-1].forward(values)

    def forward_state(self, body, food):
        """Return UP, RIGHT, DOWN, LEFT logits from raw state, with no action filter."""
        return self._evaluate(self.encode_state(body, food))

    def forward_observation(self, vector):
        """Evaluate a validated binary observation using stored affine/ReLU layers.

        Raw planes do not encode body order. This entry point checks their binary
        values and head/food/tail/body consistency; encode_state also checks the
        ordered body's contiguity. No validation computes allowed moves.
        """
        capacity = self.cols * self.rows
        try:
            values = np.asarray(vector)
            if values.shape != (4 * capacity,) or values.dtype.kind not in 'biuf':
                raise ValueError('Observation must be a numeric vector of four grid planes.')
            values = values.astype(np.float64, copy=False)
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError('Invalid Snake observation.') from exc
        if not np.isfinite(values).all() or not np.all((values == 0) | (values == 1)):
            raise ValueError('Observation values must be finite and binary.')
        head, food, tail, body = values.reshape(4, capacity)
        if (not np.all(values[:3 * capacity].reshape(3, capacity).sum(axis=1) == 1)
                or np.any(head > body) or np.any(tail > body) or np.any(food * body)
                or (np.array_equal(head, tail) and body.sum() != 1)):
            raise ValueError('Observation head, food, tail and body planes are inconsistent.')
        return self._evaluate(values)

    def save(self, path):
        """Write all mutable sparse arrays; dimensions describe their architecture."""
        payload = {
            'kind': self.policy_name, 'version': WEIGHT_VERSION,
            'cols': self.cols, 'rows': self.rows,
            'architecture': _architecture(self.cols, self.rows),
            'layers': [{
                'input_size': layer.input_size, 'output_size': layer.output_size,
                'src': layer.src.tolist(), 'dst': layer.dst.tolist(),
                'weights': layer.weights.tolist(), 'bias': layer.bias.tolist(),
            } for layer in self.layers],
        }
        # One compact line per layer keeps the file small and diffs readable.
        def dump(value):
            return json.dumps(value, separators=(',', ':'), allow_nan=False)

        header = ',\n'.join(f' {dump(key)}:{dump(value)}'
                             for key, value in payload.items() if key != 'layers')
        layers = ',\n'.join(f'  {dump(layer)}' for layer in payload['layers'])
        text = '{\n' + header + ',\n "layers":[\n' + layers + '\n ]\n}\n'
        Path(path).write_text(text, encoding='utf-8')

    @classmethod
    def load(cls, path):
        """Validate v2 arrays and load them exactly; never rebuild policy weights.

        Valid dimensions and finite weights do not certify safe actions after
        editing. Empty or repeated sparse entries are legitimate affine maps.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError('Invalid Snake shortcut weight JSON.') from exc
        if (not isinstance(payload, dict) or payload.get('kind') != cls.policy_name
                or type(payload.get('version')) is not int
                or payload['version'] != WEIGHT_VERSION):
            raise ValueError('Unsupported Snake shortcut weight format.')
        cols, rows = _dimensions(payload.get('cols'), payload.get('rows'))
        sizes = _architecture(cols, rows)
        architecture = payload.get('architecture')
        if (not isinstance(architecture, list) or any(type(n) is not int for n in architecture)
                or architecture != sizes):
            raise ValueError('Snake shortcut architecture does not match the board.')
        records = payload.get('layers')
        if not isinstance(records, list) or len(records) != len(sizes) - 1:
            raise ValueError('Snake shortcut weights must contain five sparse layers.')
        layers = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError('Each sparse layer must be an object.')
            input_size, output_size = sizes[index:index + 2]
            if (type(record.get('input_size')) is not int or record['input_size'] != input_size
                    or type(record.get('output_size')) is not int
                    or record['output_size'] != output_size):
                raise ValueError('Sparse layer dimensions do not match the architecture.')
            arrays = {}
            for name in ('src', 'dst', 'weights', 'bias'):
                raw = record.get(name)
                is_index = name in ('src', 'dst')
                types = (int,) if is_index else (int, float)
                if not isinstance(raw, list) or any(type(value) not in types for value in raw):
                    raise ValueError(f'Sparse {name} must be a flat numeric array without booleans.')
                try:
                    array = np.asarray(raw, dtype=np.intp if is_index else np.float64)
                except (TypeError, OverflowError, ValueError) as exc:
                    raise ValueError(f'Invalid sparse {name} values.') from exc
                if not np.isfinite(array).all():
                    raise ValueError(f'Sparse {name} values must be finite.')
                arrays[name] = array
            if (arrays['src'].shape != arrays['weights'].shape
                    or arrays['dst'].shape != arrays['weights'].shape
                    or arrays['bias'].shape != (output_size,)):
                raise ValueError('Sparse index, weight and bias lengths do not match.')
            if (np.any(arrays['src'] < 0) or np.any(arrays['src'] >= input_size)
                    or np.any(arrays['dst'] < 0) or np.any(arrays['dst'] >= output_size)):
                raise ValueError('Sparse indices are outside the layer dimensions.')
            layers.append(SparseLayer(**arrays, input_size=input_size, output_size=output_size))
        net = cls.__new__(cls)
        net.cols, net.rows = cols, rows
        net.cycle_ranks = _cycle_ranks(cols, rows)
        net.layers = layers
        return net
