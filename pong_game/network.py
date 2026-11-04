"""The 15-parameter analytic ReLU controller and fold helpers.

Division of labour, stated explicitly:

* Python computes the ballistic intercept ``ball_y + ball_vy * t`` and reduces
  it modulo the reflection period ``2 * SPAN`` (``wrap_period``). This handles
  any number of wall bounces and is *not* done by the network.
* The network receives the paddle position and that reduced coordinate. Its
  weights fold the coordinate once (a single triangular reflection), compare
  it with the paddle and apply a deadzone. Nothing else is involved.
"""

import numpy as np

from .config import BALL_MIN_Y, DEADZONE, HEIGHT, SPAN

PERIOD = 2.0 * SPAN
PREDICTION_MIN = BALL_MIN_Y
PREDICTION_MAX = BALL_MIN_Y + PERIOD
PADDLE_INPUT_MIN = 0.0
PADDLE_INPUT_MAX = float(HEIGHT)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_period(y: float) -> float:
    """Map any raw ball-centre coordinate into [BALL_MIN_Y, BALL_MIN_Y + 2*SPAN]."""
    return BALL_MIN_Y + (y - BALL_MIN_Y) % PERIOD


def reflect_y(y: float) -> float:
    """Fold a ball-center coordinate between its radius-adjusted walls."""
    phase = (y - BALL_MIN_Y) % PERIOD
    return BALL_MIN_Y + (phase if phase <= SPAN else PERIOD - phase)


class PongNet:
    """Fixed 2 -> 2 -> 2 -> 1 ReLU network, 15 parameters; positive output moves down.

    Inputs: ``paddle_y`` in [0, HEIGHT] and ``y_pred`` already reduced by
    ``wrap_period``. On that domain:

    * h0 = ReLU(y_pred - paddle_y + HEIGHT) is always active, i.e. linear;
    * h1 = ReLU(y_pred - BALL_MIN_Y - SPAN) is the single reflection hinge;
    * e  = h0 - 2*h1 - HEIGHT = fold(y_pred) - paddle_y;
    * output = ReLU(e - delta) - ReLU(-e - delta).

    Outside the domain the weights would silently compute something else, so
    ``forward`` rejects such inputs instead of returning a wrong command.
    """

    def __init__(self, delta: float = DEADZONE):
        if delta < 0:
            raise ValueError("delta deve essere non negativo")
        self.delta = float(delta)
        self.W1 = np.array([[-1.0, 1.0],
                            [0.0, 1.0]])
        self.b1 = np.array([float(HEIGHT), -BALL_MIN_Y - SPAN])
        self.W2 = np.array([[1.0, -2.0],
                            [-1.0, 2.0]])
        self.b2 = np.array([-HEIGHT - self.delta, HEIGHT - self.delta])
        self.W3 = np.array([[1.0, -1.0]])
        self.b3 = np.array([0.0])

    @property
    def parameter_count(self) -> int:
        return sum(getattr(self, name).size
                   for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    def forward(self, paddle_y: float, y_pred: float):
        if not PREDICTION_MIN <= y_pred <= PREDICTION_MAX:
            raise ValueError("y_pred deve essere ridotto con wrap_period")
        if not PADDLE_INPUT_MIN <= paddle_y <= PADDLE_INPUT_MAX:
            raise ValueError("paddle_y fuori dal dominio della rete")
        x = np.array([paddle_y, y_pred], dtype=float)
        u = self.relu(self.W1 @ x + self.b1)
        v = self.relu(self.W2 @ u + self.b2)
        output = float((self.W3 @ v + self.b3)[0])
        # Diagnostic readout from hidden units; it never overrides the output.
        error = float(u[0] - 2.0 * u[1] - HEIGHT)
        return output, {
            "layer1": u, "layer2": v,
            "fold": error + paddle_y, "error": error,
            "prediction": y_pred,
        }


PongNetV5 = PongNet  # Historical name kept importable.
