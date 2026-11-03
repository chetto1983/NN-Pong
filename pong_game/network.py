"""The original 45-parameter analytic ReLU controller and fold helpers."""

import numpy as np

from .config import BALL_MIN_Y, DEADZONE, SPAN


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def reflect_y(y: float) -> float:
    """Fold a ball-center coordinate between its radius-adjusted walls."""
    phase = (y - BALL_MIN_Y) % (2.0 * SPAN)
    return BALL_MIN_Y + (phase if phase <= SPAN else 2.0 * SPAN - phase)


class PongNetV5:
    """Fixed 2 -> 8 -> 2 -> 1 network; positive output moves downward.

    Layer one constructs triangular-fold ReLU hinges. Layers two and three
    compare that fold with paddle_y, including the deadzone. The diagnostic
    fold readout never overrides the network output.
    """

    def __init__(self, delta: float = DEADZONE):
        if delta < 0:
            raise ValueError("delta deve essere non negativo")
        self.delta = float(delta)
        self.W1 = np.array([
            [0, 1], [0, -1],
            [0, 1], [0, -1],
            [0, 1], [0, -1],
            [1, 0], [-1, 0],
        ], dtype=float)
        self.b1 = np.array([
            -BALL_MIN_Y, BALL_MIN_Y,
            -BALL_MIN_Y - SPAN, BALL_MIN_Y - SPAN,
            -BALL_MIN_Y - 2 * SPAN, BALL_MIN_Y - 2 * SPAN,
            0, 0,
        ], dtype=float)
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
        fold = float(BALL_MIN_Y + u[0] + u[1] - 2*u[2] - 2*u[3]
                     + 2*u[4] + 2*u[5])
        return output, {
            "layer1": u, "layer2": v,
            "fold": fold, "error": fold - paddle_y,
            "prediction": y_pred,
        }
