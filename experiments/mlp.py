"""Minimal NumPy ReLU MLP trained with Adam (no deep-learning framework needed)."""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0

    @property
    def params(self):
        return [*self.W, *self.b]

    @property
    def parameter_count(self):
        return int(sum(p.size for p in self.params))

    def forward(self, x, keep=False):
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            acts.append(x)
        return (x, acts) if keep else x

    def step(self, x, grad_fn, lr):
        out, acts = self.forward(x, keep=True)
        loss, grad = grad_fn(out)
        gw, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gw[i] = acts[i].T @ grad
            gb[i] = grad.sum(0)
            if i:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)
        self._t += 1
        for k, (p, g) in enumerate(zip(self.params, [*gw, *gb])):
            self._m[k] = 0.9 * self._m[k] + 0.1 * g
            self._v[k] = 0.999 * self._v[k] + 0.001 * g * g
            m = self._m[k] / (1 - 0.9 ** self._t)
            v = self._v[k] / (1 - 0.999 ** self._t)
            p -= lr * m / (np.sqrt(v) + 1e-8)
        return float(loss)


def mse(target):
    def fn(out):
        diff = out - target
        return np.mean(diff ** 2), 2 * diff / diff.size
    return fn


def cross_entropy(labels):
    def fn(out):
        z = out - out.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        n = len(labels)
        loss = -np.mean(np.log(p[np.arange(n), labels] + 1e-12))
        p[np.arange(n), labels] -= 1
        return loss, p / n
    return fn
