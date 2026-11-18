"""
Baseline models for comparison with ACP.
All implemented in NumPy, CPU-only.
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class AlwaysFast:
    """System 1 only. Always uses fast path, never deliberates."""

    def __init__(self, input_dim=32, hidden_dim=48, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        h = relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, 0  # never deliberates


class AlwaysDeliberate:
    """System 2 only. Always uses deliberate path."""

    def __init__(self, input_dim=32, hidden_dim=48, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        h = relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, 1  # always deliberates


class FixedMLP:
    """A single MLP with comparable total parameters to ACP combined.
    No routing—always computes the same amount.
    """

    def __init__(self, input_dim=32, hidden_dim=56, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        h = relu(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, 1  # treat as deliberate for compute cost accounting


class RandomRouter:
    """Routes to fast or deliberate with fixed probability (no learning).
    Uses two independent subnetworks like ACP but random gate.
    """

    def __init__(self, input_dim=32, hidden_dim=48, output_dim=1,
                 p_deliberate=0.5, seed=42):
        rng = np.random.RandomState(seed)
        self.p = p_deliberate

        # Fast subnetwork
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

        # Deliberate subnetwork
        self.Wd1 = rng.randn(input_dim + output_dim, hidden_dim) * 0.5
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.bd2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        h1 = relu(x @ self.W1 + self.b1)
        fast_out = h1 @ self.W2 + self.b2

        xd = np.concatenate([x, fast_out])
        hd = relu(xd @ self.Wd1 + self.bd1)
        deliberate_out = hd @ self.Wd2 + self.bd2

        deliberated = int(self.rng.rand() < self.p) if hasattr(self, 'rng') else int(x[0] > 0)
        # Use a deterministic pseudo-random based on input hash for reproducibility
        deliberated = int((abs(np.sum(x)) * 1000) % 100 < self.p * 100)
        out = deliberate_out if deliberated else fast_out
        return out, deliberated


def train_baseline(model, X, Y, epochs=100, lr=0.01):
    """Simple SGD training for baselines (MSE loss)."""
    rng = np.random.RandomState(42)
    for epoch in range(epochs):
        perm = rng.permutation(len(X))
        for idx in perm:
            x = X[idx]
            y = Y[idx][0]

            # Forward
            x_vec = x.reshape(-1)
            h = relu(x_vec @ model.W1 + model.b1)
            out = h @ model.W2 + model.b2

            # Gradients (MSE)
            dL_dout = 2.0 * (out - y)
            grads_W2 = np.outer(h, dL_dout)
            grads_b2 = dL_dout
            dL_dh = dL_dout @ model.W2.T
            dL_dz = dL_dh * (h > 0)
            grads_W1 = np.outer(x_vec, dL_dz)
            grads_b1 = dL_dz

            model.W2 -= lr * grads_W2
            model.b2 -= lr * grads_b2
            model.W1 -= lr * grads_W1
            model.b1 -= lr * grads_b1
    return model