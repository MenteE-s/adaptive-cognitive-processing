"""
Baseline models for ACP comparison. NumPy, CPU only.

AlwaysFast:        linear model (same capacity as ACP fast path)
AlwaysDeliberate:  MLP (same capacity as ACP deliberate path)
FixedMLP:          wider MLP, no routing
RandomRouter:      random gate between linear and MLP
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class LinearFast:
    """Cheap linear model. Cannot fit nonlinear targets."""

    def __init__(self, input_dim=16, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.Wf = rng.randn(input_dim, output_dim) * 0.5
        self.bf = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        return x @ self.Wf + self.bf, 0


class AlwaysDeliberate:
    """MLP only. Always deliberates."""

    def __init__(self, input_dim=16, hidden_dim=48, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        h = relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, 1


class FixedMLP:
    """Wider MLP with no routing. Always computes the same amount."""

    def __init__(self, input_dim=16, hidden_dim=56, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        h = relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, 1


class RandomRouter:
    """Random gate between linear and MLP. Gate does not learn."""

    def __init__(self, input_dim=16, hidden_dim=48, output_dim=1,
                 p_deliberate=0.5, seed=42):
        self.p_deliberate = p_deliberate
        self.rng = np.random.RandomState(seed)
        rng = np.random.RandomState(seed)
        self.Wf = rng.randn(input_dim, output_dim) * 0.5
        self.bf = np.zeros(output_dim)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        x = x.reshape(-1)
        deliberated = int(self.rng.rand() < self.p_deliberate)
        if deliberated:
            h = relu(x @ self.W1 + self.b1)
            return h @ self.W2 + self.b2, 1
        return x @ self.Wf + self.bf, 0


def train_linear(model, X, Y, epochs=100, lr=0.01, seed=42):
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        for idx in rng.permutation(len(X)):
            x = X[idx].reshape(-1)
            y = Y[idx][0]
            out = x @ model.Wf + model.bf
            dL = 2.0 * (out - y)
            model.Wf -= lr * np.outer(x, dL)
            model.bf -= lr * dL
    return model


def train_mlp(model, X, Y, epochs=100, lr=0.01, seed=42):
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        for idx in rng.permutation(len(X)):
            x = X[idx].reshape(-1)
            y = Y[idx][0]
            h = relu(x @ model.W1 + model.b1)
            out = h @ model.W2 + model.b2
            dL = 2.0 * (out - y)
            dW2 = np.outer(h, dL)
            db2 = dL
            dL_dh = dL @ model.W2.T
            dL_dz = dL_dh * (h > 0)
            dW1 = np.outer(x, dL_dz)
            db1 = dL_dz
            model.W2 -= lr * dW2
            model.b2 -= lr * db2
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
    return model


def train_all_baselines(X, Y, input_dim=16, hidden_dim=48, output_dim=1,
                        epochs=100, lr=0.01, seed=42):
    baselines = {}

    bf = LinearFast(input_dim=input_dim, output_dim=output_dim, seed=seed)
    baselines['AlwaysFast'] = train_linear(bf, X, Y, epochs=epochs, lr=lr, seed=seed)

    bd = AlwaysDeliberate(input_dim=input_dim, hidden_dim=hidden_dim,
                          output_dim=output_dim, seed=seed)
    baselines['AlwaysDeliberate'] = train_mlp(bd, X, Y, epochs=epochs, lr=lr, seed=seed)

    fm = FixedMLP(input_dim=input_dim, hidden_dim=hidden_dim + 8,
                  output_dim=output_dim, seed=seed)
    baselines['FixedMLP'] = train_mlp(fm, X, Y, epochs=epochs, lr=lr, seed=seed)

    rr = RandomRouter(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=output_dim, p_deliberate=0.5, seed=seed)
    train_linear(rr, X, Y, epochs=epochs, lr=lr, seed=seed)
    train_mlp(rr, X, Y, epochs=epochs, lr=lr, seed=seed)
    baselines['RandomRouter'] = rr

    return baselines