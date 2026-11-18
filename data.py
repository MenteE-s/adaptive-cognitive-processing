"""
Synthetic dataset for ACP.

Three difficulty levels, designed so that a linear fast path
genuinely fails on medium and hard inputs:

  Easy:   y = 2*x0 + 1          (linear, familiar cluster near dim 0)
  Medium: y = x1^2 + 1          (quadratic, novel cluster near dim 1)
  Hard:   y = x2*x3 + x4*x5     (product interactions, uniform inputs)

Returns X, Y, Y_route, Y_diff:
  Y_route: 0 = fast, 1 = deliberate
  Y_diff:  0 = easy, 1 = medium, 2 = hard
"""

import numpy as np


def generate_dataset(n_samples=5000, seed=123, input_dim=16):
    rng = np.random.RandomState(seed)

    n_easy = n_samples // 3
    n_medium = n_samples // 3
    n_hard = n_samples - n_easy - n_medium

    # Easy: familiar cluster near dim 0, linear target
    easy_center = np.zeros(input_dim)
    easy_center[0] = 1.0
    X_easy = rng.randn(n_easy, input_dim) * 0.2 + easy_center
    Y_easy = 2.0 * X_easy[:, 0:1] + 1.0
    R_easy = np.zeros((n_easy, 1))
    D_easy = np.zeros((n_easy, 1), dtype=int)

    # Medium: novel cluster near dim 1, quadratic target
    med_center = np.zeros(input_dim)
    med_center[1] = 1.0
    X_med = rng.randn(n_medium, input_dim) * 0.3 + med_center
    Y_med = X_med[:, 1:2] ** 2 + 1.0
    R_med = np.ones((n_medium, 1))
    D_med = np.ones((n_medium, 1), dtype=int)

    # Hard: uniform inputs, product interactions (linear path cannot fit)
    X_hard = rng.uniform(-1.0, 1.0, size=(n_hard, input_dim))
    Y_hard = X_hard[:, 2:3] * X_hard[:, 3:4] + X_hard[:, 4:5] * X_hard[:, 5:6]
    R_hard = np.ones((n_hard, 1))
    D_hard = np.full((n_hard, 1), 2, dtype=int)

    X = np.vstack([X_easy, X_med, X_hard])
    Y = np.vstack([Y_easy, Y_med, Y_hard])
    R = np.vstack([R_easy, R_med, R_hard])
    D = np.vstack([D_easy, D_med, D_hard])

    perm = rng.permutation(n_samples)
    return X[perm], Y[perm], R[perm], D[perm]


def fill_episodic_memory(model, n_easy=15, n_random=15, seed=42):
    """
    Memory contains easy-cluster keys (familiar) plus random keys.
    Easy inputs have high max cosine similarity (low novelty).
    Medium and hard inputs are novel (higher novelty).
    """
    rng = np.random.RandomState(seed)
    input_dim = model.input_dim

    easy_keys = np.zeros((n_easy, input_dim))
    easy_keys[:, 0] = 1.0
    easy_keys += rng.randn(n_easy, input_dim) * 0.1

    rand_keys = rng.uniform(-1.0, 1.0, size=(n_random, input_dim))

    keys = np.vstack([easy_keys, rand_keys])

    model.memory_keys = keys
    model.memory_size = n_easy + n_random
    return model