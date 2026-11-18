import numpy as np


def generate_dataset(n_samples=5000, seed=123, input_dim=32):
    """
    Synthetic regression task with 3 difficulty levels.

    Easy:    linear cluster around dim 0, target = 1.0, route = fast (0)
    Medium:  linear cluster around dim 1 with higher noise, target = 1.5,
             route = deliberate (1)
    Hard:    non-linear interaction between dims 2 and 3, target = 2.0,
             route = deliberate (1)

    Returns X, Y, Y_route where:
      X: (n_samples, input_dim)
      Y: (n_samples, 1) target values
      Y_route: (n_samples, 1) ground-truth routing label (0=fast, 1=deliberate)
    """
    rng = np.random.RandomState(seed)

    n_easy = n_samples // 3
    n_medium = n_samples // 3
    n_hard = n_samples - n_easy - n_medium

    # ---- Easy ----
    easy_means = np.zeros(input_dim)
    easy_means[0] = 1.0
    X_easy = rng.randn(n_easy, input_dim) * 0.2 + easy_means
    Y_easy = np.ones((n_easy, 1)) * 1.0
    R_easy = np.zeros((n_easy, 1))  # fast

    # ---- Medium ----
    med_means = np.zeros(input_dim)
    med_means[1] = 1.0
    X_med = rng.randn(n_medium, input_dim) * 0.4 + med_means
    Y_med = np.ones((n_medium, 1)) * 1.5
    R_med = np.ones((n_medium, 1))  # deliberate

    # ---- Hard (non-linear interaction) ----
    X_hard = rng.randn(n_hard, input_dim) * 0.5
    # Signal: x2 * x3 > 0 -> embed in dim 4 so model can learn it with deliberation
    hard_signal = (X_hard[:, 2] * X_hard[:, 3] > 0).astype(float).reshape(-1, 1)
    X_hard[:, 4] = hard_signal.flatten() * 1.0
    Y_hard = np.ones((n_hard, 1)) * 2.0
    R_hard = np.ones((n_hard, 1))  # deliberate

    X = np.vstack([X_easy, X_med, X_hard])
    Y = np.vstack([Y_easy, Y_med, Y_hard])
    R = np.vstack([R_easy, R_med, R_hard])

    perm = rng.permutation(n_samples)
    return X[perm], Y[perm], R[perm]


def fill_episodic_memory(model, n_per_class=10, seed=42):
    """
    Pre-fill episodic memory so easy inputs match stored keys (high similarity),
    while medium/hard inputs are novel (low similarity -> trigger deliberate).
    """
    rng = np.random.RandomState(seed)
    input_dim = model.input_dim

    # Easy episodes: cluster around dim 0
    easy_keys = np.zeros((n_per_class, input_dim))
    easy_keys[:, 0] = 1.0
    easy_keys += rng.randn(n_per_class, input_dim) * 0.1

    # Medium episodes: cluster around dim 1
    med_keys = np.zeros((n_per_class, input_dim))
    med_keys[:, 1] = 1.0
    med_keys += rng.randn(n_per_class, input_dim) * 0.1

    # Random / novel episodes (noise)
    rand_keys = rng.randn(n_per_class, input_dim) * 0.5

    keys = np.vstack([easy_keys, med_keys, rand_keys])
    values = np.vstack([
        np.ones((n_per_class, 1)) * 1.0,
        np.ones((n_per_class, 1)) * 1.5,
        np.ones((n_per_class, 1)) * 1.5
    ])

    model.memory_keys = keys
    model.memory_values = values
    return model