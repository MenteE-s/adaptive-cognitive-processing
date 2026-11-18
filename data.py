import numpy as np


def generate_dataset(n_samples=400, seed=123):
    """
    Synthetic regression task.
    Easy: input cluster around [1,0,0,...], target = 1.0, route = fast (0)
    Hard: input cluster around [0,1,0,...], target = 2.0, route = deliberate (1)
    """
    rng = np.random.RandomState(seed)
    input_dim = 8

    n_easy = n_samples // 2
    n_hard = n_samples - n_easy

    easy_means = np.zeros(input_dim)
    easy_means[0] = 1.0
    hard_means = np.zeros(input_dim)
    hard_means[1] = 1.0

    X_easy = rng.randn(n_easy, input_dim) * 0.15 + easy_means
    X_hard = rng.randn(n_hard, input_dim) * 0.15 + hard_means

    Y_easy = np.ones((n_easy, 1)) * 1.0
    Y_hard = np.ones((n_hard, 1)) * 2.0

    R_easy = np.zeros((n_easy, 1))  # fast
    R_hard = np.ones((n_hard, 1))   # deliberate

    X = np.vstack([X_easy, X_hard])
    Y = np.vstack([Y_easy, Y_hard])
    R = np.vstack([R_easy, R_hard])

    perm = rng.permutation(n_samples)
    return X[perm], Y[perm], R[perm]


def fill_episodic_memory(model, n_easy=10, n_random=5, seed=42):
    """
    Pre-fill memory so easy inputs have high similarity (match memory),
    hard inputs have low similarity (novel -> trigger deliberate).
    """
    rng = np.random.RandomState(seed)
    input_dim = model.input_dim

    easy_keys = np.zeros((n_easy, input_dim))
    easy_keys[:, 0] = 1.0
    easy_keys += rng.randn(n_easy, input_dim) * 0.1

    random_keys = rng.randn(n_random, input_dim) * 0.5

    keys = np.vstack([easy_keys, random_keys])
    values = np.vstack([
        np.ones((n_easy, 1)) * 1.0,
        np.ones((n_random, 1)) * 1.5
    ])

    model.memory_keys = keys
    model.memory_values = values
    return model