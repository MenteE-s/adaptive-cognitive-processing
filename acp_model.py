"""
Adaptive Cognitive Processing (ACP) - NumPy Smoke Test
Architecture: Fast path + Episodic Memory Monitor + Deliberate Refinement
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative_from_output(p):
    """Derivative of sigmoid given its output p."""
    return p * (1.0 - p)


class ACP:
    def __init__(self, input_dim=8, hidden_dim=12, output_dim=1,
                 memory_size=15, seed=42):
        rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_size = memory_size

        # ---- Fast pathway (System 1) ----
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

        # ---- Deliberate pathway (System 2) ----
        # Takes [input, fast_output] and refines the answer
        self.Wd1 = rng.randn(input_dim + output_dim, hidden_dim) * 0.5
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.bd2 = np.zeros(output_dim)

        # ---- Monitor (orchestrator) ----
        # Predicts probability to deliberate
        self.Wm = rng.randn(input_dim, 1) * 0.5
        self.bm = np.zeros(1)

        # ---- Episodic memory (hand-filled, fixed) ----
        self.memory_keys = rng.randn(memory_size, input_dim) * 0.5
        self.memory_values = rng.randn(memory_size, output_dim) * 0.5

    def cosine_similarity(self, x):
        """Similarity between input and all memory keys."""
        x = x.reshape(-1)
        x_norm = np.linalg.norm(x)
        k_norm = np.linalg.norm(self.memory_keys, axis=1)
        dot = self.memory_keys @ x
        return dot / (k_norm * x_norm + 1e-8)

    def forward(self, x, return_cache=False):
        x = x.reshape(-1)

        # --- Fast path ---
        z1 = x @ self.W1 + self.b1
        h1 = relu(z1)
        fast_out = h1 @ self.W2 + self.b2  # (output_dim,)

        # --- Monitor ---
        p_logit = x @ self.Wm + self.bm
        p = sigmoid(p_logit)[0]  # scalar probability to deliberate

        # --- Deliberate path ---
        xd = np.concatenate([x, fast_out])  # (input_dim + output_dim,)
        zd = xd @ self.Wd1 + self.bd1
        hd = relu(zd)
        deliberate_out = hd @ self.Wd2 + self.bd2  # (output_dim,)

        # --- Soft blend (differentiable routing) ---
        final_out = (1.0 - p) * fast_out + p * deliberate_out

        if return_cache:
            cache = {
                'x': x, 'z1': z1, 'h1': h1, 'fast_out': fast_out,
                'p_logit': p_logit, 'p': p,
                'xd': xd, 'zd': zd, 'hd': hd,
                'deliberate_out': deliberate_out
            }
            return final_out, cache
        return final_out

    def predict(self, x):
        """Hard-decision inference. Returns (output, deliberated_flag)."""
        final_out, cache = self.forward(x, return_cache=True)
        deliberated = int(cache['p'] > 0.5)
        out = cache['deliberate_out'] if deliberated else cache['fast_out']
        return out, deliberated

    def compute_gradients(self, x, y, y_route, alpha=1.0, lam=0.1,
                          cost_fast=0.0, cost_deliberate=1.0):
        """
        Analytical gradients for one sample.
        y: target scalar
        y_route: ground-truth routing label (0=fast, 1=deliberate)
        alpha: weight for routing supervision loss
        lam: weight for computational cost penalty
        """
        x = x.reshape(-1)
        final_out, cache = self.forward(x, return_cache=True)

        p = cache['p']
        fast_out = cache['fast_out']
        deliberate_out = cache['deliberate_out']
        h1 = cache['h1']
        z1 = cache['z1']
        xd = cache['xd']
        zd = cache['zd']
        hd = cache['hd']
        x_vec = cache['x']

        # --- Loss gradients ---
        dL_dfinal = 2.0 * (final_out - y)  # MSE

        # dL/dp
        dL_task_dp = float((-fast_out + deliberate_out)[0])
        eps = 1e-8
        dL_route_dp = -(y_route / (p + eps) - (1.0 - y_route) / (1.0 - p + eps))
        dL_compute_dp = cost_deliberate - cost_fast
        dL_dp = dL_task_dp + alpha * dL_route_dp + lam * dL_compute_dp

        # --- Gradients ---
        grads = {
            'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2),
            'Wd1': np.zeros_like(self.Wd1), 'bd1': np.zeros_like(self.bd1),
            'Wd2': np.zeros_like(self.Wd2), 'bd2': np.zeros_like(self.bd2),
            'Wm': np.zeros_like(self.Wm), 'bm': np.zeros_like(self.bm),
        }

        # Monitor: dL/d(logit) = dL/dp * dp/d(logit)
        d_sigmoid = sigmoid_derivative_from_output(np.array([p]))[0]
        dL_dlogit = float(dL_dp * d_sigmoid)
        grads['Wm'] = np.outer(x_vec, np.array([dL_dlogit]))
        grads['bm'] = np.array([dL_dlogit])

        # Split gradient to fast_out and deliberate_out
        dL_dfast = (1.0 - p) * dL_dfinal
        dL_ddeliberate = p * dL_dfinal

        # Deliberate path
        grads['Wd2'] = np.outer(hd, dL_ddeliberate)
        grads['bd2'] = dL_ddeliberate
        dL_dhd = dL_ddeliberate @ self.Wd2.T
        dL_dzd = dL_dhd * relu_derivative(zd)
        grads['Wd1'] = np.outer(xd, dL_dzd)
        grads['bd1'] = dL_dzd

        # fast_out also feeds into deliberate path via xd = [x, fast_out]
        dL_dxd = dL_dzd @ self.Wd1.T
        dL_dfast_from_deliberate = dL_dxd[self.input_dim:]
        dL_dfast_total = dL_dfast + dL_dfast_from_deliberate

        # Fast path
        grads['W2'] = np.outer(h1, dL_dfast_total)
        grads['b2'] = dL_dfast_total
        dL_dh1 = dL_dfast_total @ self.W2.T
        dL_dz1 = dL_dh1 * relu_derivative(z1)
        grads['W1'] = np.outer(x_vec, dL_dz1)
        grads['b1'] = dL_dz1

        return grads, final_out, p