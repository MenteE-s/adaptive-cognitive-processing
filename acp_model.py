"""
Adaptive Cognitive Processing (ACP) - NumPy implementation.

Architecture:
  - Fast pathway: linear model (cheap, cannot fit nonlinear targets)
  - Deliberate pathway: MLP over [x, fast_out] (expensive, fits nonlinear targets)
  - Monitor: linear input + episodic novelty feature (1 - max memory similarity)
  - Episodic memory: fixed keys from familiar (easy) inputs
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class ACP:
    def __init__(self, input_dim=16, hidden_dim=48, output_dim=1,
                 memory_size=30, use_novelty=True, seed=42):
        rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        self.use_novelty = use_novelty

        # ---- Fast pathway (linear, cheap) ----
        self.Wf = rng.randn(input_dim, output_dim) * 0.5
        self.bf = np.zeros(output_dim)

        # ---- Deliberate pathway (MLP over [x, fast_out]) ----
        self.Wd1 = rng.randn(input_dim + output_dim, hidden_dim) * 0.5
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = rng.randn(hidden_dim, output_dim) * 0.5
        self.bd2 = np.zeros(output_dim)

        # ---- Monitor: linear input + novelty feature ----
        self.Wm = rng.randn(input_dim, 1) * 0.5
        self.w_novelty = np.array([1.0])
        self.bm = np.zeros(1)

        # ---- Episodic memory keys (fixed) ----
        self.memory_keys = rng.randn(memory_size, input_dim) * 0.5

    def cosine_similarity(self, x):
        x = x.reshape(-1)
        x_norm = np.linalg.norm(x) + 1e-8
        k_norm = np.linalg.norm(self.memory_keys, axis=1) + 1e-8
        return (self.memory_keys @ x) / (k_norm * x_norm)

    def forward(self, x, return_cache=False):
        x = x.reshape(-1)

        # Fast path
        fast_out = x @ self.Wf + self.bf  # (output_dim,)

        # Episodic novelty
        sim = self.cosine_similarity(x)
        max_sim = float(np.max(sim))
        novelty = 1.0 - max_sim

        # Monitor. The novelty term can be disabled for the x-only ablation.
        novelty_term = self.w_novelty * novelty if self.use_novelty else 0.0
        p_logit = x @ self.Wm + novelty_term + self.bm
        p = sigmoid(p_logit)[0]

        # Deliberate path
        xd = np.concatenate([x, fast_out])
        zd = xd @ self.Wd1 + self.bd1
        hd = relu(zd)
        deliberate_out = hd @ self.Wd2 + self.bd2

        # Soft blend
        final_out = (1.0 - p) * fast_out + p * deliberate_out

        if return_cache:
            cache = {
                'x': x, 'fast_out': fast_out, 'max_sim': max_sim,
                'novelty': novelty, 'p_logit': p_logit, 'p': p,
                'xd': xd, 'zd': zd, 'hd': hd,
                'deliberate_out': deliberate_out
            }
            return final_out, cache
        return final_out

    def predict(self, x):
        final_out, cache = self.forward(x, return_cache=True)
        deliberated = int(cache['p'] > 0.5)
        out = cache['deliberate_out'] if deliberated else cache['fast_out']
        return out, deliberated

    def compute_gradients(self, x, y, y_route, alpha=1.0, lam=0.1,
                          cost_fast=0.0, cost_deliberate=1.0):
        x = x.reshape(-1)
        final_out, cache = self.forward(x, return_cache=True)

        p = cache['p']
        fast_out = cache['fast_out']
        deliberate_out = cache['deliberate_out']
        xd = cache['xd']
        zd = cache['zd']
        hd = cache['hd']
        max_sim = cache['max_sim']
        novelty = cache['novelty']
        x_vec = cache['x']

        # Task loss gradient
        dL_dfinal = 2.0 * (final_out - y)  # MSE

        # Routing probability gradient (chain rule through the blend)
        dL_task_dp = float((dL_dfinal * (-fast_out + deliberate_out))[0])
        eps = 1e-8
        dL_route_dp = -(y_route / (p + eps) - (1.0 - y_route) / (1.0 - p + eps))
        dL_compute_dp = cost_deliberate - cost_fast
        dL_dp = dL_task_dp + alpha * dL_route_dp + lam * dL_compute_dp

        grads = {
            'Wf': np.zeros_like(self.Wf), 'bf': np.zeros_like(self.bf),
            'Wd1': np.zeros_like(self.Wd1), 'bd1': np.zeros_like(self.bd1),
            'Wd2': np.zeros_like(self.Wd2), 'bd2': np.zeros_like(self.bd2),
            'Wm': np.zeros_like(self.Wm), 'w_novelty': np.zeros_like(self.w_novelty),
            'bm': np.zeros_like(self.bm),
        }

        # Monitor gradients
        dL_dlogit = float(dL_dp * p * (1.0 - p))
        grads['Wm'] = np.outer(x_vec, np.array([dL_dlogit]))
        if self.use_novelty:
            grads['w_novelty'] = np.array([dL_dlogit * novelty])
        else:
            grads['w_novelty'] = np.zeros_like(self.w_novelty)
        grads['bm'] = np.array([dL_dlogit])

        # Split to pathways
        dL_dfast = (1.0 - p) * dL_dfinal
        dL_ddeliberate = p * dL_dfinal

        # Deliberate path gradients
        grads['Wd2'] = np.outer(hd, dL_ddeliberate)
        grads['bd2'] = dL_ddeliberate
        dL_dhd = dL_ddeliberate @ self.Wd2.T
        dL_dzd = dL_dhd * relu_derivative(zd)
        grads['Wd1'] = np.outer(xd, dL_dzd)
        grads['bd1'] = dL_dzd

        # Fast path also feeds deliberate path via xd = [x, fast_out]
        dL_dxd = dL_dzd @ self.Wd1.T
        dL_dfast_total = dL_dfast + dL_dxd[self.input_dim:]

        # Fast path gradients (linear)
        grads['Wf'] = np.outer(x_vec, dL_dfast_total)
        grads['bf'] = dL_dfast_total

        return grads, final_out, p