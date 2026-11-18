"""
Numerical gradient check for ACP. Verifies analytical grads against finite differences.
Run: python test_gradients.py
"""

import numpy as np
from acp_model import ACP
from data import fill_episodic_memory


def loss_fn(model, x, y, y_route, alpha=1.0, lam=0.1):
    final_out = model.forward(x)
    task = float(((final_out - y) ** 2)[0])

    _, cache = model.forward(x, return_cache=True)
    p = float(cache['p'])
    eps = 1e-12
    route = -(y_route * np.log(p + eps) + (1.0 - y_route) * np.log(1.0 - p + eps))
    compute = lam * p

    return task + alpha * float(route) + compute


def main():
    rng = np.random.RandomState(0)

    model = ACP(input_dim=5, hidden_dim=7, output_dim=1,
                memory_size=6, seed=1)
    model = fill_episodic_memory(model, n_easy=3, n_random=3, seed=2)

    x = rng.randn(5)
    y = 1.0
    y_route = 1.0
    alpha = 1.0
    lam = 0.1

    grads, _, _ = model.compute_gradients(x, y, y_route, alpha=alpha, lam=lam)

    param_names = ['Wf', 'bf', 'Wd1', 'bd1', 'Wd2', 'bd2',
                   'Wm', 'w_novelty', 'bm']

    max_rel_error = 0.0
    eps = 1e-5

    for name in param_names:
        param = getattr(model, name)
        g_ana = grads[name].reshape(-1)
        flat = param.reshape(-1)
        n_params = len(flat)

        # Check up to 8 random entries per parameter tensor
        idxs = rng.choice(n_params, size=min(8, n_params), replace=False)

        for i in idxs:
            old = flat[i]

            flat[i] = old + eps
            loss_plus = loss_fn(model, x, y, y_route, alpha, lam)

            flat[i] = old - eps
            loss_minus = loss_fn(model, x, y, y_route, alpha, lam)

            flat[i] = old

            g_num = (loss_plus - loss_minus) / (2.0 * eps)
            g_ref = g_ana[i]

            denom = max(abs(g_num), abs(g_ref), 1e-8)
            rel_err = abs(g_num - g_ref) / denom
            max_rel_error = max(max_rel_error, rel_err)

            print(f"{name}[{i:>4d}] num={g_num:+.6f} ana={g_ref:+.6f} "
                  f"rel_err={rel_err:.2e}")

    print("\nMax relative error:", f"{max_rel_error:.2e}")

    if max_rel_error < 1e-4:
        print("PASS: gradients are correct.")
    else:
        print("FAIL: gradient mismatch. Investigate.")


if __name__ == "__main__":
    main()