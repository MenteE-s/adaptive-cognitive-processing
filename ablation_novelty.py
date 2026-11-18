"""
Ablation: does the episodic novelty feature change routing?

Trains the same ACP model twice per seed:
  with_novelty: monitor sees x plus 1 - max cosine similarity to memory
  x_only:       monitor sees x only

This isolates whether the memory readout contributes to the routing policy.
"""

import json
import os
import numpy as np

from data import generate_dataset
from train import train_acp
from evaluate import evaluate_model


def run(n_seeds=3, seed_base=42):
    rows = {'with_novelty': [], 'x_only': []}

    for seed in range(seed_base, seed_base + n_seeds):
        X, Y, Y_route, Y_diff = generate_dataset(
            n_samples=5000, seed=seed, input_dim=16
        )

        for label, use_novelty in [('with_novelty', True), ('x_only', False)]:
            model, _ = train_acp(
                X, Y, Y_route, input_dim=16, hidden_dim=48, output_dim=1,
                memory_size=30, epochs=100, lr=0.01, alpha=1.0, lam=0.1,
                use_novelty=use_novelty, seed=seed
            )
            r = evaluate_model(model, X, Y, Y_route, Y_diff, name=label)
            r['seed'] = seed
            rows[label].append(r)
            print(f"{label:14s} seed={seed} "
                  f"acc={r['accuracy']:.3f} "
                  f"route={r['route_accuracy']:.3f} "
                  f"delib={r['delib_rate']:.3f}")

    summary = {}
    for label, rs in rows.items():
        summary[label] = {
            'accuracy_mean': float(np.mean([r['accuracy'] for r in rs])),
            'accuracy_std': float(np.std([r['accuracy'] for r in rs])),
            'route_acc_mean': float(np.mean([r['route_accuracy'] for r in rs])),
            'route_acc_std': float(np.std([r['route_accuracy'] for r in rs])),
            'delib_rate_mean': float(np.mean([r['delib_rate'] for r in rs])),
            'delib_rate_std': float(np.std([r['delib_rate'] for r in rs])),
            'easy_delib_mean': float(np.mean([r['easy_delib_rate'] for r in rs])),
            'medium_delib_mean': float(np.mean([r['medium_delib_rate'] for r in rs])),
            'hard_delib_mean': float(np.mean([r['hard_delib_rate'] for r in rs])),
            'n_seeds': len(rs),
        }

    os.makedirs('charts', exist_ok=True)
    with open('charts/ablation_novelty.json', 'w') as f:
        json.dump({'summary': summary, 'per_seed': rows}, f, indent=2)

    print("\nSummary")
    for label, s in summary.items():
        print(f"{label:14s} acc={s['accuracy_mean']:.3f}"
              f"±{s['accuracy_std']:.3f} "
              f"route={s['route_acc_mean']:.3f}"
              f"±{s['route_acc_std']:.3f} "
              f"delib={s['delib_rate_mean']:.3f}"
              f"±{s['delib_rate_std']:.3f}")

    return summary


if __name__ == '__main__':
    run(n_seeds=3, seed_base=42)
