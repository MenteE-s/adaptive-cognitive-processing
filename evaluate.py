"""
Full experiment for ACP vs baselines, with correct per-difficulty metrics.

Saves charts/evaluation_results.json for chart generation.
"""

import numpy as np
import json
import os
from data import generate_dataset
from train import train_acp
from baselines import train_all_baselines


def evaluate_model(model, X, Y, Y_route, Y_diff, name="Model"):
    n = len(X)
    correct = 0
    route_correct = 0
    delib_count = 0

    diff_stats = {
        0: {'correct': 0, 'total': 0, 'delib': 0, 'route_correct': 0},
        1: {'correct': 0, 'total': 0, 'delib': 0, 'route_correct': 0},
        2: {'correct': 0, 'total': 0, 'delib': 0, 'route_correct': 0},
    }

    for i in range(n):
        x = X[i]
        y = Y[i][0]
        y_r = Y_route[i][0]
        d = int(Y_diff[i][0])

        out, deliberated = model.predict(x)

        if abs(out - y) < 0.5:
            correct += 1
            diff_stats[d]['correct'] += 1
        diff_stats[d]['total'] += 1

        if deliberated == int(y_r):
            route_correct += 1
            diff_stats[d]['route_correct'] += 1

        if deliberated:
            delib_count += 1
            diff_stats[d]['delib'] += 1

    results = {
        'name': name,
        'accuracy': correct / n,
        'route_accuracy': route_correct / n,
        'delib_rate': delib_count / n,
        'easy_acc': diff_stats[0]['correct'] / max(diff_stats[0]['total'], 1),
        'medium_acc': diff_stats[1]['correct'] / max(diff_stats[1]['total'], 1),
        'hard_acc': diff_stats[2]['correct'] / max(diff_stats[2]['total'], 1),
        'easy_delib_rate': diff_stats[0]['delib'] / max(diff_stats[0]['total'], 1),
        'medium_delib_rate': diff_stats[1]['delib'] / max(diff_stats[1]['total'], 1),
        'hard_delib_rate': diff_stats[2]['delib'] / max(diff_stats[2]['total'], 1),
    }
    return results


def run_full_experiment(n_samples=5000, input_dim=16, hidden_dim=48,
                        output_dim=1, memory_size=30, epochs=100, lr=0.01,
                        n_seeds=10, alpha=1.0, lam=0.1, seed_base=42):
    all_seed_results = []

    for seed in range(seed_base, seed_base + n_seeds):
        X, Y, Y_route, Y_diff = generate_dataset(n_samples=n_samples, seed=seed,
                                                 input_dim=input_dim)

        model, history = train_acp(X, Y, Y_route, input_dim=input_dim,
                                   hidden_dim=hidden_dim, output_dim=output_dim,
                                   memory_size=memory_size, epochs=epochs,
                                   lr=lr, alpha=alpha, lam=lam, seed=seed)

        baselines = train_all_baselines(X, Y, input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim,
                                        epochs=epochs, lr=lr, seed=seed)

        seed_results = [evaluate_model(model, X, Y, Y_route, Y_diff,
                                       name="ACP")]
        for name, bmodel in baselines.items():
            seed_results.append(evaluate_model(bmodel, X, Y, Y_route, Y_diff,
                                               name=name))

        all_seed_results.append(seed_results)

    model_names = [r['name'] for r in all_seed_results[0]]
    aggregated = []

    for mname in model_names:
        rows = [r for seed_res in all_seed_results
                for r in seed_res if r['name'] == mname]

        def ms(key):
            vals = [r[key] for r in rows]
            return float(np.mean(vals)), float(np.std(vals))

        acc_m, acc_s = ms('accuracy')
        route_m, route_s = ms('route_accuracy')
        delib_m, delib_s = ms('delib_rate')
        easy_m, easy_s = ms('easy_acc')
        med_m, med_s = ms('medium_acc')
        hard_m, hard_s = ms('hard_acc')
        easy_d_m, easy_d_s = ms('easy_delib_rate')
        med_d_m, med_d_s = ms('medium_delib_rate')
        hard_d_m, hard_d_s = ms('hard_delib_rate')

        aggregated.append({
            'name': mname,
            'accuracy_mean': acc_m, 'accuracy_std': acc_s,
            'route_acc_mean': route_m, 'route_acc_std': route_s,
            'delib_rate_mean': delib_m, 'delib_rate_std': delib_s,
            'easy_acc_mean': easy_m, 'easy_acc_std': easy_s,
            'medium_acc_mean': med_m, 'medium_acc_std': med_s,
            'hard_acc_mean': hard_m, 'hard_acc_std': hard_s,
            'easy_delib_mean': easy_d_m, 'easy_delib_std': easy_d_s,
            'medium_delib_mean': med_d_m, 'medium_delib_std': med_d_s,
            'hard_delib_mean': hard_d_m, 'hard_delib_std': hard_d_s,
        })

    os.makedirs('charts', exist_ok=True)
    with open('charts/evaluation_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def print_aggregated_table(aggregated):
    print("\n" + "=" * 118)
    print(f"{'Model':<18} {'Acc':>12} {'RouteAcc':>13} {'DelibRate':>13} | "
          f"{'EasyAcc':>11} {'MedAcc':>11} {'HardAcc':>11} | "
          f"{'EasyDelib':>12} {'MedDelib':>11} {'HardDelib':>11}")
    print("-" * 118)
    for r in aggregated:
        print(f"{r['name']:<18} {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} "
              f"{r['route_acc_mean']:.3f}±{r['route_acc_std']:.3f} "
              f"{r['delib_rate_mean']:.3f}±{r['delib_rate_std']:.3f} | "
              f"{r['easy_acc_mean']:.3f}±{r['easy_acc_std']:.3f} "
              f"{r['medium_acc_mean']:.3f}±{r['medium_acc_std']:.3f} "
              f"{r['hard_acc_mean']:.3f}±{r['hard_acc_std']:.3f} | "
              f"{r['easy_delib_mean']:.3f}±{r['easy_delib_std']:.3f} "
              f"{r['medium_delib_mean']:.3f}±{r['medium_delib_std']:.3f} "
              f"{r['hard_delib_mean']:.3f}±{r['hard_delib_std']:.3f}")
    print("=" * 118)


if __name__ == "__main__":
    aggregated = run_full_experiment(
        n_samples=5000, input_dim=16, hidden_dim=48,
        output_dim=1, memory_size=30, epochs=100, lr=0.01,
        n_seeds=10, alpha=1.0, lam=0.1, seed_base=42
    )
    print_aggregated_table(aggregated)