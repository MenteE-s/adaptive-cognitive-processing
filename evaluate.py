"""
Comprehensive evaluation of ACP vs baselines.
Reports accuracy, routing accuracy, and deliberation rate by difficulty.
Saves results to charts/evaluation_results.json for chart generation.
"""

import numpy as np
import json
import os
from acp_model import ACP
from data import generate_dataset, fill_episodic_memory
from baselines import AlwaysFast, AlwaysDeliberate, FixedMLP, RandomRouter, train_baseline
from train import train_acp, train_all_baselines


def evaluate_model(model, X, Y, Y_route, name="Model"):
    """Evaluate a single model. Returns dict of metrics."""
    correct = 0
    route_correct = 0
    delib_count = 0
    n = len(X)

    diff_stats = {
        'easy': {'correct': 0, 'total': 0, 'delib': 0, 'route_correct': 0},
        'medium': {'correct': 0, 'total': 0, 'delib': 0, 'route_correct': 0},
        'hard': {'correct': 0, 'total': 0, 'delib': 0, 'route_correct': 0},
    }

    n_third = n // 3
    diff_labels = ['easy'] * n_third + ['medium'] * n_third + ['hard'] * (n - 2 * n_third)

    for i in range(n):
        x = X[i]
        y = Y[i][0]
        y_r = Y_route[i][0]
        diff = diff_labels[i]

        out, deliberated = model.predict(x)

        if abs(out - y) < 0.5:
            correct += 1
            diff_stats[diff]['correct'] += 1
        diff_stats[diff]['total'] += 1

        if deliberated == int(y_r):
            route_correct += 1
            diff_stats[diff]['route_correct'] += 1

        if deliberated:
            delib_count += 1
            diff_stats[diff]['delib'] += 1

    results = {
        'name': name,
        'accuracy': correct / n,
        'route_accuracy': route_correct / n,
        'delib_rate': delib_count / n,
        'easy_acc': diff_stats['easy']['correct'] / max(diff_stats['easy']['total'], 1),
        'medium_acc': diff_stats['medium']['correct'] / max(diff_stats['medium']['total'], 1),
        'hard_acc': diff_stats['hard']['correct'] / max(diff_stats['hard']['total'], 1),
        'easy_delib_rate': diff_stats['easy']['delib'] / max(diff_stats['easy']['total'], 1),
        'medium_delib_rate': diff_stats['medium']['delib'] / max(diff_stats['medium']['total'], 1),
        'hard_delib_rate': diff_stats['hard']['delib'] / max(diff_stats['hard']['total'], 1),
    }
    return results


def run_full_experiment(n_samples=5000, input_dim=32, hidden_dim=48,
                        output_dim=1, memory_size=40, epochs=100, lr=0.01,
                        n_seeds=10, alpha=1.0, lam=0.1, seed_base=42):
    """Run experiment across multiple seeds and aggregate results."""
    all_seed_results = []

    for seed in range(seed_base, seed_base + n_seeds):
        X, Y, Y_route = generate_dataset(n_samples=n_samples, seed=seed,
                                         input_dim=input_dim)

        # Train ACP
        model, history = train_acp(X, Y, Y_route, input_dim=input_dim,
                                   hidden_dim=hidden_dim, output_dim=output_dim,
                                   memory_size=memory_size, epochs=epochs,
                                   lr=lr, alpha=alpha, lam=lam, seed=seed)

        # Train baselines
        baselines = train_all_baselines(X, Y, input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim,
                                        epochs=epochs, lr=lr, seed=seed)

        # Evaluate all
        seed_results = []
        seed_results.append(evaluate_model(model, X, Y, Y_route, name="ACP"))

        for name, bmodel in baselines.items():
            seed_results.append(evaluate_model(bmodel, X, Y, Y_route, name=name))

        all_seed_results.append(seed_results)

    # Aggregate across seeds
    model_names = [r['name'] for r in all_seed_results[0]]
    aggregated = []

    for mname in model_names:
        accs = [r['accuracy'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        route_accs = [r['route_accuracy'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        delib_rates = [r['delib_rate'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        easy_accs = [r['easy_acc'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        medium_accs = [r['medium_acc'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        hard_accs = [r['hard_acc'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        easy_delibs = [r['easy_delib_rate'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]
        hard_delibs = [r['hard_delib_rate'] for seed_res in all_seed_results for r in seed_res if r['name'] == mname]

        aggregated.append({
            'name': mname,
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'route_acc_mean': float(np.mean(route_accs)), 'route_acc_std': float(np.std(route_accs)),
            'delib_rate_mean': float(np.mean(delib_rates)), 'delib_rate_std': float(np.std(delib_rates)),
            'easy_acc_mean': float(np.mean(easy_accs)), 'easy_acc_std': float(np.std(easy_accs)),
            'medium_acc_mean': float(np.mean(medium_accs)), 'medium_acc_std': float(np.std(medium_accs)),
            'hard_acc_mean': float(np.mean(hard_accs)), 'hard_acc_std': float(np.std(hard_accs)),
            'easy_delib_mean': float(np.mean(easy_delibs)), 'easy_delib_std': float(np.std(easy_delibs)),
            'hard_delib_mean': float(np.mean(hard_delibs)), 'hard_delib_std': float(np.std(hard_delibs)),
        })

    # Save JSON for charts
    os.makedirs('charts', exist_ok=True)
    with open('charts/evaluation_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def print_aggregated_table(aggregated):
    """Print aggregated results with mean ± std."""
    print("\n" + "=" * 110)
    print(f"{'Model':<18} {'Acc':>12} {'RouteAcc':>13} {'DelibRate':>13} | "
          f"{'EasyAcc':>11} {'MedAcc':>11} {'HardAcc':>11} | "
          f"{'EasyDelib':>12} {'HardDelib':>12}")
    print("-" * 110)
    for r in aggregated:
        print(f"{r['name']:<18} {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} "
              f"{r['route_acc_mean']:.3f}±{r['route_acc_std']:.3f} "
              f"{r['delib_rate_mean']:.3f}±{r['delib_rate_std']:.3f} | "
              f"{r['easy_acc_mean']:.3f}±{r['easy_acc_std']:.3f} "
              f"{r['medium_acc_mean']:.3f}±{r['medium_acc_std']:.3f} "
              f"{r['hard_acc_mean']:.3f}±{r['hard_acc_std']:.3f} | "
              f"{r['easy_delib_mean']:.3f}±{r['easy_delib_std']:.3f} "
              f"{r['hard_delib_mean']:.3f}±{r['hard_delib_std']:.3f}")
    print("=" * 110)


if __name__ == "__main__":
    aggregated = run_full_experiment(
        n_samples=5000, input_dim=32, hidden_dim=48,
        output_dim=1, memory_size=40, epochs=100, lr=0.01,
        n_seeds=10, alpha=1.0, lam=0.1, seed_base=42
    )
    print_aggregated_table(aggregated)