"""
Generate real charts for the ACP preprint.
Run this AFTER evaluate.py completes.
Reads charts/evaluation_results.json and saves PNG files to charts/.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def load_results(path='charts/evaluation_results.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def plot_model_comparison(aggregated, save_path='charts/model_comparison.png'):
    models = [r['name'] for r in aggregated]
    accs = [r['accuracy_mean'] for r in aggregated]
    acc_stds = [r['accuracy_std'] for r in aggregated]
    delibs = [r['delib_rate_mean'] for r in aggregated]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4' if m == 'ACP' else '#d3d3d3' for m in models]

    axes[0].bar(models, accs, yerr=acc_stds, capsize=5,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Task Accuracy by Model', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(models, delibs, capsize=5,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Average Deliberation Rate by Model', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Fraction of inputs deliberated')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_difficulty_breakdown(aggregated, save_path='charts/difficulty_breakdown.png'):
    models = [r['name'] for r in aggregated]
    easy = [r['easy_acc_mean'] for r in aggregated]
    medium = [r['medium_acc_mean'] for r in aggregated]
    hard = [r['hard_acc_mean'] for r in aggregated]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, easy, width, label='Easy', color='#2ca02c', alpha=0.8)
    ax.bar(x, medium, width, label='Medium', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, hard, width, label='Hard', color='#d62728', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Difficulty Level', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_routing_by_difficulty(acp_row, save_path='charts/routing_by_difficulty.png'):
    labels = ['Easy', 'Medium', 'Hard']
    rates = [acp_row['easy_delib_mean'],
             acp_row['medium_delib_mean'],
             acp_row['hard_delib_mean']]
    stds = [acp_row['easy_delib_std'],
            acp_row['medium_delib_std'],
            acp_row['hard_delib_std']]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = ax.bar(labels, rates, yerr=stds, capsize=6, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Deliberation Rate')
    ax.set_title('ACP Routing Decision by Task Difficulty', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_pareto(aggregated, save_path='charts/pareto.png'):
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in aggregated:
        color = '#1f77b4' if r['name'] == 'ACP' else '#d3d3d3'
        ax.scatter(r['delib_rate_mean'], r['accuracy_mean'], s=120,
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.annotate(r['name'], (r['delib_rate_mean'], r['accuracy_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Average Compute Cost (Deliberation Rate)')
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Accuracy vs Compute Cost (Pareto Frontier)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = load_results()
    if results is None:
        print("No charts/evaluation_results.json found. Run evaluate.py first.")
        print("Using placeholder data for layout.")
        aggregated = [
            {'name': 'ACP', 'accuracy_mean': 0.992, 'accuracy_std': 0.005,
             'route_acc_mean': 0.954, 'route_acc_std': 0.003,
             'delib_rate_mean': 0.646, 'delib_rate_std': 0.005,
             'easy_acc_mean': 0.991, 'easy_acc_std': 0.007,
             'medium_acc_mean': 0.992, 'medium_acc_std': 0.006,
             'hard_acc_mean': 0.993, 'hard_acc_std': 0.004,
             'easy_delib_mean': 0.642, 'easy_delib_std': 0.010,
             'medium_delib_mean': 0.650, 'medium_delib_std': 0.010,
             'hard_delib_mean': 0.652, 'hard_delib_std': 0.011},
            {'name': 'AlwaysFast', 'accuracy_mean': 0.993, 'accuracy_std': 0.002,
             'route_acc_mean': 0.333, 'route_acc_std': 0.000,
             'delib_rate_mean': 0.0, 'delib_rate_std': 0.0,
             'easy_acc_mean': 0.993, 'easy_acc_std': 0.002,
             'medium_acc_mean': 0.993, 'medium_acc_std': 0.003,
             'hard_acc_mean': 0.993, 'hard_acc_std': 0.004,
             'easy_delib_mean': 0.0, 'easy_delib_std': 0.0,
             'medium_delib_mean': 0.0, 'medium_delib_std': 0.0,
             'hard_delib_mean': 0.0, 'hard_delib_std': 0.0},
            {'name': 'AlwaysDeliberate', 'accuracy_mean': 0.993, 'accuracy_std': 0.002,
             'route_acc_mean': 0.667, 'route_acc_std': 0.000,
             'delib_rate_mean': 1.0, 'delib_rate_std': 0.0,
             'easy_acc_mean': 0.993, 'easy_acc_std': 0.002,
             'medium_acc_mean': 0.993, 'medium_acc_std': 0.003,
             'hard_acc_mean': 0.993, 'hard_acc_std': 0.004,
             'easy_delib_mean': 1.0, 'easy_delib_std': 0.0,
             'medium_delib_mean': 1.0, 'medium_delib_std': 0.0,
             'hard_delib_mean': 1.0, 'hard_delib_std': 0.0},
            {'name': 'FixedMLP', 'accuracy_mean': 0.994, 'accuracy_std': 0.003,
             'route_acc_mean': 0.667, 'route_acc_std': 0.000,
             'delib_rate_mean': 1.0, 'delib_rate_std': 0.0,
             'easy_acc_mean': 0.994, 'easy_acc_std': 0.003,
             'medium_acc_mean': 0.995, 'medium_acc_std': 0.003,
             'hard_acc_mean': 0.994, 'hard_acc_std': 0.003,
             'easy_delib_mean': 1.0, 'easy_delib_std': 0.0,
             'medium_delib_mean': 1.0, 'medium_delib_std': 0.0,
             'hard_delib_mean': 1.0, 'hard_delib_std': 0.0},
            {'name': 'RandomRouter', 'accuracy_mean': 0.141, 'accuracy_std': 0.062,
             'route_acc_mean': 0.505, 'route_acc_std': 0.006,
             'delib_rate_mean': 0.507, 'delib_rate_std': 0.012,
             'easy_acc_mean': 0.149, 'easy_acc_std': 0.058,
             'medium_acc_mean': 0.136, 'medium_acc_std': 0.059,
             'hard_acc_mean': 0.138, 'hard_acc_std': 0.070,
             'easy_delib_mean': 0.498, 'easy_delib_std': 0.010,
             'medium_delib_mean': 0.500, 'medium_delib_std': 0.010,
             'hard_delib_mean': 0.507, 'hard_delib_std': 0.016},
        ]
    else:
        aggregated = results

    plot_model_comparison(aggregated)
    plot_difficulty_breakdown(aggregated)
    acp_row = next(r for r in aggregated if r['name'] == 'ACP')
    plot_routing_by_difficulty(acp_row)
    plot_pareto(aggregated)
    print("All charts generated in charts/ folder")