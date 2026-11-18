"""
Generate charts for the ACP preprint.
Requires charts/evaluation_results.json, produced by evaluate.py.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json

DISPLAY = {
    'ACP': 'ACP',
    'AlwaysFast': 'Always Fast',
    'AlwaysDeliberate': 'Always Deliberate',
    'FixedMLP': 'Fixed MLP',
    'RandomRouter': 'Random Router',
}


def load_results(path='charts/evaluation_results.json'):
    if not os.path.exists(path):
        print(f"Missing {path}. Run evaluate.py first.")
        raise SystemExit(1)
    with open(path, 'r') as f:
        return json.load(f)


def add_footer(fig):
    fig.text(0.99, 0.005, 'MenteE AI | Syed Syab Ahmad Shah',
             ha='right', va='bottom', fontsize=7, color='grey')


def plot_model_comparison(aggregated, save_path='charts/model_comparison.png'):
    models = [DISPLAY[r['name']] for r in aggregated]
    accs = [r['accuracy_mean'] for r in aggregated]
    acc_stds = [r['accuracy_std'] for r in aggregated]
    delibs = [r['delib_rate_mean'] for r in aggregated]
    delib_stds = [r['delib_rate_std'] for r in aggregated]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4' if m == 'ACP' else '#d3d3d3' for m in models]

    axes[0].bar(models, accs, yerr=acc_stds, capsize=5,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Task Accuracy by Model', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(models, delibs, yerr=delib_stds, capsize=5,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Average Deliberation Rate by Model', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Fraction of inputs deliberated')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_difficulty_breakdown(aggregated, save_path='charts/difficulty_breakdown.png'):
    models = [DISPLAY[r['name']] for r in aggregated]
    easy = [r['easy_acc_mean'] for r in aggregated]
    medium = [r['medium_acc_mean'] for r in aggregated]
    hard = [r['hard_acc_mean'] for r in aggregated]
    easy_s = [r['easy_acc_std'] for r in aggregated]
    med_s = [r['medium_acc_std'] for r in aggregated]
    hard_s = [r['hard_acc_std'] for r in aggregated]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, easy, width, yerr=easy_s, capsize=3, label='Easy',
           color='#2ca02c', alpha=0.85, edgecolor='black')
    ax.bar(x, medium, width, yerr=med_s, capsize=3, label='Medium',
           color='#ff7f0e', alpha=0.85, edgecolor='black')
    ax.bar(x + width, hard, width, yerr=hard_s, capsize=3, label='Hard',
           color='#d62728', alpha=0.85, edgecolor='black')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Difficulty Level', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    add_footer(fig)
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
    bars = ax.bar(labels, rates, yerr=stds, capsize=6, color=colors,
                  alpha=0.85, edgecolor='black')

    ax.set_ylabel('Deliberation Rate')
    ax.set_title('ACP Routing Decision by Task Difficulty', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_pareto(aggregated, save_path='charts/pareto.png'):
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in aggregated:
        color = '#1f77b4' if r['name'] == 'ACP' else '#d3d3d3'
        ax.errorbar(r['delib_rate_mean'], r['accuracy_mean'],
                    xerr=r['delib_rate_std'], yerr=r['accuracy_std'],
                    fmt='o', markersize=10, color=color,
                    ecolor='gray', capsize=4, markeredgecolor='black')
        if r['name'] == 'FixedMLP':
            xytext, ha = (-8, 8), 'right'
        elif r['name'] == 'AlwaysDeliberate':
            xytext, ha = (-8, -16), 'right'
        else:
            xytext, ha = (8, 8), 'left'
        ax.annotate(DISPLAY[r['name']], (r['delib_rate_mean'], r['accuracy_mean']),
                    xytext=xytext, textcoords='offset points',
                    fontsize=9, ha=ha)

    ax.set_xlabel('Deliberation Rate (fraction using the expensive path)')
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Accuracy vs Deliberation Rate', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = load_results()
    os.makedirs('charts', exist_ok=True)

    plot_model_comparison(results)
    plot_difficulty_breakdown(results)
    acp_row = next(r for r in results if r['name'] == 'ACP')
    plot_routing_by_difficulty(acp_row)
    plot_pareto(results)
    print("All charts generated in charts/ folder")