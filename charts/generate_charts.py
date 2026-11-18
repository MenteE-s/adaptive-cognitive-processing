"""
Generate charts for the ACP preprint.

Requires:
  charts/evaluation_results.json  (from evaluate.py)
  charts/ablation_novelty.json    (from ablation_novelty.py, optional)
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

COLOR_ACP = '#1f77b4'
COLOR_OTHER = '#d3d3d3'
COLORS_DIFF = ['#2ca02c', '#ff7f0e', '#d62728']
COLOR_MEM = '#1f77b4'
COLOR_XONLY = '#d62728'


def load_results(path='charts/evaluation_results.json'):
    if not os.path.exists(path):
        print(f"Missing {path}. Run evaluate.py first.")
        raise SystemExit(1)
    with open(path, 'r') as f:
        return json.load(f)


def load_ablation(path='charts/ablation_novelty.json'):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)['summary']


def add_footer(fig):
    fig.text(0.99, 0.005, 'MenteE AI | Syed Syab Ahmad Shah',
             ha='right', va='bottom', fontsize=7, color='grey')


def annotate_bars(ax, fmt='{:.2f}', fontsize=9, dy=2):
    """Put the value on top of every bar in the axes."""
    for bar in ax.patches:
        h = bar.get_height()
        if h <= 0:
            continue
        ax.annotate(fmt.format(h),
                    (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va='bottom', fontsize=fontsize)


def plot_model_comparison(aggregated, save_path='charts/model_comparison.png'):
    models = [DISPLAY[r['name']] for r in aggregated]
    accs = [r['accuracy_mean'] for r in aggregated]
    acc_stds = [r['accuracy_std'] for r in aggregated]
    delibs = [r['delib_rate_mean'] for r in aggregated]
    delib_stds = [r['delib_rate_std'] for r in aggregated]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLOR_ACP if r['name'] == 'ACP' else COLOR_OTHER for r in aggregated]

    axes[0].bar(models, accs, yerr=acc_stds, capsize=5,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Task Accuracy by Model', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.08)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    annotate_bars(axes[0], fontsize=8)

    axes[1].bar(models, delibs, yerr=delib_stds, capsize=5,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Average Deliberation Rate by Model', fontsize=13,
                      fontweight='bold')
    axes[1].set_ylabel('Fraction of inputs deliberated')
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)
    annotate_bars(axes[1], fontsize=8)

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
           color=COLORS_DIFF[0], alpha=0.9, edgecolor='black')
    ax.bar(x, medium, width, yerr=med_s, capsize=3, label='Medium',
           color=COLORS_DIFF[1], alpha=0.9, edgecolor='black')
    ax.bar(x + width, hard, width, yerr=hard_s, capsize=3, label='Hard',
           color=COLORS_DIFF[2], alpha=0.9, edgecolor='black')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Difficulty Level', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.08)
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
    bars = ax.bar(labels, rates, yerr=stds, capsize=6,
                  color=COLORS_DIFF, alpha=0.9, edgecolor='black')

    ax.set_ylabel('Deliberation Rate')
    ax.set_title('ACP Routing Decision by Task Difficulty', fontsize=13,
                 fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                f'{rate * 100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_ablation(summary, save_path='charts/ablation.png'):
    if summary is None:
        print("No ablation summary; skipping ablation chart.")
        return

    keys = ['with_novelty', 'x_only']
    acc = [summary[k]['accuracy_mean'] for k in keys]
    acc_std = [summary[k]['accuracy_std'] for k in keys]
    route = [summary[k]['route_acc_mean'] for k in keys]
    route_std = [summary[k]['route_acc_std'] for k in keys]
    easy = [summary[k]['easy_delib_mean'] for k in keys]
    med = [summary[k]['medium_delib_mean'] for k in keys]
    hard = [summary[k]['hard_delib_mean'] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(2)
    width = 0.35
    labels = ['x + novelty', 'x only']

    axes[0].bar(x - width / 2, acc, width, yerr=acc_std, capsize=4,
                label='Task accuracy', color=COLOR_ACP, edgecolor='black')
    axes[0].bar(x + width / 2, route, width, yerr=route_std, capsize=4,
                label='Routing accuracy', color='#ff7f0e', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Accuracy With and Without Memory', fontsize=12,
                      fontweight='bold')
    axes[0].legend(loc='lower left')
    axes[0].grid(axis='y', alpha=0.3)
    annotate_bars(axes[0], fontsize=8)

    x3 = np.arange(3)
    axes[1].bar(x3 - width / 2, [easy[0], med[0], hard[0]], width,
                label='x + novelty', color=COLOR_MEM, edgecolor='black')
    axes[1].bar(x3 + width / 2, [easy[1], med[1], hard[1]], width,
                label='x only', color=COLOR_XONLY, edgecolor='black')
    axes[1].set_xticks(x3)
    axes[1].set_xticklabels(['Easy', 'Medium', 'Hard'])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel('Deliberation rate')
    axes[1].set_title('Where the Deliberation Goes', fontsize=12,
                      fontweight='bold')
    axes[1].legend(loc='center right')
    axes[1].grid(axis='y', alpha=0.3)
    annotate_bars(axes[1], fontsize=8)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_pareto(aggregated, save_path='charts/pareto.png'):
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in aggregated:
        color = COLOR_ACP if r['name'] == 'ACP' else COLOR_OTHER
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
        label = f"{DISPLAY[r['name']]} ({r['accuracy_mean'] * 100:.1f}%)"
        ax.annotate(label, (r['delib_rate_mean'], r['accuracy_mean']),
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


def plot_hero(results, acp_row, ablation, save_path='charts/hero_summary.png'):
    """One-glance summary image for the preprint and social posts."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    width = 0.35

    # A. Routing gradient
    ax = axes[0][0]
    rates = [acp_row['easy_delib_mean'], acp_row['medium_delib_mean'],
             acp_row['hard_delib_mean']]
    stds = [acp_row['easy_delib_std'], acp_row['medium_delib_std'],
            acp_row['hard_delib_std']]
    ax.bar(['Easy', 'Medium', 'Hard'], rates, yerr=stds, capsize=5,
           color=COLORS_DIFF, alpha=0.9, edgecolor='black')
    for i, r in enumerate(rates):
        ax.text(i, r + 0.03, f'{r * 100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Deliberation rate')
    ax.set_title('A. Deliberates only when the input is hard',
                 fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)

    # B. Accuracy vs deliberation
    ax = axes[0][1]
    for r in results:
        color = COLOR_ACP if r['name'] == 'ACP' else COLOR_OTHER
        ax.errorbar(r['delib_rate_mean'], r['accuracy_mean'],
                    xerr=r['delib_rate_std'], yerr=r['accuracy_std'],
                    fmt='o', markersize=9, color=color,
                    ecolor='gray', capsize=3, markeredgecolor='black')
        label = f"{DISPLAY[r['name']]} ({r['accuracy_mean'] * 100:.1f}%)"
        if r['name'] == 'FixedMLP':
            xytext, ha = (-8, 16), 'right'
        elif r['name'] == 'AlwaysDeliberate':
            xytext, ha = (-8, -26), 'right'
        else:
            xytext, ha = (8, 8), 'left'
        ax.annotate(label, (r['delib_rate_mean'], r['accuracy_mean']),
                    xytext=xytext, textcoords='offset points',
                    fontsize=9, ha=ha)
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Deliberation rate')
    ax.set_ylabel('Task accuracy')
    ax.set_title('B. Same accuracy, two-thirds of the deliberation',
                 fontsize=12, fontweight='bold', loc='left')
    ax.grid(alpha=0.3)

    # C. Memory ablation
    ax = axes[1][0]
    if ablation:
        vals = [ablation['with_novelty']['accuracy_mean'],
                ablation['x_only']['accuracy_mean']]
        errs = [ablation['with_novelty']['accuracy_std'],
                ablation['x_only']['accuracy_std']]
        bars = ax.bar(['x + novelty', 'x only'], vals, yerr=errs, capsize=5,
                      color=[COLOR_MEM, COLOR_XONLY], alpha=0.9,
                      edgecolor='black')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f'{v * 100:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Task accuracy')
    ax.set_title('C. Memory does real work (3 seeds)',
                 fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.3)

    # D. Capacity gap on Hard inputs
    ax = axes[1][1]
    acp = next(r for r in results if r['name'] == 'ACP')
    fast = next(r for r in results if r['name'] == 'AlwaysFast')
    acp_vals = [acp['easy_acc_mean'], acp['medium_acc_mean'],
                acp['hard_acc_mean']]
    fast_vals = [fast['easy_acc_mean'], fast['medium_acc_mean'],
                 fast['hard_acc_mean']]
    x3 = np.arange(3)
    ax.bar(x3 - width / 2, acp_vals, width, label='ACP',
           color=COLOR_ACP, edgecolor='black')
    ax.bar(x3 + width / 2, fast_vals, width, label='Linear fast path',
           color='#7f7f7f', edgecolor='black')
    for xi, v in zip(x3 - width / 2, acp_vals):
        ax.text(xi, v + 0.02, f'{v * 100:.0f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for xi, v in zip(x3 + width / 2, fast_vals):
        ax.text(xi, v + 0.02, f'{v * 100:.0f}%', ha='center', va='bottom',
                fontsize=9)
    ax.set_xticks(x3)
    ax.set_xticklabels(['Easy', 'Medium', 'Hard'])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Task accuracy')
    ax.set_title('D. The linear fast path cannot handle Hard inputs',
                 fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Think While You Speak: Adaptive Cognitive Processing',
                 fontsize=15, fontweight='bold', y=0.995)
    add_footer(fig)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = load_results()
    ablation = load_ablation()
    acp_row = next(r for r in results if r['name'] == 'ACP')
    os.makedirs('charts', exist_ok=True)

    plot_model_comparison(results)
    plot_difficulty_breakdown(results)
    plot_routing_by_difficulty(acp_row)
    plot_pareto(results)
    plot_ablation(ablation)
    plot_hero(results, acp_row, ablation)
    print("All charts generated in charts/ folder")
