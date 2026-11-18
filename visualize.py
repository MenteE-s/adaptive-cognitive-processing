"""
Visualization charts for the ACP preprint and social media (X, LinkedIn).
Generates:
  - results_overview.png: accuracy, routing, deliberation by model
  - difficulty_breakdown.png: accuracy per difficulty level
  - routing_by_difficulty.png: deliberation rate by task difficulty
  - pareto.png: accuracy vs compute cost
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_model_comparison(aggregated, save_path='results_overview.png'):
    """Bar chart: accuracy and deliberation rate by model."""
    models = [r['name'] for r in aggregated]
    accs = [r['accuracy_mean'] for r in aggregated]
    acc_stds = [r['accuracy_std'] for r in aggregated]
    delibs = [r['delib_rate_mean'] for r in aggregated]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].bar(models, accs, yerr=acc_stds, capsize=5,
                color=['steelblue' if m == 'ACP' else 'lightgray' for m in models],
                edgecolor='black', linewidth=0.5)
    axes[0].set_title('Task Accuracy by Model', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)

    # Deliberation rate
    axes[1].bar(models, delibs, capsize=5,
                color=['steelblue' if m == 'ACP' else 'lightgray' for m in models],
                edgecolor='black', linewidth=0.5)
    axes[1].set_title('Average Deliberation Rate by Model', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Fraction of inputs deliberated')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def plot_difficulty_breakdown(aggregated, save_path='difficulty_breakdown.png'):
    """Grouped bar chart: accuracy per difficulty level by model."""
    models = [r['name'] for r in aggregated]
    easy = [r['easy_acc_mean'] for r in aggregated]
    medium = [r['medium_acc_mean'] for r in aggregated]
    hard = [r['hard_acc_mean'] for r in aggregated]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, easy, width, label='Easy', color='#2ecc71', alpha=0.8)
    ax.bar(x, medium, width, label='Medium', color='#f39c12', alpha=0.8)
    ax.bar(x + width, hard, width, label='Hard', color='#e74c3c', alpha=0.8)

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


def plot_routing_by_difficulty(acp_results, save_path='routing_by_difficulty.png'):
    """Bar chart: ACP deliberation rate by task difficulty."""
    labels = ['Easy (should not think)', 'Medium', 'Hard (should think)']
    rates = [acp_results['easy_delib_mean'],
             acp_results['medium_delib_mean'],
             acp_results['hard_delib_mean']]
    stds = [acp_results['easy_delib_std'],
            acp_results['medium_delib_std'],
            acp_results['hard_delib_std']]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
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


def plot_pareto(aggregated, save_path='pareto.png'):
    """Accuracy vs average compute cost (deliberation rate as cost proxy)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in aggregated:
        color = 'steelblue' if r['name'] == 'ACP' else 'lightgray'
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


def plot_training_history(history, save_path='training_history.png'):
    """Training curves for ACP."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['loss'], color='black', linewidth=2)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['acc'], label='Accuracy', color='green', linewidth=2)
    axes[1].plot(history['route_acc'], label='Routing Accuracy', color='blue', linewidth=2)
    axes[1].set_title('Accuracy vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history['delib_rate'], color='orange', linewidth=2)
    axes[2].set_title('Deliberation Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


if __name__ == "__main__":
    # Demo with dummy aggregated data
    dummy = [
        {'name': 'ACP', 'accuracy_mean': 0.95, 'accuracy_std': 0.02,
         'route_acc_mean': 0.92, 'route_acc_std': 0.03,
         'delib_rate_mean': 0.67, 'delib_rate_std': 0.04,
         'easy_acc_mean': 0.98, 'easy_acc_std': 0.01,
         'medium_acc_mean': 0.94, 'medium_acc_std': 0.02,
         'hard_acc_mean': 0.93, 'hard_acc_std': 0.03,
         'easy_delib_mean': 0.10, 'easy_delib_std': 0.02,
         'medium_delib_mean': 0.85, 'medium_delib_std': 0.05,
         'hard_delib_mean': 0.95, 'hard_delib_std': 0.03},
        {'name': 'AlwaysFast', 'accuracy_mean': 0.55, 'accuracy_std': 0.03,
         'route_acc_mean': 0.33, 'route_acc_std': 0.02,
         'delib_rate_mean': 0.0, 'delib_rate_std': 0.0,
         'easy_acc_mean': 0.95, 'easy_acc_std': 0.02,
         'medium_acc_mean': 0.40, 'medium_acc_std': 0.04,
         'hard_acc_mean': 0.30, 'hard_acc_std': 0.03,
         'easy_delib_mean': 0.0, 'easy_delib_std': 0.0,
         'medium_delib_mean': 0.0, 'medium_delib_std': 0.0,
         'hard_delib_mean': 0.0, 'hard_delib_std': 0.0},
        {'name': 'AlwaysDeliberate', 'accuracy_mean': 0.90, 'accuracy_std': 0.02,
         'route_acc_mean': 0.67, 'route_acc_std': 0.03,
         'delib_rate_mean': 1.0, 'delib_rate_std': 0.0,
         'easy_acc_mean': 0.92, 'easy_acc_std': 0.02,
         'medium_acc_mean': 0.90, 'medium_acc_std': 0.02,
         'hard_acc_mean': 0.88, 'hard_acc_std': 0.02,
         'easy_delib_mean': 1.0, 'easy_delib_std': 0.0,
         'medium_delib_mean': 1.0, 'medium_delib_std': 0.0,
         'hard_delib_mean': 1.0, 'hard_delib_std': 0.0},
        {'name': 'FixedMLP', 'accuracy_mean': 0.85, 'accuracy_std': 0.03,
         'route_acc_mean': 0.67, 'route_acc_std': 0.02,
         'delib_rate_mean': 1.0, 'delib_rate_std': 0.0,
         'easy_acc_mean': 0.88, 'easy_acc_std': 0.02,
         'medium_acc_mean': 0.86, 'medium_acc_std': 0.03,
         'hard_acc_mean': 0.81, 'hard_acc_std': 0.03,
         'easy_delib_mean': 1.0, 'easy_delib_std': 0.0,
         'medium_delib_mean': 1.0, 'medium_delib_std': 0.0,
         'hard_delib_mean': 1.0, 'hard_delib_std': 0.0},
        {'name': 'RandomRouter', 'accuracy_mean': 0.70, 'accuracy_std': 0.04,
         'route_acc_mean': 0.50, 'route_acc_std': 0.02,
         'delib_rate_mean': 0.5, 'delib_rate_std': 0.0,
         'easy_acc_mean': 0.75, 'easy_acc_std': 0.03,
         'medium_acc_mean': 0.68, 'medium_acc_std': 0.04,
         'hard_acc_mean': 0.67, 'hard_acc_std': 0.04,
         'easy_delib_mean': 0.5, 'easy_delib_std': 0.0,
         'medium_delib_mean': 0.5, 'medium_delib_std': 0.0,
         'hard_delib_mean': 0.5, 'hard_delib_std': 0.0},
    ]

    plot_model_comparison(dummy)
    plot_difficulty_breakdown(dummy)
    plot_routing_by_difficulty(dummy[0])
    plot_pareto(dummy)
    print("Demo charts generated with placeholder data.")
    print("Run evaluate.py first to get real numbers, then re-run visualize.py.")