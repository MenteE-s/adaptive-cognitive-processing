import matplotlib.pyplot as plt
import numpy as np
import os


def plot_history(history, save_path='results.png'):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(history['loss'], color='black', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['acc'], label='Task Accuracy', color='green', linewidth=2)
    axes[0, 1].plot(history['route_acc'], label='Routing Accuracy', color='blue', linewidth=2)
    axes[0, 1].set_title('Accuracy vs Epoch', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['delib_rate'], color='orange', linewidth=2)
    axes[1, 0].set_title('Deliberation Rate vs Epoch', fontsize=12)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Fraction of inputs deliberated')
    axes[1, 0].grid(True, alpha=0.3)

    # Path allocation bar
    axes[1, 1].bar(['Fast Path', 'Deliberate Path'], [50, 50],
                   color=['gray', 'red'], alpha=0.7)
    axes[1, 1].set_title('Path Allocation (illustrative)', fontsize=12)
    axes[1, 1].set_ylabel('%')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


def plot_routing_by_difficulty(easy_delib_rate, hard_delib_rate,
                               save_path='routing_by_difficulty.png'):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Easy (should not think)', 'Hard (should think)'],
                  [easy_delib_rate, hard_delib_rate],
                  color=['lightcoral', 'steelblue'])
    ax.set_ylabel('Deliberation Rate')
    ax.set_title('Routing Decision by Task Difficulty')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


if __name__ == "__main__":
    dummy = {'loss': [1.0, 0.8, 0.6], 'acc': [0.5, 0.7, 0.85],
             'route_acc': [0.5, 0.75, 0.9], 'delib_rate': [0.3, 0.5, 0.55]}
    plot_history(dummy)
    plot_routing_by_difficulty(0.1, 0.85)