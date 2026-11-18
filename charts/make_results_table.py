"""
Generate LaTeX table snippets from charts/evaluation_results.json.
Writes:
  charts/results_table.tex       (main comparison table)
  charts/difficulty_table.tex    (per-difficulty table)

Deliberation rate is a routing metric, not a wall-clock compute measurement.
The monitor's fixed cost is discussed in the paper.
"""

import json
import os

ORDER = ['ACP', 'AlwaysFast', 'AlwaysDeliberate', 'FixedMLP', 'RandomRouter']
LABELS = {
    'ACP': 'ACP',
    'AlwaysFast': 'Always Fast',
    'AlwaysDeliberate': 'Always Deliberate',
    'FixedMLP': 'Fixed MLP',
    'RandomRouter': 'Random Router',
}


def load(path='charts/evaluation_results.json'):
    with open(path, 'r') as f:
        rows = json.load(f)
    return {r['name']: r for r in rows}


def main_table(data):
    lines = []
    for name in ORDER:
        r = data[name]
        lines.append(
            f"{LABELS[name]} & "
            f"{r['accuracy_mean']:.3f}$\\pm${r['accuracy_std']:.3f} & "
            f"{r['route_acc_mean']:.3f}$\\pm${r['route_acc_std']:.3f} & "
            f"{r['delib_rate_mean']:.3f}$\\pm${r['delib_rate_std']:.3f} \\\\"
        )
    with open('charts/results_table.tex', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote charts/results_table.tex')


def difficulty_table(data):
    lines = []
    for name in ORDER:
        r = data[name]
        lines.append(
            f"{LABELS[name]} & "
            f"{r['easy_acc_mean']:.3f} & "
            f"{r['medium_acc_mean']:.3f} & "
            f"{r['hard_acc_mean']:.3f} & "
            f"{r['easy_delib_mean']:.3f} & "
            f"{r['medium_delib_mean']:.3f} & "
            f"{r['hard_delib_mean']:.3f} \\\\"
        )
    with open('charts/difficulty_table.tex', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote charts/difficulty_table.tex')


def ablation_table(path='charts/ablation_novelty.json'):
    if not os.path.exists(path):
        print(f"No {path}; skipping ablation table.")
        return
    with open(path, 'r') as f:
        summary = json.load(f)['summary']
    labels = {'with_novelty': 'ACP (x + novelty)', 'x_only': 'ACP (x only)'}
    lines = []
    for key in ['with_novelty', 'x_only']:
        r = summary[key]
        lines.append(
            f"{labels[key]} & "
            f"{r['accuracy_mean']:.3f}$\\pm${r['accuracy_std']:.3f} & "
            f"{r['route_acc_mean']:.3f}$\\pm${r['route_acc_std']:.3f} & "
            f"{r['delib_rate_mean']:.3f}$\\pm${r['delib_rate_std']:.3f} \\\\"
        )
    with open('charts/ablation_table.tex', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote charts/ablation_table.tex')


if __name__ == '__main__':
    data = load()
    main_table(data)
    difficulty_table(data)
    ablation_table()
