# Charts

Figures for the ACP preprint:

**Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems**

Authors: Syed Syab Ahmad Shah and Team MenteE AI  
Contact: syab@menteeai.org  
ORCID: [0009-0003-9183-582X](https://orcid.org/0009-0003-9183-582X)  
Organization: [MenteE AI](https://menteeai.org)

## Files

| File | Description |
|------|-------------|
| `generate_charts.py` | Reads the JSON results and writes the PNG figures |
| `make_results_table.py` | Reads `evaluation_results.json` and writes the LaTeX table snippets |
| `model_comparison.png` | Accuracy and deliberation rate by model |
| `difficulty_breakdown.png` | Accuracy per difficulty level (Easy / Medium / Hard) |
| `routing_by_difficulty.png` | ACP deliberation rate by task difficulty |
| `pareto.png` | Accuracy vs deliberation rate |
| `ablation.png` | Memory ablation: accuracy and deliberation allocation |
| `hero_summary.png` | One-glance summary of the four key results (for posts) |
| `results_table.tex` | Main comparison table (generated) |
| `difficulty_table.tex` | Per-difficulty table (generated) |
| `ablation_table.tex` | Memory versus x-only ablation table (generated) |
| `evaluation_results.json` | Aggregated metrics produced by `evaluate.py` |
| `ablation_novelty.json` | Ablation metrics produced by `ablation_novelty.py` |

## Generate

From the repository root:

```bash
python evaluate.py
python charts/make_results_table.py
python charts/generate_charts.py
```

## LaTeX

`paper.ltx` references these files from this folder:

- `charts/routing_by_difficulty.png`
- `charts/difficulty_breakdown.png`
- `charts/pareto.png`
- `charts/ablation.png`
- `charts/results_table.tex`
- `charts/difficulty_table.tex`
- `charts/ablation_table.tex`

`charts/hero_summary.png` is a one-glance summary for social posts; the paper does not embed it.

Keep this folder next to `paper.ltx` when compiling on Overleaf or locally with `pdflatex`. If your online IDE requires a `.tex` extension, rename `paper.ltx` to `paper.tex`; the content is standard LaTeX.

## Citation

```
@misc{shah2026acp,
  title = {Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems},
  author = {Shah, Syed Syab Ahmad and Team MenteE AI},
  year = {2026},
  doi = {10.5281/zenodo.24353178},
  url = {https://github.com/MenteE-s/adaptive-cognitive-processing},
  note = {MenteE AI, https://menteeai.org}
}
```
