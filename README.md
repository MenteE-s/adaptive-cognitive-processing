# adaptive-cognitive-processing

**Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems**

A NumPy implementation of an **Adaptive Cognitive Processing (ACP)** architecture. A fast linear pathway answers by default. A monitor checks episodic memory and triggers a deliberate MLP pathway when the input is novel.

## Experimental Setup (v1.1)

| Setting | Value |
|---------|-------|
| Framework | NumPy (CPU only, no GPU) |
| Samples | 5,000 synthetic regression examples |
| Difficulty levels | Easy / Medium / Hard |
| Input dimension | 16 |
| Fast pathway | Linear (cheap, cannot fit nonlinear targets) |
| Deliberate pathway | MLP, 48 hidden units over [x, fast output] |
| Episodic memory | 30 fixed keys (15 familiar, 15 random) |
| Monitor | Linear input + episodic novelty feature (1 - max cosine similarity) |
| Seeds | 10 (mean ± std reported) |
| Baselines | Always Fast (linear), Always Deliberate (MLP), Fixed MLP, Random Router |

## Tasks

| Level | Target | Distribution | Ground-truth route |
|-------|--------|--------------|--------------------|
| Easy | y = 2*x0 + 1 | Familiar cluster near dim 0 | Fast |
| Medium | y = x1^2 + 1 | Novel cluster near dim 1 | Deliberate |
| Hard | y = x2*x3 + x4*x5 | Uniform in [-1, 1]^16 | Deliberate |

The linear fast path can fit Easy but cannot fit Medium or Hard. This creates a genuine fast/deliberate gap.

## Key Result

Across 10 seeds, ACP reaches **93.5% task accuracy** and **98.7% routing accuracy** while deliberating on **67.2%** of inputs. Always Deliberate reaches 93.7% at 100% deliberation, and Always Fast reaches 37.0% at 0% deliberation. On Hard inputs, ACP reaches 86.3% versus 26.4% for Always Fast.

Removing the episodic novelty feature drops task accuracy to **73.7%** over 3 seeds. The x-only monitor still deliberates on 66.2% of inputs, but it deliberates on only 87.9% of Hard inputs, and those misroutes are costly. The memory readout is doing real work.

Deliberation rate is a routing metric, not a wall-clock compute measurement. The paper discusses the monitor's fixed cost and why a smaller memory would be needed to turn the routing policy into a net FLOP saving.

## Files

| File | Purpose |
|------|---------|
| `acp_model.py` | ACP: linear fast path, MLP deliberate path, memory-aware monitor |
| `data.py` | Dataset generator with difficulty labels + episodic memory filler |
| `baselines.py` | Linear Fast, Always Deliberate, Fixed MLP, Random Router |
| `train.py` | ACP training loop |
| `evaluate.py` | 10-seed experiment with per-difficulty metrics, saves JSON |
| `ablation_novelty.py` | Memory versus x-only routing ablation |
| `test_gradients.py` | Finite-difference gradient check for ACP |
| `charts/generate_charts.py` | Generates paper figures from evaluation JSON |
| `charts/make_results_table.py` | Generates LaTeX table snippets from evaluation JSON |
| `paper.ltx` | LaTeX preprint source |
| `requirements.txt` | `numpy`, `matplotlib` |

## Run

```bash
pip install -r requirements.txt

# Verify analytical gradients
python test_gradients.py

# Train ACP once (smoke test)
python train.py

# Full experiment: 10 seeds, all baselines
python evaluate.py

# Routing ablation: memory feature versus x only
python ablation_novelty.py

# Generate LaTeX tables and paper figures
python charts/make_results_table.py
python charts/generate_charts.py

# Compile paper
pdflatex paper.ltx
```

## Metrics

- Task accuracy (overall and per difficulty)
- Routing accuracy (deliberate flag vs ground-truth route)
- Deliberation rate (overall and per difficulty)
- Accuracy vs deliberation rate (routing tradeoff)

## Authors

- **Syed Syab Ahmad Shah** (syab@menteeai.org), ORCID: [0009-0003-9183-582X](https://orcid.org/0009-0003-9183-582X)
- **Team MenteE AI**, https://menteeai.org

## Citation

```
@misc{shah2026acp,
  title = {Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems},
  author = {Shah, Syed Syab Ahmad and Team MenteE AI},
  year = {2026},
  url = {https://github.com/MenteE-s/adaptive-cognitive-processing},
  note = {MenteE AI, https://menteeai.org}
}
```

## License

MIT. See [LICENSE](LICENSE). Open source for public research.
