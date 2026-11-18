# adaptive-cognitive-processing

**Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems**

A NumPy implementation of an **Adaptive Cognitive Processing (ACP)** architecture — a fast pathway (System 1) answers by default, while a monitor checks episodic memory and triggers a deliberate pathway (System 2) only when the input is novel or uncertain.

## Experimental Scope (v1.0)

| Setting | Value |
|---------|-------|
| Framework | NumPy (CPU only, no GPU) |
| Samples | 5,000 synthetic regression examples |
| Difficulty levels | Easy / Medium / Hard |
| Input dimension | 32 |
| Hidden dimension | 48 |
| Episodic memory | 40 hand-filled episodes |
| Seeds | 10 (mean ± std reported) |
| Baselines | Always Fast, Always Deliberate, Fixed MLP, Random Router |

## Files

| File | Purpose |
|------|---------|
| `acp_model.py` | Core ACP architecture: fast path, monitor, deliberate path, episodic memory |
| `data.py` | Synthetic dataset generator (3 difficulties) + memory filler |
| `baselines.py` | Always Fast, Always Deliberate, Fixed MLP, Random Router |
| `train.py` | Training loop for ACP and baselines |
| `evaluate.py` | Full experiment: 5 seeds, aggregated metrics, comparison table |
| `visualize.py` | Charts for preprint + social media |
| `requirements.txt` | `numpy`, `matplotlib` |
| `README.md` | This file |
| `.gitignore` | Excludes caches and generated PNGs |

## Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train ACP once (quick smoke test)
python train.py

# Run full experiment (5 seeds, all baselines, aggregated results)
python evaluate.py

# Generate charts for the paper / X / LinkedIn
python visualize.py
```

## Metrics reported

- **Task Accuracy** (overall and per difficulty: easy / medium / hard)
- **Routing Accuracy** (did it choose fast for easy, deliberate for hard?)
- **Deliberation Rate** (% of inputs that triggered System 2)
- **Compute Cost proxy** (fast = 0, deliberate = 1)
- **Pareto frontier**: accuracy vs. compute cost across models

## Expected behavior

- **Easy inputs** → low deliberation rate, high accuracy via fast path
- **Hard inputs** → high deliberation rate, high accuracy via deliberate path
- **Overall compute stays low** because the model does **not** deliberate on everything
- ACP should dominate the Pareto frontier: better accuracy per unit of compute than baselines

## Citation

If you build on this idea, please reference **menteeai.org**.

Suggested citation:

```
@misc{mentee_acp_2026,
  title = {Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems},
  author = {MenteE AI},
  year = {2026},
  url = {https://github.com/MenteE-s/adaptive-cognitive-processing}
}
```

## License

MIT — open source, public research. Take the idea with reference.