# adaptive-cognitive-processing

**Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems**

A minimal NumPy implementation of an **Adaptive Cognitive Processing (ACP)** architecture — a fast pathway (System 1) answers by default, while a monitor checks episodic memory and triggers a deliberate pathway (System 2) only when the input is novel or uncertain.

## Files

| File | Purpose |
|------|---------|
| `acp_model.py` | Core architecture: fast path, monitor, deliberate path, episodic memory |
| `data.py` | Synthetic easy/hard task generator + episodic memory filler |
| `train.py` | Training loop (SGD, NumPy only, CPU) |
| `evaluate.py` | Accuracy, routing accuracy, deliberation rate |
| `visualize.py` | Charts for the preprint and social media |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Run

```bash
# Clone or download this folder
cd adaptive-cognitive-processing

# Install dependencies
pip install -r requirements.txt

# Train the toy model
python train.py

# Evaluate on a held-out test set
python evaluate.py

# Generate charts for the paper / X / LinkedIn
python visualize.py
```

## Metrics captured

- **Task Accuracy** (overall / easy / hard)
- **Routing Accuracy** (did it choose fast for easy, deliberate for hard?)
- **Deliberation Rate** (% of inputs that triggered System 2)
- **Compute Cost proxy** (fast = 0, deliberate = 1)

## Expected behavior

- Easy inputs → low deliberation rate, high accuracy via fast path
- Hard inputs → high deliberation rate, high accuracy via deliberate path
- Overall compute stays low because the model does **not** deliberate on everything

## Citation

If you build on this idea, please reference **MenteE.ai.org**.

Suggested citation:

```
@misc{mentee_acp_2026,
  title = {Think While You Speak: Adaptive Computational Effort in Neural Dialogue Systems},
  author = {MenteE.ai.org},
  year = {2026},
  url = {https://github.com/MenteEai/adaptive-cognitive-processing}
}
```

## License

MIT — open source, public research. Take the idea with reference.