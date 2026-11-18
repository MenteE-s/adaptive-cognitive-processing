import numpy as np
from acp_model import ACP
from data import generate_dataset, fill_episodic_memory


def evaluate():
    input_dim = 8
    hidden_dim = 12
    output_dim = 1
    memory_size = 15

    X, Y, Y_route = generate_dataset(n_samples=200, seed=999)  # test set
    model = ACP(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, memory_size=memory_size, seed=42)
    model = fill_episodic_memory(model)

    correct = 0
    easy_correct = 0
    hard_correct = 0
    route_correct = 0
    delib_count = 0
    easy_count = 0
    hard_count = 0
    easy_delib = 0
    hard_delib = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i][0]
        y_r = Y_route[i][0]
        out, deliberated = model.predict(x)

        if y_r == 1:
            hard_count += 1
            if abs(out - y) < 0.5:
                hard_correct += 1
            if deliberated:
                hard_delib += 1
        else:
            easy_count += 1
            if abs(out - y) < 0.5:
                easy_correct += 1
            if deliberated:
                easy_delib += 1

        if abs(out - y) < 0.5:
            correct += 1
        if deliberated == int(y_r):
            route_correct += 1
        if deliberated:
            delib_count += 1

    n = len(X)
    print("\n=== Evaluation Results ===")
    print(f"Overall Accuracy : {correct / n:.3f}")
    print(f"Easy Accuracy    : {easy_correct / max(easy_count, 1):.3f} (n={easy_count})")
    print(f"Hard Accuracy    : {hard_correct / max(hard_count, 1):.3f} (n={hard_count})")
    print(f"Routing Accuracy : {route_correct / n:.3f}")
    print(f"Deliberation Rate: {delib_count / n:.3f}")
    print(f"Easy Delib Rate  : {easy_delib / max(easy_count, 1):.3f}  (should be near 0)")
    print(f"Hard Delib Rate  : {hard_delib / max(hard_count, 1):.3f}  (should be near 1)")


if __name__ == "__main__":
    evaluate()