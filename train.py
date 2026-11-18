import numpy as np
from acp_model import ACP
from data import generate_dataset, fill_episodic_memory


def train():
    input_dim = 8
    hidden_dim = 12
    output_dim = 1
    memory_size = 15
    epochs = 100
    lr = 0.01
    alpha = 1.0   # routing loss weight
    lam = 0.1     # compute cost weight

    X, Y, Y_route = generate_dataset(n_samples=400, seed=123)

    model = ACP(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, memory_size=memory_size, seed=42)
    model = fill_episodic_memory(model)

    history = {'loss': [], 'acc': [], 'route_acc': [], 'delib_rate': []}

    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        epoch_loss = 0.0
        correct = 0
        route_correct = 0
        delib_count = 0

        for idx in perm:
            x = X[idx]
            y = Y[idx][0]
            y_route = Y_route[idx][0]

            grads, final_out, p = model.compute_gradients(
                x, y, y_route, alpha=alpha, lam=lam
            )

            # SGD update (in-place on numpy arrays)
            for key in grads:
                arr = getattr(model, key)
                arr -= lr * grads[key]

            # Hard-decision metrics
            out, deliberated = model.predict(x)
            epoch_loss += ((final_out - y) ** 2).item()
            if abs(out - y) < 0.5:
                correct += 1
            if deliberated == int(y_route):
                route_correct += 1
            if deliberated:
                delib_count += 1

        n = len(X)
        history['loss'].append(epoch_loss / n)
        history['acc'].append(correct / n)
        history['route_acc'].append(route_correct / n)
        history['delib_rate'].append(delib_count / n)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | "
                  f"Loss {history['loss'][-1]:.4f} | "
                  f"Acc {history['acc'][-1]:.3f} | "
                  f"RouteAcc {history['route_acc'][-1]:.3f} | "
                  f"DelibRate {history['delib_rate'][-1]:.3f}")

    return model, history


if __name__ == "__main__":
    model, history = train()