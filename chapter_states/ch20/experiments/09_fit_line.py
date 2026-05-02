"""Experiment 09 — Fit a linear model y = w x + b to noisy data.

Generates 20 points from y = 2x + 0.5 + noise, then trains w and b to
minimise mean squared error using torch.optim.SGD. Reports the loss at
selected steps and the learned (w, b) at the end.
"""

import torch


def main() -> None:
    torch.manual_seed(0)

    # Synthetic data: 20 points in [-1, 1], with true y = 2x + 0.5 + N(0, 0.1)
    n = 20
    x_data = torch.linspace(-1.0, 1.0, n)
    true_w, true_b = 2.0, 0.5
    y_data = true_w * x_data + true_b + 0.1 * torch.randn(n)

    # Parameters to learn — start at zero
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=0.1)

    print("step | w        | b        | loss")
    print("-----+----------+----------+---------")
    for step in range(1, 51):
        opt.zero_grad()
        y_pred = w * x_data + b                 # forward
        loss = ((y_pred - y_data) ** 2).mean()  # MSE
        loss.backward()                         # populate w.grad, b.grad
        opt.step()                              # w ← w − lr·grad, same for b
        if step in (1, 5, 10, 20, 50):
            print(f"  {step:2d} | {w.item():.6f} | {b.item():.6f} | {loss.item():.6f}")

    print(f"\nlearned: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"true:    w = {true_w:.4f}, b = {true_b:.4f}")


if __name__ == "__main__":
    main()
