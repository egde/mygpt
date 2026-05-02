"""Experiment 04 — Autograd on a quadratic, checked against the analytical derivative."""

import torch


def main() -> None:
    x = torch.tensor(3.0, requires_grad=True)
    y = x * x + 2 * x + 1   # f(x) = x^2 + 2x + 1
    y.backward()
    x_val = x.item()
    print(f"f({x_val:g})  = {y.item()}")
    print(f"f'({x_val:g}) = {x.grad.item()}  (analytical: 2*({x_val:g}) + 2 = {2*x_val + 2:g})")


if __name__ == "__main__":
    main()
