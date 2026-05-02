"""Experiment 08 — The same minimisation as exp 07, using torch.optim.SGD.

Verifies that the four-line training loop produces identical x values
to the manual version in exp 07.
"""

import torch


def main() -> None:
    x = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.SGD([x], lr=0.1)

    print(f"step | x        | f(x)")
    print(f"-----+----------+---------")
    print(f"   0 | {x.item():.6f} |")

    for step in range(1, 11):
        opt.zero_grad()
        loss = (x - 3) ** 2
        loss.backward()
        opt.step()
        print(f"  {step:2d} | {x.item():.6f} | {loss.item():.6f}")

    for step in range(11, 101):
        opt.zero_grad()
        loss = (x - 3) ** 2
        loss.backward()
        opt.step()

    print(f"\nfinal x after 100 steps: {x.item():.6f}")


if __name__ == "__main__":
    main()
