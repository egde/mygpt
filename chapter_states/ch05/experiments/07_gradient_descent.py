"""Experiment 07 — Gradient descent on f(x) = (x - 3)^2 by hand.

Starts from x = 0, learning rate eta = 0.1. Prints the trajectory of x
for the first 10 steps, then runs to step 100 and prints the final x.
"""

import torch


def main() -> None:
    x = torch.tensor(0.0, requires_grad=True)
    eta = 0.1

    print(f"step | x        | f(x)     | grad")
    print(f"-----+----------+----------+--------")
    print(f"   0 | {x.item():.6f} |          |")

    for step in range(1, 11):
        if x.grad is not None:
            x.grad.zero_()                       # reset gradient
        loss = (x - 3) ** 2                      # forward
        loss.backward()                          # compute grad
        g = x.grad.item()                        # save for printing
        with torch.no_grad():                    # update without tracking
            x.sub_(eta * x.grad)                 # x ← x − η·grad
        print(f"  {step:2d} | {x.item():.6f} | {loss.item():.6f} | {g:.4f}")

    # Continue silently to step 100
    for step in range(11, 101):
        if x.grad is not None:
            x.grad.zero_()
        loss = (x - 3) ** 2
        loss.backward()
        with torch.no_grad():
            x.sub_(eta * x.grad)

    print(f"\nfinal x after 100 steps: {x.item():.6f}")
    print(f"target:                   3.000000")


if __name__ == "__main__":
    main()
