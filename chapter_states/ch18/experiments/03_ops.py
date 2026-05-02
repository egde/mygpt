"""Experiment 03 — Matrix multiplication, broadcasting, and a seeded random tensor."""

import torch


def main() -> None:
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print("A @ B =")
    print(A @ B)
    print()

    # Broadcasting: a (3,) vector added to a (2, 3) matrix
    M = torch.zeros(2, 3)
    v = torch.tensor([1.0, 2.0, 3.0])
    print("M + v =")
    print(M + v)
    print()

    # Seeded random tensor — reproducible across runs and machines
    torch.manual_seed(0)
    R = torch.randn(2, 3)
    print("torch.randn(2, 3) with seed 0:")
    print(R)


if __name__ == "__main__":
    main()
