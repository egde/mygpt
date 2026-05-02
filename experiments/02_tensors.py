"""Experiment 02 — Tensors: shape, dtype, device, indexing, reshape."""

import torch


def main() -> None:
    # A 1-D tensor (a vector)
    v = torch.tensor([0.0, 1.0, 2.0, 3.0])
    print(repr(v))
    print(f"shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    print()

    # A 2-D tensor (a matrix)
    M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(repr(M))
    print(f"shape={tuple(M.shape)}")
    print()

    # arange and reshape — useful for setting up examples
    x = torch.arange(12)
    print("arange(12) =", x)
    print("reshape(3, 4) =")
    print(x.reshape(3, 4))
    print("reshape(2, 2, 3).shape =", tuple(x.reshape(2, 2, 3).shape))


if __name__ == "__main__":
    main()
