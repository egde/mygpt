"""Experiment 10 — One-hot @ W and W[id] are the same thing.

Demonstrates that multiplying a one-hot vector by an embedding matrix
gives the same result as indexing the matrix by the id directly. The
lookup is the cheap implementation; the one-hot is the conceptual
explanation.
"""

import torch
import torch.nn.functional as F

from mygpt import set_seed


def main() -> None:
    set_seed(0)
    V, C = 4, 4
    W = torch.randn(V, C)
    print("Embedding matrix W (V=4, C=4):")
    print(W)
    print()

    target_id = 1
    onehot = F.one_hot(torch.tensor(target_id), num_classes=V).float()
    print(f"one-hot for id={target_id}: {onehot}")

    out_onehot = onehot @ W
    out_index = W[target_id]
    print(f"\none-hot @ W   = {out_onehot}")
    print(f"W[{target_id}]            = {out_index}")
    print(f"identical:    {torch.equal(out_onehot, out_index)}")


if __name__ == "__main__":
    main()
