"""Experiment 06 — From token ids (Chapter 3 mygpt addition) to one-hot vectors."""

import torch
import torch.nn.functional as F

from mygpt import VOCAB, to_ids


def main() -> None:
    ids = to_ids(["I", "love", "AI", "!"])
    print(f"ids:   {ids}")
    print(f"shape: {tuple(ids.shape)}")

    V = len(VOCAB)
    onehot = F.one_hot(ids, num_classes=V).float()
    print(f"\none-hot shape: {tuple(onehot.shape)}  # (T, V) = (4, {V})")
    print("one-hot rows (one row per token, V columns):")
    print(onehot)


if __name__ == "__main__":
    main()
