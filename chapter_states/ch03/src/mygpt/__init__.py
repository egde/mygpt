"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

This file holds the package-level constants and small helpers used in every
chapter. After Chapter 3, it knows about the running-example vocabulary and
how to turn a list of tokens into a PyTorch tensor of ids.
"""

import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids.

    Each entry of `tokens` must appear in `VOCAB`; otherwise `VOCAB.index`
    raises a `ValueError` and the conversion fails.
    """
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
    sample = list(VOCAB)
    ids = to_ids(sample)
    print(f"to_ids({sample}) = {ids}")
