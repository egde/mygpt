"""Experiment 27 — Build a tiny corpus and sample one batch."""

import torch

from mygpt import VOCAB, get_batch, set_seed


def main() -> None:
    set_seed(0)

    # Build a corpus by repeating the running example 64 times
    data = torch.tensor([VOCAB.index(t) for t in (["I", "love", "AI", "!"] * 64)], dtype=torch.long)
    print(f"Corpus length: {len(data)} tokens")
    print(f"First 12 tokens: {data[:12].tolist()}")
    print()

    # One batch
    set_seed(42)
    x, y = get_batch(data, batch_size=4, seq_len=8)
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print()
    print("x (first row):", x[0].tolist())
    print("y (first row):", y[0].tolist())
    print("(y is x shifted left by one — the next-token target.)")


if __name__ == "__main__":
    main()
