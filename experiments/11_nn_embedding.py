"""Experiment 11 — nn.Embedding as a packaged lookup.

Creates a small embedding, looks up token ids, and confirms the output
is exactly the corresponding rows of the embedding's weight matrix.
"""

import torch
import torch.nn as nn

from mygpt import VOCAB, set_seed, to_ids


def main() -> None:
    set_seed(0)
    V, C = len(VOCAB), 4
    emb = nn.Embedding(V, C)

    print(f"nn.Embedding({V}, {C})")
    print(f"  weight.shape         = {tuple(emb.weight.shape)}")
    print(f"  weight.requires_grad = {emb.weight.requires_grad}")
    print(f"  total parameters     = {sum(p.numel() for p in emb.parameters())}")
    print()

    print("emb.weight (V x C):")
    print(emb.weight)
    print()

    # 1-D ids → 2-D embedded vectors
    ids = to_ids(["I", "love", "AI", "!"])
    out = emb(ids)
    print(f"emb(ids) where ids={ids.tolist()}:")
    print(out)
    print(f"\nshape: {tuple(ids.shape)} -> {tuple(out.shape)}")

    # Confirm: emb(ids) == emb.weight[ids]
    print(f"matches emb.weight[ids]: {torch.equal(out, emb.weight[ids])}")


if __name__ == "__main__":
    main()
