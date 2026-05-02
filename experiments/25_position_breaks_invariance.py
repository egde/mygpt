"""Experiment 25 — Position embeddings break self-attention's permutation invariance.

Without position embeddings, token 3 at position 3 has the same vector as
token 3 at position 0. With position embeddings added, they differ.
"""

import torch
import torch.nn as nn

from mygpt import TokenEmbedding, set_seed


def main() -> None:
    set_seed(0)
    V, C, max_seq = 4, 4, 8
    te = TokenEmbedding(V, C)
    pe = nn.Embedding(max_seq, C)

    # Two id sequences that share the token "3" at different positions
    ids1 = torch.tensor([0, 1, 2, 3])
    ids2 = torch.tensor([3, 2, 1, 0])

    # Without position embedding: token 3 always has the same vector
    print("Without position embedding (token 3 row):")
    print(f"  ids1 position 3: {te(ids1)[3]}")
    print(f"  ids2 position 0: {te(ids2)[0]}")
    print(f"  identical: {torch.equal(te(ids1)[3], te(ids2)[0])}")
    print()

    # With position embedding: same token at different positions differs
    def with_pos(ids):
        T = ids.shape[-1]
        positions = torch.arange(T)
        return te(ids) + pe(positions)

    v1 = with_pos(ids1)
    v2 = with_pos(ids2)
    print("With position embedding (token 3 at different positions):")
    print(f"  ids1 position 3 (token 3 at pos 3): {v1[3]}")
    print(f"  ids2 position 0 (token 3 at pos 0): {v2[0]}")
    print(f"  identical: {torch.equal(v1[3], v2[0])}")


if __name__ == "__main__":
    main()
