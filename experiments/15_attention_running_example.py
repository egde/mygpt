"""Experiment 15 — End-to-end self-attention on the running example.

Embeds 'I love AI !', runs SingleHeadAttention, and prints the
intermediate attention weights so you can see the lower-triangular
structure imposed by the causal mask.
"""

import math

import torch
import torch.nn.functional as F

from mygpt import VOCAB, SingleHeadAttention, TokenEmbedding, set_seed, to_ids


def main() -> None:
    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(vocab_size=V, embed_dim=C)
    attn = SingleHeadAttention(embed_dim=C, head_dim=C)

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)

    # Re-run attention manually so we can pull out the weights matrix
    Q = attn.W_Q(x)
    K = attn.W_K(x)
    V_proj = attn.W_V(x)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(attn.head_dim)
    mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
    scores = scores + mask
    weights = F.softmax(scores, dim=-1)

    print("Attention weights (B=1, T=4, T=4) — first batch element:")
    torch.set_printoptions(precision=4)
    print(weights[0].detach())
    print()
    print(f"row sums: {weights[0].sum(dim=-1).detach()}")
    print()

    # Standard module output
    out = attn(x)
    print(f"Final attention output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
