"""Experiment 19 — MultiHeadAttention(C, num_heads=1) reduces to
SingleHeadAttention(C, head_dim=C).

Builds both with the same seed and same input; confirms torch.equal()
on the two outputs.
"""

import torch

from mygpt import MultiHeadAttention, SingleHeadAttention, set_seed


def main() -> None:
    set_seed(0)
    sha = SingleHeadAttention(embed_dim=4, head_dim=4, max_seq_len=64, dropout=0.0)
    sha.eval()

    set_seed(0)
    mha = MultiHeadAttention(embed_dim=4, num_heads=1, max_seq_len=64, dropout=0.0)
    mha.eval()

    set_seed(42)
    x = torch.randn(1, 4, 4)

    with torch.no_grad():
        out_sha = sha(x)
        out_mha = mha(x)

    print("SingleHeadAttention(4, 4):")
    print(out_sha)
    print()
    print("MultiHeadAttention(4, num_heads=1):")
    print(out_mha)
    print()
    print(f"identical:    {torch.equal(out_sha, out_mha)}")
    print(f"max abs diff: {(out_sha - out_mha).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
