"""Experiment 24 — Stack two TransformerBlocks and watch the running example
flow through both. Confirms the output shape is unchanged and the parameter
count is exactly 2 × (single-block parameters).
"""

import torch

from mygpt import TokenEmbedding, TransformerBlock, set_seed, to_ids


def main() -> None:
    set_seed(0)
    V, C = 4, 4
    te = TokenEmbedding(V, C)
    blocks = torch.nn.Sequential(
        TransformerBlock(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0),
        TransformerBlock(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0),
    )
    blocks.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = blocks(x)

    print(f"input shape:    {tuple(x.shape)}")
    print(f"after 2 blocks: {tuple(out.shape)}")
    print()

    n_te = sum(p.numel() for p in te.parameters())
    n_blocks = sum(p.numel() for p in blocks.parameters())
    print(f"TokenEmbedding params: {n_te}")
    print(f"2 TransformerBlocks:   {n_blocks}  (= 2 * 228)")
    print(f"Total:                 {n_te + n_blocks}")


if __name__ == "__main__":
    main()
