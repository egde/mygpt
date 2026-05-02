"""Experiment 12 — End-to-end: text → ids → dense vectors via mygpt.TokenEmbedding."""

import torch

from mygpt import VOCAB, TokenEmbedding, set_seed, to_ids


def main() -> None:
    set_seed(0)

    V, C = len(VOCAB), 4
    te = TokenEmbedding(vocab_size=V, embed_dim=C)
    print(f"TokenEmbedding(V={V}, C={C}) — {sum(p.numel() for p in te.parameters())} params\n")

    # 1-D path: a single sentence
    ids = to_ids(["I", "love", "AI", "!"])
    out = te(ids)
    print(f"single sentence:")
    print(f"  ids       shape={tuple(ids.shape)}    {ids.tolist()}")
    print(f"  embedded  shape={tuple(out.shape)}  (T, C)")
    print(out)
    print()

    # 2-D path: a batch of sentences
    batch = torch.stack([
        to_ids(["I", "love", "AI", "!"]),
        to_ids(["AI", "love", "I", "!"]),
    ])
    out_batch = te(batch)
    print(f"batched:")
    print(f"  ids       shape={tuple(batch.shape)}    (B, T)")
    print(f"  embedded  shape={tuple(out_batch.shape)}  (B, T, C)")


if __name__ == "__main__":
    main()
