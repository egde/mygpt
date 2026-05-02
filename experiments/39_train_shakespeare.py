"""Experiment 39 — Train a 207k-parameter GPT on Tiny Shakespeare.

2,000 AdamW steps with B=16, T=64. Loss drops from ~41 (random init,
confidently wrong) to ~2.08 (a meaningful improvement over log(65) ≈ 4.17,
the random-baseline cross-entropy).
"""

import math
import time

import torch

from mygpt import GPT, CharTokenizer, get_batch, set_seed


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)

    set_seed(0)
    model = GPT(
        vocab_size=tok.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, T = 16, 64
    print(f"params:         {n_params:,}")
    print(f"vocab_size:     {tok.vocab_size}")
    print(f"corpus chars:   {len(data):,}")
    print(f"batch_size B:   {B}")
    print(f"seq_len T:      {T}")
    print(f"reference log(V) = {math.log(tok.vocab_size):.4f}\n")

    set_seed(42)
    t0 = time.time()
    for step in range(1, 2001):
        x, y = get_batch(data, B, T)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step in (1, 10, 100, 500, 1000, 2000):
            print(f"step {step:>4}: loss = {loss.item():.4f}  ({time.time()-t0:.1f}s)")

    torch.save(model.state_dict(), "shakespeare_gpt.pt")
    tok.save("shakespeare_tokenizer.json")
    print("\nSaved shakespeare_gpt.pt and shakespeare_tokenizer.json")


if __name__ == "__main__":
    main()
