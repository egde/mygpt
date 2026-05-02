"""Experiment 28 — Train a tiny GPT on a repeating-cycle corpus.

200 AdamW steps with B=4, T=8. Loss drops from ~5.27 (random init,
confidently wrong) to ~0.0035 (essentially memorised the cycle).
"""

import math

import torch

from mygpt import GPT, get_batch, set_seed


def main() -> None:
    set_seed(0)
    model = GPT(
        vocab_size=4,
        embed_dim=8,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.tensor([0, 1, 2, 3] * 64, dtype=torch.long)

    set_seed(42)
    B, T = 4, 8
    print(f"Training on a length-{len(data)} corpus, batch_size={B}, seq_len={T}")
    print(f"Reference: log(V) = log(4) = {math.log(4):.4f}\n")

    for step in range(1, 201):
        x, y = get_batch(data, B, T)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step in (1, 10, 50, 100, 200):
            print(f"step {step:>3}: loss = {loss.item():.4f}")

    # Save the trained model so Chapter 15 can load it
    torch.save(model.state_dict(), "trained_gpt.pt")
    print("\nSaved trained model to trained_gpt.pt")


if __name__ == "__main__":
    main()
