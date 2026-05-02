"""Experiment 30 — Greedy generation from the trained GPT.

argmax(logits) at every step, no randomness, prompt='I', 7 new tokens.
"""

import torch

from mygpt import GPT, VOCAB, set_seed


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
    model.load_state_dict(torch.load("trained_gpt.pt"))
    model.eval()

    ids = torch.tensor([[VOCAB.index("I")]])  # (1, 1)
    print(f"prompt: {[VOCAB[i] for i in ids[0].tolist()]}")

    with torch.no_grad():
        for _ in range(7):
            logits, _ = model(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)

    print(f"output: {[VOCAB[i] for i in ids[0].tolist()]}")


if __name__ == "__main__":
    main()
