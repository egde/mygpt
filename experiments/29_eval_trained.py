"""Experiment 29 — Verify the trained GPT predicts the cycle correctly.

For every prefix of the cycle, computes argmax(logits[-1]) and compares
to the true next token.
"""

import torch

from mygpt import GPT, VOCAB, set_seed


def main() -> None:
    # Re-build a fresh model with the same architecture and load the trained
    # weights from experiment 28.
    set_seed(0)  # only affects model init, which load_state_dict overwrites
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

    print("Trained model's argmax next-token prediction:\n")
    for L in range(1, 9):
        # Build a length-L prefix of the cycle starting from token 0
        prefix = torch.tensor([(i % 4) for i in range(L)]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(prefix)
        pred = logits[0, -1, :].argmax().item()
        actual = (prefix[0, -1].item() + 1) % 4
        ok = "OK" if pred == actual else "WRONG"
        print(f"  prefix={prefix[0].tolist()!s:<28}  predicted={VOCAB[pred]!r:<7}  expected={VOCAB[actual]!r:<7}  {ok}")


if __name__ == "__main__":
    main()
