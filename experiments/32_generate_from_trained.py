"""Experiment 32 — Use mygpt.generate on the trained model.

prompt='I', generate 7 new tokens with top_k=1 (greedy). Stays within
the trained T=8 context window — see experiment 33 for what happens
beyond it.
"""

import torch

from mygpt import GPT, VOCAB, generate, set_seed, to_ids


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

    prompt = to_ids(["I"]).unsqueeze(0)
    out = generate(model, prompt, max_new_tokens=7, top_k=1)

    print(f"prompt: {[VOCAB[i] for i in prompt[0].tolist()]}")
    print(f"output: {[VOCAB[i] for i in out[0].tolist()]}")


if __name__ == "__main__":
    main()
