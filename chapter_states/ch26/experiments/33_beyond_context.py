"""Experiment 33 — Generation degrades beyond the trained T=8 context.

prompt='I', generate 12 tokens with top_k=1. The first 8 tokens (positions
0..7) are the perfect cycle; the rest (positions 8..) drift, because
the position embeddings at those positions were never trained.
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
    out = generate(model, prompt, max_new_tokens=12, top_k=1)

    tokens = [VOCAB[i] for i in out[0].tolist()]
    cycle = ["I", "love", "AI", "!"] * 4   # what we'd expect if the cycle held
    print(f"output:   {tokens}")
    print(f"expected: {cycle[:len(tokens)]}")
    print()
    for t, (got, want) in enumerate(zip(tokens, cycle)):
        ok = "OK" if got == want else "DRIFT"
        print(f"  position {t}: got={got!r:<7} expected={want!r:<7} {ok}")


if __name__ == "__main__":
    main()
