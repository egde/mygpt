"""Experiment 40 — Sample from the trained Shakespeare GPT."""

import torch

from mygpt import GPT, CharTokenizer, generate, set_seed


def main() -> None:
    tok = CharTokenizer.load("shakespeare_tokenizer.json")
    set_seed(0)
    model = GPT(
        vocab_size=tok.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout=0.0,
    )
    model.load_state_dict(torch.load("shakespeare_gpt.pt"))

    set_seed(0)
    prompt = tok.encode("ROMEO:").unsqueeze(0)
    out = generate(model, prompt, max_new_tokens=200, temperature=1.0, top_k=10)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
