"""Experiment 36 — Save a tokenizer to disk and reload it.

The reloaded tokenizer encodes to the same ids as the original.
"""

from mygpt import CharTokenizer


def main() -> None:
    text = "I love AI !"
    tok1 = CharTokenizer.from_text(text)
    tok1.save("tokenizer.json")
    print(f"saved tokenizer.json (vocab_size={tok1.vocab_size})")

    tok2 = CharTokenizer.load("tokenizer.json")
    print(f"loaded tokenizer.json (vocab_size={tok2.vocab_size})")

    sample = "love AI"
    ids1 = tok1.encode(sample)
    ids2 = tok2.encode(sample)
    print(f"sample:        {sample!r}")
    print(f"original ids:  {ids1.tolist()}")
    print(f"reloaded ids:  {ids2.tolist()}")
    print(f"identical:     {ids1.tolist() == ids2.tolist()}")


if __name__ == "__main__":
    main()
