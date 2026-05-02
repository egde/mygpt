"""Experiment 42 — Train a BPE tokenizer on Tiny Shakespeare with 512 merges.

Saves the trained tokenizer to shakespeare.bpe.json (small JSON file, < 5 KB).
~70 s on M1 MPS; reloading is instant.
"""

import time

from mygpt import BPETokenizer


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    t0 = time.time()
    tok = BPETokenizer.from_text(text, num_merges=512)
    elapsed = time.time() - t0

    print(f"training took:    {elapsed:.1f}s")
    print(f"vocab_size:       {tok.vocab_size}")
    print(f"chars (alphabet): {len(tok.chars)}")
    print(f"merges:           {len(tok.merges)}")
    print()
    print(f"first 10 alphabet chars: {tok.chars[:10]}")
    print(f"first 10 merges:")
    for i, (a, b) in enumerate(tok.merges[:10]):
        print(f"  {i + 1}. ({a!r}, {b!r}) -> {(a + b)!r}")

    tok.save("shakespeare.bpe.json")
    print()
    print("saved to shakespeare.bpe.json")


if __name__ == "__main__":
    main()
