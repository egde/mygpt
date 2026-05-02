"""Experiment 44 — Compare BPE vs Char encoding length on the full corpus.

Skipping over the slow BPE encode (~20 s); the result is deterministic.
"""

import time

from mygpt import BPETokenizer, CharTokenizer


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    char_tok = CharTokenizer.from_text(text)
    bpe_tok = BPETokenizer.load("shakespeare.bpe.json")

    t0 = time.time()
    char_ids = char_tok.encode(text)
    t_char = time.time() - t0

    t0 = time.time()
    bpe_ids = bpe_tok.encode(text)
    t_bpe = time.time() - t0

    print(f"corpus chars:           {len(text):,}")
    print(f"CharTokenizer ids:      {len(char_ids):,}  ({t_char:.1f}s to encode)")
    print(f"BPETokenizer  ids:      {len(bpe_ids):,}  ({t_bpe:.1f}s to encode)")
    print(f"BPE compression ratio:  {len(char_ids) / len(bpe_ids):.2f}x")
    print(f"Char vocab_size:        {char_tok.vocab_size}")
    print(f"BPE  vocab_size:        {bpe_tok.vocab_size}")


if __name__ == "__main__":
    main()
