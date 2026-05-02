"""Experiment 37 — Build the training corpus tensor via CharTokenizer.

This is the same shape of input the Ch.14 training loop expects — just
with vocab_size=8 (characters) instead of 4 (whole words).
"""

import torch

from mygpt import CharTokenizer


def main() -> None:
    text = "I love AI ! " * 16   # 192 characters total
    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)

    print(f"text length (chars): {len(text)}")
    print(f"vocab_size:          {tok.vocab_size}")
    print(f"data shape:          {tuple(data.shape)}")
    print(f"data.dtype:          {data.dtype}")
    print(f"first 24 ids:        {data[:24].tolist()}")
    print(f"decoded[:24]:        {tok.decode(data[:24])!r}")


if __name__ == "__main__":
    main()
