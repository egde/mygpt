"""Experiment 38 — Tokenize Tiny Shakespeare."""

from mygpt import CharTokenizer


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)

    print(f"corpus chars:   {len(text):,}")
    print(f"vocab_size:     {tok.vocab_size}")
    print(f"chars:          {tok.chars}")
    print()
    print(f"data shape:     {tuple(data.shape)}")
    print(f"data.dtype:     {data.dtype}")
    print(f"first 60 ids:   {data[:60].tolist()}")


if __name__ == "__main__":
    main()
