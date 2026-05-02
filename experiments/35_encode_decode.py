"""Experiment 35 — Build a CharTokenizer and round-trip encode/decode."""

from mygpt import CharTokenizer


def main() -> None:
    text = "I love AI ! I love AI !"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    back = tok.decode(ids)

    print(f"text:       {text!r}")
    print(f"vocab_size: {tok.vocab_size}")
    print(f"ids shape:  {tuple(ids.shape)}")
    print(f"first 12 ids: {ids[:12].tolist()}")
    print(f"decoded:    {back!r}")
    print(f"round-trip ok: {back == text}")


if __name__ == "__main__":
    main()
