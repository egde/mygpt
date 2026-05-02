"""Experiment 34 — Build a character alphabet from a string."""


def main() -> None:
    text = "I love AI !"
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    print(f"text:        {text!r}")
    print(f"vocab_size:  {len(chars)}")
    print(f"chars:       {chars}")
    print(f"stoi:        {stoi}")
    print(f"itos:        {itos}")


if __name__ == "__main__":
    main()
