"""Experiment 43 — Encode/decode a sample with the trained BPE tokenizer."""

from mygpt import BPETokenizer


def main() -> None:
    tok = BPETokenizer.load("shakespeare.bpe.json")

    sample = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    ids = tok.encode(sample)
    back = tok.decode(ids)

    print(f"sample chars:  {len(sample)}")
    print(f"encoded ids:   {len(ids)}")
    print(f"compression:   {len(sample) / len(ids):.2f}x")
    print(f"round-trip ok: {back == sample}")
    print()
    print("First 20 ids decoded as their token strings:")
    for i, tok_id in enumerate(ids[:20].tolist()):
        print(f"  {i:>2}: id={tok_id:>3}  symbol={tok.vocab[tok_id]!r}")


if __name__ == "__main__":
    main()
