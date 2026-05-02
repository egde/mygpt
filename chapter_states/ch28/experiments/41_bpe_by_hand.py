"""Experiment 41 — Byte-pair encoding traced by hand on Sennrich's example.

The algorithm has five steps:
  - Initialise: words → (chars + '</w>') tuples, with their counts.
  - Count adjacent pair frequencies across the whole vocabulary.
  - Pick the most-frequent pair.
  - Merge that pair everywhere it occurs.
  - Repeat steps 2-4 until the desired number of merges.
"""

import collections


def init_vocab(corpus_words: list[str]) -> dict[tuple[str, ...], int]:
    """Each word becomes a tuple of (chars + '</w>'); value is its frequency."""
    counts = collections.Counter(corpus_words)
    return {tuple(list(word) + ["</w>"]): freq for word, freq in counts.items()}


def get_pair_counts(
    vocab: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int]:
    """Count adjacent (a, b) pair frequencies across the whole vocabulary."""
    pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str],
    vocab: dict[tuple[str, ...], int],
) -> dict[tuple[str, ...], int]:
    """Replace every adjacent (a, b) in vocab's keys with 'ab'."""
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


def main() -> None:
    corpus = "low low low low low lower newer newer newer newer wider".split()
    vocab = init_vocab(corpus)

    print("Initial vocab:")
    for k, v in vocab.items():
        print(f"  {v}: {' '.join(k)}")
    print()

    merges = []
    for step in range(10):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair, best_count = max(pairs.items(), key=lambda kv: kv[1])
        merged = best_pair[0] + best_pair[1]
        print(
            f"Step {step + 1}: merge ({best_pair[0]!r}, {best_pair[1]!r}) "
            f"-> {merged!r}  (freq={best_count})"
        )
        merges.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

    print()
    print("Final vocab:")
    for k, v in vocab.items():
        print(f"  {v}: {' '.join(k)}")


if __name__ == "__main__":
    main()
