import collections
import json

import torch


class CharTokenizer:
    """Character-level tokenizer.

    Vocabulary = the sorted list of distinct characters seen in the training
    text. Token id = position in that list.
    """

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer whose vocabulary is the alphabet of `text`."""
        return cls(sorted(set(text)))

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` to a 1-D long tensor of ids."""
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode a 1-D tensor of ids back to a string."""
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        """Persist the tokenizer to a JSON file at `path`."""
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Reload a tokenizer from a JSON file produced by `save`."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"])


class BPETokenizer:
    """Byte-pair-encoding tokenizer.

    Training: scan the text as a sequence of single-character symbols, count
    adjacent-pair frequencies, merge the most-frequent pair into a new symbol,
    repeat for `num_merges` iterations.

    The serialised state is two JSON-friendly objects:
      - `chars`: the initial alphabet (sorted unique characters of the training text)
      - `merges`: the ordered list of (a, b) string-pair merge rules
    The vocabulary is `chars + [a + b for (a, b) in merges]`, with id `i` given
    by position in that combined list.
    """

    def __init__(self, chars: list[str], merges: list[tuple[str, str]]) -> None:
        self.chars = list(chars)
        self.merges = [tuple(m) for m in merges]
        # Build the vocab in id order: alphabet first, then merges in creation order.
        vocab = list(self.chars)
        for a, b in self.merges:
            vocab.append(a + b)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = {s: i for i, s in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str, num_merges: int) -> "BPETokenizer":
        """Train a BPE tokenizer on `text` for `num_merges` merge iterations."""
        chars = sorted(set(text))
        symbols = list(text)
        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            pairs: collections.Counter[tuple[str, str]] = collections.Counter()
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            a, b = best
            merged = a + b
            merges.append((a, b))
            new_symbols: list[str] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return cls(chars, merges)

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` to a 1-D long tensor of ids by applying the merges in order."""
        symbols = list(text)
        for a, b in self.merges:
            new_symbols: list[str] = []
            i = 0
            n = len(symbols)
            merged = a + b
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return torch.tensor([self.stoi[s] for s in symbols], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode a 1-D tensor of ids back to a string."""
        return "".join(self.vocab[int(i)] for i in ids)

    def save(self, path: str) -> None:
        """Persist the tokenizer to a JSON file at `path`."""
        with open(path, "w") as f:
            json.dump({"chars": self.chars, "merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Reload a tokenizer from a JSON file produced by `save`."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"], [tuple(m) for m in data["merges"]])
