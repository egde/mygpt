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
