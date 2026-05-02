"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 4 the package knows about the running-example vocabulary,
how to turn tokens into id tensors (Chapter 3), and how to seed PyTorch's
random number generator for reproducibility (this chapter).
"""

import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids."""
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's CPU random number generator.

    Call this at the start of any experiment that uses `torch.randn`,
    `torch.rand`, or randomly initialised modules — it makes the run
    reproducible across machines.
    """
    torch.manual_seed(seed)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
    sample = list(VOCAB)
    ids = to_ids(sample)
    print(f"to_ids({sample}) = {ids}")
    set_seed(0)
    print(f"after set_seed(0), torch.randn(3) = {torch.randn(3)}")
