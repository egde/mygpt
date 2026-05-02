import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids."""
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's CPU random number generator."""
    torch.manual_seed(seed)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) pairs from a 1-D token tensor."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y
