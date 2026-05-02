import math

import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids."""
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's RNGs across whatever devices are available."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device(arg: str = "auto") -> torch.device:
    """Resolve a device spec to a torch.device.

    ``"auto"`` prefers CUDA over MPS over CPU. The other strings
    (``"cuda"``, ``"mps"``, ``"cpu"``) are passed through.
    """
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


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


def cosine_warmup_lr(
    step: int, warmup: int, total: int, max_lr: float, min_lr: float = 0.0
) -> float:
    """Cosine learning-rate schedule with linear warmup.

    Step indexing is 1-based: at step 1, returns max_lr / warmup (or max_lr if
    warmup == 0). After step >= total, returns min_lr.
    """
    if warmup > 0 and step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def estimate_val_loss(
    model: "GPT",
    val_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    n_eval_batches: int = 10,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_eval_batches):
        x, y = get_batch(val_data, batch_size, seq_len)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)
