import torch

from mygpt.model import GPT
from mygpt.tokenizer import CharTokenizer


def save_checkpoint(model: "GPT", tokenizer: "CharTokenizer", path: str) -> None:
    """Bundle model weights, tokenizer, and architecture into one .ckpt file."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_chars": tokenizer.chars,
            "config": {
                "vocab_size":     model.vocab_size,
                "embed_dim":      model.embed_dim,
                "num_heads":      model.num_heads,
                "num_kv_heads":   getattr(model, "num_kv_heads", model.num_heads),
                "num_layers":     model.num_layers,
                "max_seq_len":    model.max_seq_len,
                "norm_type":      getattr(model, "norm_type", "layer"),
                "position_type":  getattr(model, "position_type", "learned"),
            },
        },
        path,
    )


def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`.

    Always loads to CPU; the caller is responsible for `.to(device)` afterwards.
    Pre-Ch.24 checkpoints have no `norm_type` field; pre-Ch.25 checkpoints have
    no `position_type` field; pre-Ch.26 checkpoints have no `num_kv_heads` field.
    All three default to their original behaviour (``"layer"``, ``"learned"``,
    and ``num_kv_heads = num_heads``) so old `.ckpt` files continue to load.
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
    model = GPT(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads", config["num_heads"]),
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,
        norm_type=config.get("norm_type", "layer"),
        position_type=config.get("position_type", "learned"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer
