import os

import torch

from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer, CharTokenizer


def save_checkpoint(
    model: "GPT",
    tokenizer: "CharTokenizer | BPETokenizer",
    path: str,
) -> None:
    """Bundle model weights, tokenizer, and architecture into one .ckpt file.

    The tokenizer's full state is serialised so generation can reload without
    retraining.  ``tokenizer_kind`` is ``"char"`` (Part-I default) or ``"bpe"``
    (Ch.23+); the field is absent on pre-Ch.28 checkpoints, in which case
    ``load_checkpoint`` falls back to ``"char"``.

    The save is atomic: bytes are written to ``path + ".tmp"`` and then renamed
    over ``path``, so a Ctrl-C during writing leaves any prior checkpoint at
    ``path`` intact.
    """
    if isinstance(tokenizer, BPETokenizer):
        tokenizer_kind = "bpe"
        tokenizer_state = {
            "chars": tokenizer.chars,
            "merges": tokenizer.merges,
        }
    else:
        tokenizer_kind = "char"
        tokenizer_state = {"chars": tokenizer.chars}

    payload = {
        "model_state_dict": model.state_dict(),
        # Part-I-and-earlier compatibility: char tokenizer state continues to
        # be mirrored under the original key.  Newer loaders prefer the
        # `tokenizer_kind` + `tokenizer_state` pair below.
        "tokenizer_chars": tokenizer.chars,
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_state": tokenizer_state,
        "config": {
            "vocab_size":     model.vocab_size,
            "embed_dim":      model.embed_dim,
            "num_heads":      model.num_heads,
            "num_kv_heads":   getattr(model, "num_kv_heads", model.num_heads),
            "num_layers":     model.num_layers,
            "max_seq_len":    model.max_seq_len,
            "norm_type":      getattr(model, "norm_type", "layer"),
            "position_type":  getattr(model, "position_type", "learned"),
            "tokenizer_kind": tokenizer_kind,
        },
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    import os
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer | BPETokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`.

    Always loads to CPU; the caller is responsible for `.to(device)` afterwards.
    Pre-Ch.24 checkpoints have no `norm_type` field; pre-Ch.25 checkpoints have
    no `position_type` field; pre-Ch.26 checkpoints have no `num_kv_heads` field;
    pre-Ch.28 checkpoints have no `tokenizer_kind` field.
    All four default to their original behaviour (``"layer"``, ``"learned"``,
    ``num_kv_heads = num_heads``, and ``"char"``) so old `.ckpt` files continue
    to load.
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    tokenizer_kind = config.get("tokenizer_kind", ckpt.get("tokenizer_kind", "char"))
    if tokenizer_kind == "bpe":
        state = ckpt["tokenizer_state"]
        tokenizer = BPETokenizer(state["chars"], state["merges"])
    else:
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
