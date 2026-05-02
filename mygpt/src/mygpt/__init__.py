"""mygpt — a tiny GPT-2-like language model, built one chapter at a time."""

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


@torch.no_grad()
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


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_O = nn.Linear(head_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V
        return self.out_drop(self.W_O(out))


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the (cos, sin) lookup table for rotary position embeddings.

    head_dim must be even. Returns two tensors of shape ``(max_seq_len, head_dim // 2)``.

    The i-th frequency is ``θ_i = base ** (-2i / head_dim)`` for ``i ∈ [0, head_dim/2)``;
    the angle for position ``m`` and pair ``i`` is ``θ_i · m``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x`` of shape ``(..., T, head_dim)``.

    Pairs are formed by even/odd dim indices: dim ``2i`` and dim ``2i+1`` rotate
    together by angle ``θ_i · pos``.
    """
    T = x.shape[-2]
    cos_t = cos[:T]  # (T, head_dim // 2)
    sin_t = sin[:T]
    # Broadcast over leading dims (e.g., batch and head)
    while cos_t.dim() < x.dim() - 1:
        cos_t = cos_t.unsqueeze(0)
        sin_t = sin_t.unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos_t - x_odd * sin_t
    rotated_odd = x_even * sin_t + x_odd * cos_t
    out = torch.stack([rotated_even, rotated_odd], dim=-1)
    return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        position_type: str = "learned",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.position_type = position_type

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

        if position_type == "rope":
            cos, sin = precompute_rope_cache(self.head_dim, max_seq_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.position_type == "rope":
            Q = apply_rope(Q, self.rope_cos, self.rope_sin)
            K = apply_rope(K, self.rope_cos, self.rope_sin)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_drop(self.W_O(out))


class MLP(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) / torch.sqrt(var + self.eps)
        return x_normed * self.weight + self.bias


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Llama / Mistral default).

    Compared to LayerNorm:
      - drops the bias term,
      - drops the mean subtraction (no centring),
      - normalises by the root-mean-square of x along the last axis.

    Forward: ``x / sqrt(mean(x²) + eps) * weight``.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _make_norm(embed_dim: int, norm_type: str) -> nn.Module:
    """Norm-class selector. `norm_type` is 'layer' or 'rms'."""
    if norm_type == "layer":
        return LayerNorm(embed_dim)
    if norm_type == "rms":
        return RMSNorm(embed_dim)
    raise ValueError(f"unknown norm_type: {norm_type!r} (expected 'layer' or 'rms')")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        max_seq_len=64,
        dropout=0.0,
        norm_type="layer",
        position_type="learned",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.position_type = position_type
        self.ln1 = _make_norm(embed_dim, norm_type)
        self.mha = MultiHeadAttention(
            embed_dim, num_heads, max_seq_len, dropout, position_type=position_type
        )
        self.ln2 = _make_norm(embed_dim, norm_type)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Full GPT-2-style decoder-only transformer with weight-tied head."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len=64, dropout=0.0, norm_type="layer",
                 position_type="learned"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.norm_type = norm_type
        self.position_type = position_type

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        if position_type == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        # If position_type == "rope", positions are applied inside attention;
        # no learned position embedding is allocated here.
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout,
                             norm_type, position_type)
            for _ in range(num_layers)
        ])
        self.ln_f = _make_norm(embed_dim, norm_type)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"input length T={T} exceeds max_seq_len={self.max_seq_len}")
        x = self.token_embedding(ids)
        if self.position_type == "learned":
            positions = torch.arange(T, device=ids.device)
            x = x + self.position_embedding(positions)
        x = self.embed_drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = x @ self.token_embedding.embedding.weight.T
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def generate(
    model: "GPT",
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively generate max_new_tokens after prompt_ids.

    Inputs:
        model:           a trained GPT (or compatible Module returning logits).
        prompt_ids:      long tensor of shape (B, T_prompt).
        max_new_tokens:  how many tokens to append.
        temperature:     softmax temperature; <1 sharpens, >1 flattens.
        top_k:           if given, restrict sampling to the top_k most-likely
                         tokens at each step.

    Output:
        long tensor of shape (B, T_prompt + max_new_tokens).
    """
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        ids_cond = (
            ids[:, -model.max_seq_len :] if ids.shape[1] > model.max_seq_len else ids
        )
        with torch.no_grad():
            logits, _ = model(ids_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_ids], dim=1)
    return ids


import json


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
    no `position_type` field. Both default to their original behaviour
    (``"layer"`` and ``"learned"``) so old `.ckpt` files continue to load.
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
    model = GPT(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,
        norm_type=config.get("norm_type", "layer"),
        position_type=config.get("position_type", "learned"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer


def _train_command(args) -> None:
    device = pick_device(args.device)

    with open(args.text_file) as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode(text).to(device)

    # Train/val split (val_split = 0 keeps Ch.17-style "all data is train")
    if args.val_split > 0.0:
        n_train = int((1.0 - args.val_split) * len(data))
        train_data = data[:n_train]
        val_data = data[n_train:]
    else:
        train_data = data
        val_data = None

    set_seed(0)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        norm_type=args.norm,
        position_type=args.position,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device:       {device}")
    print(f"precision:    {args.precision}")
    print(f"norm:         {args.norm}")
    print(f"position:     {args.position}")
    print(f"corpus chars: {len(text):,}")
    print(f"train chars:  {len(train_data):,}")
    if val_data is not None:
        print(f"val chars:    {len(val_data):,}")
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")
    print(f"schedule:     {args.schedule} (warmup={args.warmup})")
    print(f"max_grad_norm:{args.max_grad_norm}")

    set_seed(42)
    for step in range(1, args.steps + 1):
        # LR schedule
        if args.schedule == "cosine":
            lr_t = cosine_warmup_lr(step, args.warmup, args.steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_t

        x, y = get_batch(train_data, args.batch_size, args.seq_len)
        optimizer.zero_grad()
        if args.precision == "bf16":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        if args.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        is_print_step = step == 1 or step % args.print_every == 0 or step == args.steps
        is_val_step = (
            val_data is not None
            and args.val_every > 0
            and (step % args.val_every == 0 or step == args.steps)
        )
        if is_print_step or is_val_step:
            line = f"step {step:>5}: loss = {loss.item():.4f}"
            if is_val_step:
                vl = estimate_val_loss(
                    model, val_data, args.batch_size, args.seq_len
                )
                line += f"  val = {vl:.4f}"
            if args.schedule == "cosine":
                line += f"  lr = {lr_t:.2e}"
            print(line)

    save_checkpoint(model, tokenizer, args.output)
    print(f"\nsaved checkpoint to {args.output}")


def _generate_command(args) -> None:
    device = pick_device(args.device)
    print(f"device: {device}\n")
    model, tokenizer = load_checkpoint(args.checkpoint)
    model.to(device)
    set_seed(args.seed)
    prompt = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    if args.precision == "bf16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = generate(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
    else:
        out = generate(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(out[0]))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mygpt",
        description="Tiny GPT trainer and text generator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a GPT on a plain-text file.")
    p_train.add_argument("text_file", help="Path to a UTF-8 text file.")
    p_train.add_argument("--output", default="model.ckpt", help="Checkpoint output path.")
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--seq-len", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--embed-dim", type=int, default=64)
    p_train.add_argument("--num-heads", type=int, default=4)
    p_train.add_argument("--num-layers", type=int, default=4)
    p_train.add_argument("--max-seq-len", type=int, default=64)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--print-every", type=int, default=500)
    p_train.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda → mps → cpu in that order.",
    )
    p_train.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Forward-pass precision. fp32 (default) is bit-deterministic; bf16 uses torch.autocast.",
    )
    p_train.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of the corpus held out as validation data (0.0 = none, default).",
    )
    p_train.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Print val loss every N steps. Requires --val-split > 0.",
    )
    p_train.add_argument(
        "--schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="LR schedule. 'constant' (default) holds at --lr; 'cosine' linearly warms up over --warmup steps then cosine-decays to 0.",
    )
    p_train.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup steps for the cosine schedule (no effect if --schedule=constant).",
    )
    p_train.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Gradient-norm clip threshold. 0.0 (default) disables clipping.",
    )
    p_train.add_argument(
        "--norm",
        choices=["layer", "rms"],
        default="layer",
        help="Normalisation: 'layer' (default; LayerNorm, Ch.10) or 'rms' (RMSNorm, Llama default).",
    )
    p_train.add_argument(
        "--position",
        choices=["learned", "rope"],
        default="learned",
        help="Position embedding: 'learned' (default; nn.Embedding, Ch.12) or 'rope' (rotary, Llama default).",
    )
    p_train.set_defaults(func=_train_command)

    p_gen = sub.add_parser("generate", help="Generate text from a checkpoint.")
    p_gen.add_argument("--checkpoint", required=True)
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max-new-tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=10)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda → mps → cpu in that order.",
    )
    p_gen.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Forward-pass precision. fp32 (default) is bit-deterministic; bf16 uses torch.autocast.",
    )
    p_gen.set_defaults(func=_generate_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
