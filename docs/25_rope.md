---
title: 25. RoPE — rotary position embeddings
nav_order: 7
parent: Part II — Advanced Topics
---

# Chapter 25 — RoPE: rotary position embeddings

Chapter 12 added position information to the model with a *learned* `nn.Embedding(max_seq_len, embed_dim)`. The §15.9 experiment exposed its weakness: at any position the model has not been trained on, the position embedding is still at its random initialisation, and generation drifts into garbage. Modern open-weight LLMs — Llama, Mistral, Qwen, GPT-NeoX — all replaced learned position embeddings with **RoPE**, a parameter-free rotation applied to the query and key vectors inside attention.

By the end of this chapter you will have:

- understood RoPE's central idea — pair up the dimensions of $Q$ and $K$, rotate each pair by an angle proportional to its position,
- added two helpers (`precompute_rope_cache`, `apply_rope`) and a `position_type` knob to `MultiHeadAttention`, `TransformerBlock`, and `GPT`,
- added a `--position {learned, rope}` CLI flag (default `learned` so prior chapters bit-reproduce),
- watched RoPE *replace* the learned position embedding entirely (the model is **4,096 parameters smaller** at our toy scale — exactly the size of the dropped `nn.Embedding(64, 64)`),
- seen RoPE produce a **lower training loss** than learned positions on the same corpus (1.7812 vs 2.0785 at step 2000) — the inductive bias is helping more than the lost parameters hurt at this scale,
- understood the structural property — RoPE works at *any* position index, so generation past trained `seq_len` no longer collapses (the §15.9 failure mode is gone).

Backward compat is preserved: pre-Ch.25 checkpoints have no `position_type` field and default to `"learned"` on load.

---

## 25.1 Setup

This chapter assumes Chapter 24 — `mygpt/` has `RMSNorm`, `_make_norm`, `--norm`, and `norm_type` in the checkpoint config. We are about to add three more pieces along the same plan: helpers, threading, CLI flag.

```bash
ls tinyshakespeare.txt || curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 25.2 Why position embeddings, again, briefly

Recall §6: self-attention is *permutation-invariant* — shuffling the input tokens produces the same set of attention scores (just permuted). Without something position-dependent, the model literally cannot tell `"I love AI !"` from `"AI love I !"`. Chapter 12 fixed this by adding a *learned* per-position vector to the token embedding before the first attention layer:

$$
x_i \leftarrow \text{TokenEmbed}(\text{id}_i) + \text{PositionEmbed}(i)
$$

That's straightforward but has two well-known weaknesses:

1. **Bounded.** `PositionEmbed` is `nn.Embedding(max_seq_len, embed_dim)` — a finite lookup table. Position $i \geq \text{max\_seq\_len}$ has no row to look up; even position $i$ within `max_seq_len` but outside the *trained* range is a row at random initialisation (§15.9 demonstrated this bites in practice).
2. **Information lives in the wrong place.** Position is added once at the input. Attention then mixes information across positions in every layer — by the time we are deep in the model, the position signal has been smeared across many other features.

RoPE addresses both. It encodes position as a **rotation** applied to $Q$ and $K$ inside *every* attention layer. There are no learned position parameters, so the position signal is mathematically valid at any index. And because position is injected directly into $Q$ and $K$, the dot-product attention naturally measures *relative position* — the angle between the rotated $Q_i$ and $K_j$ depends only on $i - j$, not on absolute $i$ or $j$.

---

## 25.3 The rotation, geometrically

Take a query vector $\mathbf{q} \in \mathbb{R}^{d_h}$ at position $m$ inside one attention head. RoPE pairs up the dimensions of $\mathbf{q}$ — dim $0$ with dim $1$, dim $2$ with dim $3$, …, dim $d_h - 2$ with dim $d_h - 1$ — and rotates each pair by an angle.

For pair $i \in [0, d_h/2)$, the angle is $\theta_i \cdot m$, where

$$
\theta_i = \frac{1}{\text{base}^{2i / d_h}}, \qquad \text{base} = 10{,}000.
$$

The first pair ($i = 0$) has angle $1 \cdot m = m$ — it rotates by *one radian per position*. The last pair ($i = d_h/2 - 1$) has angle $\theta \approx 1/\text{base}$ — it rotates by very nearly zero radians per position. The pairs in between span the geometric series between those extremes. This spectrum of angular speeds is what makes RoPE encode position at multiple scales simultaneously.

Rotation in $\mathbb{R}^2$ is one matrix:

$$
R(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}.
$$

So for pair $i$ at position $m$, RoPE applies $R(\theta_i m)$ to the two-dimensional sub-vector $(q_{2i}, q_{2i+1})$:

$$
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix}
= R(\theta_i m) \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
= \begin{pmatrix} q_{2i}\cos(\theta_i m) - q_{2i+1}\sin(\theta_i m) \\ q_{2i}\sin(\theta_i m) + q_{2i+1}\cos(\theta_i m) \end{pmatrix}.
$$

The same rotation is applied to $K$ at each position. (We do *not* rotate $V$ — values are content carried unchanged into the output.)

The reason this measures *relative* position: when you compute the attention dot product $\mathbf{q}_i^\top \mathbf{k}_j$ between a rotated query at position $i$ and a rotated key at position $j$, the dot product of two rotated 2-vectors is a function of $\cos((i - j) \theta)$ — only the *difference* $i - j$ matters. Two tokens five positions apart get the same rotational relationship regardless of where they sit in the sequence.

---

## 25.4 The two helpers

We need a precomputed `(cos, sin)` lookup keyed by position, and a function that applies it to a tensor of shape `(..., T, d_h)`.

**Append the following two helpers to** 📄 `src/mygpt/attention.py` (right before `class MultiHeadAttention`):

```python
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
```

The cache holds `cos` and `sin` tables — one row per position, one column per pair. `apply_rope` looks up the first `T` rows for our current sequence and broadcasts over batch and head dimensions. The math at the end (`stack` + `flatten(-2)`) interleaves the rotated even and odd parts back into the original layout — `[even_0, odd_0, even_1, odd_1, …]`.

(There are two RoPE conventions in the wild — "interleaved" pairs (which we use here) versus "block" pairs that split the head into halves. They are mathematically equivalent for training; Llama 2's reference code uses the block variant. We pick the interleaved version because the math reads more naturally as "pair up consecutive dims".)

---

## 25.5 Threading `position_type` through attention, blocks, and GPT

Three changes, each small. `MultiHeadAttention` gets a `position_type` parameter and, when it is `"rope"`, registers a precomputed cache and applies it inside `forward`. `TransformerBlock` and `GPT` just pass the parameter through.

**Replace `MultiHeadAttention` in** 📄 `src/mygpt/attention.py`:

```python
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
```

Three changes from Ch.24's version: a new `position_type` parameter; a conditional `register_buffer("rope_cos"/"rope_sin", ...)` in `__init__`; and a conditional `apply_rope` block inside `forward`, between the head-split and the score computation.

**Replace `TransformerBlock` in** 📄 `src/mygpt/block.py`:

```python
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
```

(One new constructor parameter, passed through to `MultiHeadAttention`. Forward unchanged.)

**Replace `GPT` in** 📄 `src/mygpt/model.py`:

```python
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
        logits = x @ self.token_embedding.embedding.weight.T  # (B, T, V)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
```

Three changes from Ch.24's `GPT`:
- new `position_type` parameter, stored as `self.position_type`,
- the `position_embedding` allocation is now *conditional* on `position_type == "learned"`,
- `forward` skips the position-embedding addition when `position_type != "learned"`.

When `position_type="rope"`, the `nn.Embedding(max_seq_len, embed_dim)` is **not allocated at all** — the model is `max_seq_len * embed_dim` parameters smaller. For our toy `64 * 64 = 4,096` parameters; for GPT-2 small it would be `1024 * 768 ≈ 786 k`.

---

## 25.6 Checkpoint format adds `position_type`

Same backward-compat dance as Ch.24. Add `position_type` to the saved config; default `"learned"` on load so pre-Ch.25 checkpoints continue to work.

**Replace `save_checkpoint` in** 📄 `src/mygpt/checkpoint.py`:

```python
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
```

**Replace `load_checkpoint` in** 📄 `src/mygpt/checkpoint.py`:

```python
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
```

---

## 25.7 The `--position` CLI flag

Three edits to `_train_command` (parallel to Ch.24's `--norm` work): pass `position_type=args.position` to `GPT(...)`, add a `position:` print line, and register the flag on `p_train`.

**In `_train_command`, the `GPT(...)` constructor call:**

```python
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
```

**And the print block (right after `norm:`):**

```python
    print(f"device:       {device}")
    print(f"precision:    {args.precision}")
    print(f"norm:         {args.norm}")
    print(f"position:     {args.position}")
    print(f"corpus chars: {len(text):,}")
    # … rest unchanged
```

**In `main`'s argparse setup, add to `p_train`** (right after the `--norm` block, before `set_defaults(...)`):

```python
    p_train.add_argument(
        "--position",
        choices=["learned", "rope"],
        default="learned",
        help="Position embedding: 'learned' (default; nn.Embedding, Ch.12) or 'rope' (rotary, Llama default).",
    )
```

---

## 25.8 Backward-compat: defaults still reproduce Ch.21

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output sh-learned.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
norm:         layer
position:     learned
corpus chars: 1,115,394
train chars:  1,115,394
vocab_size:   65
params:       207,296
steps:        2000
schedule:     constant (warmup=0)
max_grad_norm:0.0
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to sh-learned.ckpt
```

Same loss curve as every previous "default" run. The new `position: learned` line appears in the header and nothing else changes. Backward-compat preserved.

---

## 25.9 The RoPE run

```bash
uv run mygpt train tinyshakespeare.txt --device mps --position rope --output sh-rope.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
norm:         layer
position:     rope
corpus chars: 1,115,394
train chars:  1,115,394
vocab_size:   65
params:       203,200
steps:        2000
schedule:     constant (warmup=0)
max_grad_norm:0.0
step     1: loss = 55.6207
step   500: loss = 2.3110
step  1000: loss = 1.9569
step  1500: loss = 1.8594
step  2000: loss = 1.7812

saved checkpoint to sh-rope.ckpt
```

Three things to read off:

**1. Parameter count drops from 207,296 to 203,200** — exactly **4,096 fewer parameters**. That is `max_seq_len * embed_dim = 64 * 64 = 4,096`, the size of the dropped `nn.Embedding(64, 64)` position table. (RoPE's `cos` and `sin` tables of shape `(64, 32)` are *buffers*, not parameters: they are precomputed deterministically from the head dimension, not learned, so they don't count toward the model's parameter budget.)

**2. Step-1 loss is *higher* than learned (55.6 vs 41.0).** Both models start from the same `set_seed(0)` initialisation, but with RoPE the rotation immediately mixes random initial weights into a different random direction, producing a confidently-different wrong prediction on the step-1 batch. As gradient descent kicks in, this extra randomness disappears — by step 500 RoPE is *ahead* of learned.

**3. Final loss is *substantially better* than learned (1.78 vs 2.08, a ~14% reduction at the same parameter scale).** This is the real win of RoPE: the model gets position information at every attention layer (not just at input), and the relative-position structure is baked in for free. On a small corpus like Tiny Shakespeare, this inductive bias is worth more than the lost parameters cost. (At GPT-2-scale and beyond the gap narrows; RoPE's win there is mostly the longer-context generalisation property covered in §25.10.)

---

## 25.10 The structural property: position-extrapolation by construction

Recall §15.9: a model trained with `seq_len=8` and a learned position embedding produced sensible tokens for positions 0–7 and started drifting at position 8 onward. That drift was *forced* — positions 8 through `max_seq_len-1` had position-embedding rows at random initialisation that the gradients had never touched.

With RoPE, **no learned position parameter exists at all**. The cos/sin lookup is deterministically computed from `head_dim`, `max_seq_len`, and the chosen base — the same formula gives valid values at *any* position. A model trained with `seq_len=8` and RoPE can generate at position 100 using the *same* mathematical formula it used at position 5: extrapolation past trained positions stops being a hard wall and becomes a soft, gradual quality drop (the model has just never *seen* such long-range relative offsets during training, so it has not learned to use them well — but the position machinery itself is intact).

We do not re-run the §15.9 experiment here because retraining a `seq_len=8` model takes another ~30 s. The §25.11 experiments include the recipe.

---

## 25.11 Sampling from each model

```bash
uv run mygpt generate --checkpoint sh-learned.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output:**

```text
device: cpu

ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

Same Ch.17 §17.6 sample we have been seeing all of Part II. Backward-compat preserved.

```bash
uv run mygpt generate --checkpoint sh-rope.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output:**

```text
device: cpu

ROMEO:
Thy whis whaths be dedood nevery words welevet and furwight nawaren? hish houghterevissed one bried
thou lordied we allosendy thour non my fathereven ost to a man is hither.
Fow I not sweel this. Boo
```

The RoPE sample is qualitatively different — and arguably *better*. The text uses more sentence-shaped punctuation (`?`, `.`), the word fragments cluster into pronounceable shapes, and the rhythm has more variation (long-short-long phrases). At a final loss of 1.78 vs 2.08, the RoPE model is genuinely more confident in better choices.

---

## 25.12 Experiments

1. **The §15.9 redux on RoPE.** Retrain with the §15.9 setup — short trained `seq_len=8` but `max_seq_len=64`:
   ```bash
   # Learned positions (will fail past pos 7, like §15.9)
   uv run mygpt train tinyshakespeare.txt --device mps \
     --seq-len 8 --steps 1000 \
     --position learned --output sh-short-learned.ckpt
   # RoPE (graceful fall-off)
   uv run mygpt train tinyshakespeare.txt --device mps \
     --seq-len 8 --steps 1000 \
     --position rope --output sh-short-rope.ckpt
   ```
   Generate 100 tokens from each. The learned-position model collapses past position 7 (the failure mode from §15.9). The RoPE model continues to produce coherent-shaped text — the quality fades gradually as positions stretch beyond what was trained, rather than failing at a hard boundary.

2. **The cos/sin cache.** `cos, sin = precompute_rope_cache(head_dim=16, max_seq_len=8)`. Print `cos[0]` (the angles at position 0 — should be all `1.0`s) and `cos[1]` (position 1, with one rotation per pair). The first column is `cos(1) ≈ 0.5403`; the last column is `cos(1/10000^((d_h-2)/d_h)) ≈ cos(0.0001) ≈ 1.0` (the slowest rotation). This is the spectrum of frequencies §25.3 described.

3. **`apply_rope` is its own inverse at position 0.** At `pos = 0`, every angle is zero, every cosine is 1, every sine is 0. So `apply_rope` at position 0 is literally the identity. Verify: build a random tensor `x = torch.randn(2, 4, 1, 16)` (B=2 batch, h=4 heads, T=1 single position, d_h=16), apply rope, check `torch.allclose(out, x)`.

4. **Saved-config inspection.** `python -c 'import torch; print(torch.load("sh-rope.ckpt", map_location="cpu")["config"])'`. The output now includes both `'norm_type': 'layer'` and `'position_type': 'rope'`. Pre-Ch.24 checkpoints have neither field; pre-Ch.25 checkpoints have only `norm_type`. All three load through `load_checkpoint` because of the `.get(..., default)` fallbacks.

After each experiment, restore any file you changed before moving on.

---

## 25.13 Exercises

1. **Why pair dimensions, not rotate the whole vector at once?** A single $d_h$-dimensional rotation matrix has $\binom{d_h}{2}$ degrees of freedom. By pairing dimensions and rotating each pair independently, RoPE uses only $d_h/2$ scalars (the angles) — a *block-diagonal* rotation. Argue that this is exactly the right amount of expressiveness: each pair encodes one scalar position, the model gets $d_h/2$ independent position channels, and the parameter count of the rotation is zero (the angles are deterministic from the position).

2. **Why $\theta_i$ as a geometric sequence?** A natural alternative would be a linear progression: $\theta_i = i / d_h$ instead of $\theta_i = \text{base}^{-2i/d_h}$. Argue why the geometric ladder is preferable: the slow pairs encode *long-range* position differences (their angle changes barely at all between adjacent positions), the fast pairs encode *short-range* differences. With a linear ladder the highest-frequency pair would alias at position $\pi d_h / 2 \approx 25$, well within typical contexts.

3. **Why no rotation on V?** $Q$ and $K$ jointly determine *which* positions the model attends to — they participate in the dot product that picks the attention weights. $V$ is the *content* that gets carried through. Rotating $V$ would add position information to content, mixing the two roles. Argue that the cleanest design separates the two: $Q, K$ encode "where to attend" (rotated); $V$ encodes "what to carry" (not rotated).

4. **`cos[:T]` slicing.** Inside `apply_rope`, we do `cos_t = cos[:T]`. For a model with `max_seq_len=64`, `cos` has shape `(64, head_dim/2)`. At inference time with $T = 8$, we slice the first 8 rows. Argue why this is correct (positions are 0-indexed and the cache is laid out in position order) and why nothing breaks if we ever try `T > max_seq_len` (the existing `MultiHeadAttention.forward` raises `ValueError` *before* we reach `apply_rope`).

---

## 25.14 What's next

The next chapter, **Chapter 26 — GQA: grouped-query attention**, makes one more architectural change before we run the modern recipe end-to-end. GQA shares each $K$ and $V$ head across multiple $Q$ heads — a memory-saving trick that becomes essential at GPT-2 scale and beyond, where the KV cache during generation dominates the per-step inference cost.

> **Looking ahead — what to remember from this chapter**
>
> 1. RoPE encodes position as a rotation of `(q_{2i}, q_{2i+1})` pairs and `(k_{2i}, k_{2i+1})` pairs by angles `θ_i · pos`. No learned position parameters.
> 2. The frequency ladder `θ_i = base^{-2i/d_h}` covers many timescales: the first pair rotates ~1 radian per position, the last barely rotates at all.
> 3. Position information is injected at *every* attention layer (not just at input), and the dot-product structure makes attention naturally measure *relative* position.
> 4. Backward-compat: `--position learned` (default) reproduces every prior chapter's run; `--position rope` opts into the new behaviour. Pre-Ch.25 checkpoints default to `"learned"` on load.

On to [Chapter 26 — GQA: grouped-query attention](26_gqa.md).
