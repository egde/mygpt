---
title: 24. RMSNorm replaces LayerNorm
nav_order: 6
parent: Part II — Advanced Topics
---

# Chapter 24 — RMSNorm replaces LayerNorm

`LayerNorm` (Chapter 10) does two things to each token vector: it subtracts the mean, then it divides by the standard deviation. Modern transformer architectures — Llama, Mistral, Qwen, every Llama-derived open-weight model — use a simpler operation called **RMSNorm**: skip the mean subtraction, divide by the root mean square instead, and drop the bias. Slightly fewer parameters, slightly faster, no observable training-quality penalty.

By the end of this chapter you will have:

- understood the difference between LayerNorm and RMSNorm in one formula,
- added an `RMSNorm` class to `mygpt` next to `LayerNorm`,
- threaded a `norm_type` config option through `GPT`, `TransformerBlock`, and the checkpoint format,
- added a `--norm {layer, rms}` flag to `mygpt train` (default `layer` so Ch.17–23 expected outputs continue to bit-reproduce),
- trained the same Tiny Shakespeare model with both norms and compared their parameter counts and loss curves,
- confirmed that every pre-Ch.24 checkpoint still loads correctly (the new `norm_type` field defaults to `"layer"` when missing).

---

## 24.1 Setup

This chapter assumes Chapter 23 — `mygpt/` has `BPETokenizer` and an `import collections`. None of those are touched in this chapter; only the model architecture changes.

Confirm Tiny Shakespeare is in place:

```bash
ls tinyshakespeare.txt || curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 24.2 LayerNorm vs RMSNorm in one formula

Recall LayerNorm from Chapter 10. For each token vector $\mathbf{x} \in \mathbb{R}^{C}$:

$$
\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu(\mathbf{x})}{\sqrt{\sigma^2(\mathbf{x}) + \varepsilon}} \cdot \mathbf{w} + \mathbf{b},
\quad
\mu = \tfrac{1}{C}\sum_i x_i,
\quad
\sigma^2 = \tfrac{1}{C}\sum_i (x_i - \mu)^2.
$$

Two trainable vectors of size $C$ each: gain $\mathbf{w}$ and bias $\mathbf{b}$. Total parameters per LayerNorm instance: $2C$.

RMSNorm (Zhang & Sennrich, 2019) drops both the mean subtraction and the bias:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{C}\sum_i x_i^2 + \varepsilon}} \cdot \mathbf{w}.
$$

One trainable vector of size $C$: gain $\mathbf{w}$. Total parameters per RMSNorm instance: $C$. **Half the parameters, half the arithmetic ops in the forward pass.**

Why does this work just as well? Empirically, the *centering* step in LayerNorm doesn't carry much load — the invariance to the mean of $\mathbf{x}$ that LayerNorm provides is mostly redundant with what attention and the MLP already give the model. The bias term is similarly inessential: every linear layer in the model already has a bias (or doesn't, by design choice), so a separate norm-bias is a duplicate degree of freedom. The Zhang & Sennrich paper trained a Transformer on a translation benchmark with each variant and showed indistinguishable convergence. Llama adopted RMSNorm in 2023; everything downstream from Llama uses it.

There is no theory that *RMSNorm is better*. The argument is pragmatic: same training quality, slightly cheaper, simpler code.

---

## 24.3 The `RMSNorm` class

Identical shape to `LayerNorm` from §10 — a `nn.Module` with one trainable `weight` parameter. The forward pass is two lines.

**Append the following class to** 📄 `src/mygpt/norm.py` (right after the existing `LayerNorm` class):

```python
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
```

Three things to read off:

- **One parameter, not two.** `self.weight` is a vector of `C` ones at init; there is no `self.bias`. The half-parameter saving compounds: a 4-block transformer has 9 norm instances (2 per block + 1 final), so for `C = 64` we save `9 × 64 = 576` parameters.
- **`x.pow(2).mean(...)`** is the *mean of squared values*; its square root is the **root mean square**. For a centred vector ($\mu = 0$) the RMS equals the standard deviation; in general it equals $\sqrt{\sigma^2 + \mu^2}$. RMSNorm tolerates non-centred input by design.
- **No `unbiased=False` argument** like LayerNorm needed. We compute the mean over all `C` elements; there is no Bessel-correction question because there is no variance computation.

---

## 24.4 The norm-selector helper

We want `GPT` and `TransformerBlock` to be able to use either norm without duplicating their constructors. A small factory function does the dispatch.

**Append the following helper to** 📄 `src/mygpt/norm.py` (right after `RMSNorm`):

```python
def _make_norm(embed_dim: int, norm_type: str) -> nn.Module:
    """Norm-class selector. `norm_type` is 'layer' or 'rms'."""
    if norm_type == "layer":
        return LayerNorm(embed_dim)
    if norm_type == "rms":
        return RMSNorm(embed_dim)
    raise ValueError(f"unknown norm_type: {norm_type!r} (expected 'layer' or 'rms')")
```

The leading underscore signals "internal helper" — students should pass `norm_type` to `GPT()` directly, not call `_make_norm` themselves.

---

## 24.5 Threading `norm_type` through the model

`TransformerBlock` and `GPT` both currently hard-code `LayerNorm`. We change them to accept a `norm_type` parameter and dispatch via `_make_norm`. Default `"layer"` so existing call sites work unchanged.

**Replace `TransformerBlock` in** 📄 `src/mygpt/block.py`:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=64, dropout=0.0, norm_type="layer"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.ln1 = _make_norm(embed_dim, norm_type)
        self.mha = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = _make_norm(embed_dim, norm_type)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

(One new constructor parameter, two `_make_norm` calls replacing direct `LayerNorm` instantiation, and a stored `self.norm_type` for introspection. Forward is unchanged.)

**Replace `GPT.__init__` in** 📄 `src/mygpt/model.py`:

```python
class GPT(nn.Module):
    """Full GPT-2-style decoder-only transformer with weight-tied head."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len=64, dropout=0.0, norm_type="layer"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.norm_type = norm_type

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.ln_f = _make_norm(embed_dim, norm_type)
```

(Two new lines: `self.norm_type = norm_type` and the `norm_type` argument added to `TransformerBlock(...)`. The final norm `ln_f` switches from direct `LayerNorm(embed_dim)` to `_make_norm(embed_dim, norm_type)`. The `forward` method is unchanged from Ch.13.)

---

## 24.6 Checkpoint format adds `norm_type`

A model trained with RMSNorm has a *different* set of parameters from one trained with LayerNorm — different shapes (no bias tensors), different state-dict keys. The checkpoint must record which norm was used, or `load_checkpoint` will try to construct the wrong architecture.

We add one new field, `norm_type`, to the saved config. **Crucially**, when loading we read it with `.get(..., "layer")` — pre-Ch.24 checkpoints don't have this field, and we want them to default to `LayerNorm` (their original behavior).

**Replace `save_checkpoint` in** 📄 `src/mygpt/checkpoint.py`:

```python
def save_checkpoint(model: "GPT", tokenizer: "CharTokenizer", path: str) -> None:
    """Bundle model weights, tokenizer, and architecture into one .ckpt file."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_chars": tokenizer.chars,
            "config": {
                "vocab_size":  model.vocab_size,
                "embed_dim":   model.embed_dim,
                "num_heads":   model.num_heads,
                "num_layers":  model.num_layers,
                "max_seq_len": model.max_seq_len,
                "norm_type":   getattr(model, "norm_type", "layer"),
            },
        },
        path,
    )
```

The `getattr(..., "layer")` is defensive — a model created in Ch.23 code wouldn't have a `norm_type` attribute; we fall back to `"layer"`.

**Replace `load_checkpoint` in** 📄 `src/mygpt/checkpoint.py`:

```python
def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`.

    Always loads to CPU; the caller is responsible for `.to(device)` afterwards.
    Pre-Ch.24 checkpoints have no `norm_type` field; they default to ``"layer"``
    so old `.ckpt` files continue to load correctly.
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer
```

The `config.get("norm_type", "layer")` is the backward-compatibility hinge. **A Chapter 18 `shakespeare.ckpt` will load without modification, because its config has no `norm_type` key, and we default it to `"layer"`.**

---

## 24.7 The `--norm` CLI flag

Wire it through `_train_command`. Two edits: the `GPT(...)` call now passes `norm_type=args.norm`, and the print block adds a `norm:` line (matching the `device:` and `precision:` lines already there).

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
    ).to(device)
```

**And the print block (right after `precision:`):**

```python
    print(f"device:       {device}")
    print(f"precision:    {args.precision}")
    print(f"norm:         {args.norm}")
    print(f"corpus chars: {len(text):,}")
    # … rest unchanged
```

**In `main`'s argparse setup, add to `p_train`** (right after the `--max-grad-norm` block, before `set_defaults(...)`):

```python
    p_train.add_argument(
        "--norm",
        choices=["layer", "rms"],
        default="layer",
        help="Normalisation: 'layer' (default; LayerNorm, Ch.10) or 'rms' (RMSNorm, Llama default).",
    )
```

---

## 24.8 Backward-compat: defaults still reproduce Ch.21

Sanity check: `mygpt train` with the default `--norm layer` must still produce the Ch.21 / Ch.20 / Ch.19 / Ch.17 default loss curve. If it didn't, we'd have changed semantics, not just added a feature.

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output sh-layer.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
norm:         layer
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

saved checkpoint to sh-layer.ckpt
```

The new `norm: layer` line appears in the header but the loss values are identical to every previous chapter's default run. With every flag at its default, the loop is the same loop. Backward-compat preserved.

---

## 24.9 The RMSNorm run

Now the same training with `--norm rms`:

```bash
uv run mygpt train tinyshakespeare.txt --device mps --norm rms --output sh-rms.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
norm:         rms
corpus chars: 1,115,394
train chars:  1,115,394
vocab_size:   65
params:       206,720
steps:        2000
schedule:     constant (warmup=0)
max_grad_norm:0.0
step     1: loss = 41.2164
step   500: loss = 2.5935
step  1000: loss = 2.3517
step  1500: loss = 2.1689
step  2000: loss = 2.0752

saved checkpoint to sh-rms.ckpt
```

Two things to read off:

**1. Parameter count drops from 207,296 to 206,720** — exactly **576 fewer parameters**. Where did they go? RMSNorm has only `embed_dim` parameters per instance (the gain), versus LayerNorm's `2 × embed_dim` (gain + bias). The difference is `embed_dim` per norm instance. Our model has 9 norm instances (2 per `TransformerBlock` × 4 blocks + 1 final `ln_f`), so we save `9 × 64 = 576` parameters.

**2. Loss curve is *barely* different from LayerNorm's** — within ~0.5% at every step, and slightly *lower* at the final step (2.0752 vs 2.0785). The difference is dominated by initialisation noise, not by an architectural advantage. RMSNorm is **not better** than LayerNorm at this scale; it is **comparable, slightly cheaper**. That is exactly the empirical result Zhang & Sennrich reported in 2019 and the basis on which Llama adopted it.

---

## 24.10 Backward-compat: pre-Ch.24 checkpoint still loads

The `load_checkpoint` `config.get("norm_type", "layer")` default means any `.ckpt` file produced by Ch.18 / 19 / 20 / 21 / 22 / 23 still works. Sanity-check by generating from the LayerNorm checkpoint we just saved:

```bash
uv run mygpt generate --checkpoint sh-layer.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output (matches Ch.17 §17.6 byte-for-byte):**

```text
device: cpu

ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

Now the RMSNorm checkpoint:

```bash
uv run mygpt generate --checkpoint sh-rms.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output (RMSNorm produces a different sample — same architecture topology, different parameters because of the missing bias terms):**

```text
device: cpu

ROMEO:
Thy momed haveseltered ad neards way.
ISele fent hourese his therere his herins ore sese. I him
We athor to siely deall: my art, tronen my betyod wely stone.

FRUPENO:
My your sarin shus the shere wa
```

The first 14 characters (`ROMEO:\nThy momed`) match the LayerNorm sample exactly because at `--top-k 10` and very low entropy the most-likely next token wins regardless of small parameter differences. The two samples diverge once the distribution flattens enough to let the per-device RNG (Ch.19) and the per-parameter weights (LayerNorm vs RMSNorm) disagree.

---

## 24.11 Experiments

1. **Inspect the saved config.** `python -c 'import torch; print(torch.load("sh-rms.ckpt", map_location="cpu")["config"])'`. The output includes `'norm_type': 'rms'`. Try the same on `sh-layer.ckpt`; it shows `'norm_type': 'layer'`. Now try an **older** `.ckpt` from Ch.18 — its config dict has no `norm_type` key at all, but `load_checkpoint` still works because of the `.get(..., "layer")` default.

2. **A wrong norm at load time.** Suppose you forced `norm_type="rms"` when loading a LayerNorm checkpoint. The `model.load_state_dict(ckpt["model_state_dict"])` call would raise an error — the LayerNorm checkpoint contains `*.bias` keys for every norm, but the RMSNorm-built model has no `*.bias` parameters to receive them. Try it: edit `load_checkpoint` to hard-code `norm_type="rms"`, then run generate on `sh-layer.ckpt`. Compare the error message PyTorch produces to your prediction.

3. **Manual RMSNorm.** Open a Python REPL and verify the formula:
   ```python
   import torch
   x = torch.tensor([3.0, 4.0])     # rms = sqrt((9+16)/2) = sqrt(12.5) = 3.536
   from mygpt import RMSNorm
   norm = RMSNorm(embed_dim=2)       # weight = [1.0, 1.0]
   y = norm(x)                       # ≈ x / 3.536 = [0.849, 1.131]
   print(y)
   ```
   Confirm by hand.

4. **Param-count formula.** A `GPT(V, C, h, N, L=64)` with LayerNorm has `V*C + L*C + N*(12 C² + 9 C) + 2C` parameters (Ch.12 §12.5 formula). Argue from §24.2 that the RMSNorm version has `V*C + L*C + N*(12 C² + 8 C) + C` parameters — `(N + 1) * C` fewer. For our `V=65, C=64, N=4` model, that is `5 * 64 = 320`… wait, let me recount. Each block has 2 norms (so $2 \times C$ saved, $\times N = 4$ blocks = $8C$), plus the final `ln_f` ($C$ saved), so total saved = $9C = 576$ for $C=64$. Confirm against the §24.9 numbers (207,296 → 206,720 = 576 saved). ✓

---

## 24.12 Exercises

1. **Why no bias?** Argue that adding a bias to RMSNorm would not break it functionally (the model can still learn) but is *redundant* given that every linear layer downstream has its own bias. (Hint: $\text{RMSNorm}(\mathbf{x}) \cdot \mathbf{w} + \mathbf{b}$ can be absorbed into the linear layer that follows it, by noting that `Linear(W, B)(input + b) = Linear(W, B + W b)(input)`. The bias of the norm and the bias of the next linear are *the same parameter* up to where you put it.)

2. **Eps placement.** RMSNorm's eps is added inside the `sqrt`: $\sqrt{\text{mean}(x^2) + \varepsilon}$. LayerNorm's eps is the same (inside the sqrt, after the variance). Argue that placing eps *outside* the sqrt would also work numerically but would shrink the gradient near zero — sketch the gradient of $1 / (\sqrt{u} + \varepsilon)$ vs $1 / \sqrt{u + \varepsilon}$ near $u = 0$.

3. **Why is `mean(x²)` not `mean(x)²`?** LayerNorm subtracts the mean *before* squaring; RMSNorm computes `mean(x²)` *directly* — without subtracting the mean. Argue these are the same iff $\mu(\mathbf{x}) = 0$, and they differ by exactly $\mu(\mathbf{x})^2$ when $\mu \neq 0$. (This is the same identity you saw in Ch.10's variance derivation: $\sigma^2 = \mathbb{E}[x^2] - \mathbb{E}[x]^2$.)

4. **Initialisation: gain=1, no bias.** Both `LayerNorm` and `RMSNorm` initialise `weight` to ones and (for LayerNorm) `bias` to zeros. Argue that this initialisation makes both a *no-op at the start of training* on a centred unit-variance input, so the network's first forward pass is well-defined. (Hint: with $\mathbf{x}$ unit-variance and zero mean, both norms compute approximately $\mathbf{x} / 1 \cdot \mathbf{1} + \mathbf{0} = \mathbf{x}$.)

---

## 24.13 What's next

The next chapter, **Chapter 25 — RoPE: rotary position embeddings**, replaces the *learned* position embedding (Ch.12) with a *parameter-free* rotation of the query and key vectors. RoPE has the structural property that the §15.9 "untrained position embedding" failure mode disappears — the model can generate at any position, including positions beyond what it was trained on, because there are no learned position parameters at all.

> **Looking ahead — what to remember from this chapter**
>
> 1. RMSNorm is LayerNorm minus the mean subtraction and minus the bias.
> 2. Half the parameters per norm instance, half the forward-pass arithmetic, comparable training quality.
> 3. `--norm rms` opt-in; `--norm layer` (default) bit-reproduces every prior chapter's run.
> 4. Checkpoints carry their `norm_type` in the config; pre-Ch.24 checkpoints default to `"layer"` on load, so old `.ckpt` files keep working.

On to [Chapter 25 — RoPE: rotary position embeddings](25_rope.md).
