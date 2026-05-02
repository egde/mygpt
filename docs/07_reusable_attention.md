---
title: 7. A reusable attention module
nav_order: 8
parent: Part I — LLM Fundamentals
---

# Chapter 7 — A reusable attention module

The `SingleHeadAttention` we wrote in Chapter 6 *works*, but it has two engineering problems we will hit the moment we try to compose it into a real model. This chapter fixes both — without changing the mathematics. By the end you will have:

- replaced the freshly-allocated causal mask with a **buffer** registered on the module — faster, GPU-aware, and saved with the model;
- introduced **dropout** on both the attention weights and the output projection — the regularisation every real transformer uses;
- understood `register_buffer` and `nn.Module.train()` / `.eval()`, the two pieces of `nn.Module` machinery you will need from Chapter 11 onward.

The chapter ends with the *same* `SingleHeadAttention` class you already imported from `mygpt`, just better. Numerically, with `dropout=0`, the new version produces byte-for-byte identical output to the Chapter 6 version.

There is no maths in this chapter beyond what you already know.

---

## 7.1 What's wrong with §6.9

Re-read the `forward` method we wrote in Chapter 6:

```python
def forward(self, x):
    B, T, C = x.shape
    Q = self.W_Q(x); K = self.W_K(x); V = self.W_V(x)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    return self.W_O(out)
```

Two problems:

1. **The causal mask is allocated inside `forward`.** Every time you call the module, PyTorch creates a brand-new $(T, T)$ tensor full of $-\infty$ and zeros. For training that means *thousands* of allocations a second. Worse: the mask is allocated on the CPU, so if `x` is on a GPU, the addition triggers an implicit (and slow) device transfer. And when you save and re-load the model, the mask is recreated rather than restored — fine for a function of $T$, but a sign that the code is not following PyTorch's conventions.
2. **There is no regularisation.** Real transformers use **dropout** — randomly zeroing a fraction of intermediate activations during training — to prevent the model from memorising the training set. GPT-2 uses dropout in three places per attention head: on the attention weights, on the output projection, and on the embedding sum. We add the first two in this chapter.

Both fixes are small and standard PyTorch idioms.

---

## 7.2 Setup

This chapter assumes you finished Chapter 6 — `mygpt/` exists with `VOCAB`, `to_ids`, `set_seed`, `TokenEmbedding`, and `SingleHeadAttention`.

If you skipped Chapter 6, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 6 ending state from `docs/_state_after_ch06.md` (or just paste the version from §6.9 into your own `src/mygpt/__init__.py`).

You are ready.

---

## 7.3 The causal mask should be a buffer

A **buffer** is a tensor that belongs to a `Module` but is not a parameter — it does not receive gradients, but it does:

- move to GPU automatically when you call `module.to("cuda")`,
- save and restore with `torch.save` / `torch.load`,
- show up in `module.named_buffers()` for inspection,
- *not* be passed to the optimiser (so it is not updated during training).

The causal mask is exactly this kind of object: a fixed tensor we want sitting alongside the parameters, riding along with the module wherever it goes, but never being updated. The right tool is `register_buffer`.

The pattern: in `__init__`, build the mask **once** for the largest $T$ we might ever see (call it `max_seq_len`), and register it. In `forward`, slice it down to the actual $T$ of the current batch.

```python
# In __init__:
mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
self.register_buffer("causal_mask", mask)

# In forward:
B, T, C = x.shape
scores = scores + self.causal_mask[:T, :T]
```

Two things to notice:

- We allocate the mask **once**, when the module is constructed. Subsequent forward passes do `self.causal_mask[:T, :T]`, which is a view (zero allocation) into the existing buffer.
- `register_buffer` takes a *string name* and a tensor. The tensor becomes accessible as `self.<name>` after registration — so the line `scores + self.causal_mask[:T, :T]` works.

The `max_seq_len` is a hyperparameter of the module. We will call this the **context window** throughout the rest of the tutorial; `max_seq_len` is the code-level identifier, "context window" is the prose name for the same thing. Set it to the longest sequence length you intend to support; if you call the module on longer inputs it will index out of bounds. Real GPT models set this to 1024, 2048, or higher; for our running example we use 64 — comfortably more than the four tokens of `"I love AI !"`.

(Note: the *context window* is the architecture's maximum. The **trained context length** — the `seq_len` actually used during training — can be smaller, and §15.9 will show what happens if you generate past it.)

---

## 7.4 `register_buffer` in detail

A small experiment to internalise the difference between buffers and parameters.

**Save the following to** 📄 `experiments/16_buffer_vs_param.py`:

```python
"""Experiment 16 — Buffers vs parameters: what each one does.

Builds a tiny module with one of each and inspects how PyTorch tracks them.
"""

import torch
import torch.nn as nn


class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))     # learnable
        self.register_buffer("offset", torch.ones(2))  # not learnable, but module-owned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight + self.offset + x


def main() -> None:
    m = Demo()

    print("named_parameters:")
    for n, p in m.named_parameters():
        print(f"  {n}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
    print()

    print("named_buffers:")
    for n, b in m.named_buffers():
        print(f"  {n}: shape={tuple(b.shape)}, requires_grad={b.requires_grad}")
    print()

    # The optimiser sees only parameters
    n_optim_params = len(list(m.parameters()))
    print(f"len(list(m.parameters())) = {n_optim_params}  # only the learnable bit")
    n_state = len(m.state_dict())
    print(f"len(m.state_dict())       = {n_state}  # parameters AND buffers — both saved")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/16_buffer_vs_param.py
```

**Expected output:**

```text
named_parameters:
  weight: shape=(2,), requires_grad=True

named_buffers:
  offset: shape=(2,), requires_grad=False

len(list(m.parameters())) = 1  # only the learnable bit
len(m.state_dict())       = 2  # parameters AND buffers — both saved
```

The buffer is invisible to the optimiser (`m.parameters()` returns only `weight`) but visible to `state_dict()` (which is what `torch.save` writes). That is exactly what we want for the causal mask.

---

## 7.5 Dropout

Dropout is a regularisation operation introduced by Srivastava et al. (2014). During **training**, it randomly sets a fraction $p$ of its inputs to zero and scales the rest by $1/(1-p)$ so the *expected* sum is unchanged. During **inference** (when `module.eval()` has been called), it is the identity — no zeros, no scaling.

The behaviour switch is controlled by `nn.Module.training`, a boolean attribute every module has. You toggle it with `module.train()` and `module.eval()`. PyTorch's `nn.Dropout` reads `self.training` on every call to decide what to do.

Why drop randomly during training? The standard explanation: the model cannot rely on any specific subset of features being available, so it has to learn redundant, generalisable representations. The intuition is fuzzy; the empirical effect (better validation loss) is robust.

A small experiment to see dropout's two modes side by side.

**Save the following to** 📄 `experiments/17_dropout_modes.py`:

```python
"""Experiment 17 — Dropout in train mode vs eval mode."""

import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(42)
    drop = nn.Dropout(p=0.5)
    x = torch.ones(2, 4)
    print(f"x = {x}")
    print()

    # Train mode (default after construction)
    drop.train()
    out_train = drop(x)
    print("dropout(x) in train mode (random zeros, others scaled by 1/(1-0.5) = 2):")
    print(out_train)
    print()

    # Eval mode
    drop.eval()
    out_eval = drop(x)
    print("dropout(x) in eval mode (identity — same as input):")
    print(out_eval)


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/17_dropout_modes.py
```

**Expected output:**

```text
x = tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])

dropout(x) in train mode (random zeros, others scaled by 1/(1-0.5) = 2):
tensor([[2., 2., 2., 2.],
        [0., 2., 0., 0.]])

dropout(x) in eval mode (identity — same as input):
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])
```

The randomness in train mode is itself seeded — the same `torch.manual_seed` produces the same dropout pattern. We use `dropout=0` in this chapter (so the math matches Chapter 6 exactly) but the layers are still there, ready to do something the moment we set `dropout > 0` in Chapter 11 onwards.

---

## 7.6 Refactoring `SingleHeadAttention`

We now rewrite `mygpt.SingleHeadAttention` to use both fixes. The signature gains two new arguments — `max_seq_len` (default 64) and `dropout` (default 0.0). With the defaults, the module behaves identically to Chapter 6.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 7 the SingleHeadAttention module uses register_buffer for
the causal mask (allocated once, in __init__) and adds dropout layers
on both the attention weights and the output projection.

With dropout=0 (the default), forward output is byte-for-byte identical
to the Chapter 6 version.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids."""
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's CPU random number generator."""
    torch.manual_seed(seed)


class TokenEmbedding(nn.Module):
    """Map a tensor of integer token ids to a tensor of dense embedding vectors."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class SingleHeadAttention(nn.Module):
    """Single-head causal self-attention with a registered causal mask and dropout.

    Inputs:
        x: tensor of shape (B, T, embed_dim).

    Outputs:
        tensor of shape (B, T, embed_dim).

    Constructor arguments:
        embed_dim:    width of the input/output embedding axis (C).
        head_dim:     width of the head's internal Q/K/V (d_h). For single-head
                      we set head_dim = embed_dim; multi-head (Chapter 8) sets
                      head_dim < embed_dim and runs several heads in parallel.
        max_seq_len:  the largest sequence length the module is willing to
                      process. The causal mask is allocated once with this size
                      in __init__ and sliced down in forward.
        dropout:      probability of zeroing each entry in the attention weights
                      and in the output projection. Default 0.0 reproduces the
                      Chapter 6 behaviour exactly.
    """

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

        # Causal mask: allocated once, sliced per-call. Buffer so it moves
        # with the module (to GPU, into checkpoints) without ever receiving
        # gradients.
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

        Q = self.W_Q(x)                                              # (B, T, head_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)                          # (B, T, T)
        weights = self.attn_drop(weights)
        out = weights @ V                                             # (B, T, head_dim)
        return self.out_drop(self.W_O(out))                           # (B, T, embed_dim)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(vocab_size=V, embed_dim=C)
    attn = SingleHeadAttention(embed_dim=C, head_dim=C, max_seq_len=64, dropout=0.0)
    attn.eval()  # no randomness for the dropout layers

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = attn(x)

    print(f"\nToken ids shape:           {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):  {tuple(x.shape)}")
    print(f"Attention output (B, T, C): {tuple(out.shape)}")

    n_te = sum(p.numel() for p in te.parameters())
    n_attn_params = sum(p.numel() for p in attn.parameters())
    n_attn_buffers = sum(b.numel() for b in attn.buffers())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"SingleHeadAttention parameters:   {n_attn_params}")
    print(f"SingleHeadAttention buffers:      {n_attn_buffers}  (causal_mask, not trained)")
    print(f"Total parameters:                 {n_te + n_attn_params}")
```

Run the package entry-point:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4

Token ids shape:           (1, 4)
Embedded shape (B, T, C):  (1, 4, 4)
Attention output (B, T, C): (1, 4, 4)

TokenEmbedding parameters:        16
SingleHeadAttention parameters:   64
SingleHeadAttention buffers:      4096  (causal_mask, not trained)
Total parameters:                 80
```

Three things to notice:

- **The 64 parameters are unchanged.** The refactor did not introduce any new learnable weights: `nn.Dropout` has none, and `register_buffer` wraps a fixed tensor.
- **The buffer is 4096 = 64 × 64 entries.** Allocated once, in `__init__`, regardless of how many forward passes the module sees.
- **Total parameter count stays at 80**, matching the Chapter 6 ending state — no model surgery, just engineering.

---

## 7.7 Verifying behaviour matches Chapter 6 with `dropout=0`

Engineering refactors must not change the output. With `dropout=0` (and `eval()` mode for safety), the new module should produce byte-for-byte the same output as the Chapter 6 version on the same input.

**Save the following to** 📄 `experiments/18_refactor_equivalence.py`:

```python
"""Experiment 18 — The refactored SingleHeadAttention with dropout=0 produces
identical output to a hand-coded Chapter-6 version.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mygpt import SingleHeadAttention, set_seed


class Ch6SingleHeadAttention(nn.Module):
    """The Chapter-6 version, locally for comparison."""

    def __init__(self, embed_dim: int, head_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_O = nn.Linear(head_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.W_Q(x); K = self.W_K(x); V = self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return self.W_O(out)


def main() -> None:
    # Build both with the SAME seed-0 init order so their W_Q, W_K, W_V, W_O all match.
    set_seed(0)
    old = Ch6SingleHeadAttention(embed_dim=4, head_dim=4)

    set_seed(0)
    new = SingleHeadAttention(embed_dim=4, head_dim=4, max_seq_len=64, dropout=0.0)
    new.eval()

    # Same input
    set_seed(42)
    x = torch.randn(1, 4, 4)

    with torch.no_grad():
        out_old = old(x)
        out_new = new(x)

    print("OLD (Chapter 6):")
    print(out_old)
    print()
    print("NEW (Chapter 7):")
    print(out_new)
    print()
    print(f"identical:    {torch.equal(out_old, out_new)}")
    print(f"max abs diff: {(out_old - out_new).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/18_refactor_equivalence.py
```

**Expected output:**

```text
OLD (Chapter 6):
tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
         [-0.1713,  0.5772,  0.2182, -0.4687],
         [-0.3211,  0.5328,  0.1321, -0.3144],
         [-0.1588,  0.2404,  0.0839, -0.0570]]])

NEW (Chapter 7):
tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
         [-0.1713,  0.5772,  0.2182, -0.4687],
         [-0.3211,  0.5328,  0.1321, -0.3144],
         [-0.1588,  0.2404,  0.0839, -0.0570]]])

identical:    True
max abs diff: 0.000e+00
```

Byte-for-byte identical. The refactor is mathematically a no-op on the dropout=0 path; the changes are entirely about how the module organises its state.

---

## 7.8 Experiments

1. **`max_seq_len` is enforced.** In `experiments/18_refactor_equivalence.py`, change `max_seq_len=64` to `max_seq_len=2`, then change the input length to `x = torch.randn(1, 4, 4)`. Re-run. The new module raises `ValueError: input length T=4 exceeds max_seq_len=2`. The old (Chapter 6) module would have happily allocated a 4×4 mask and produced output. Your call which behaviour is safer — we picked the strict one.
2. **Dropout actually fires in train mode.** In `experiments/18_refactor_equivalence.py`, construct the new attention with `dropout=0.5` and *do not* call `.eval()`. Re-run. The output now differs from the old one, and successive runs at the same seed produce different outputs (every forward draws a fresh dropout mask). Confirm row-sums-to-1 of `weights` is **broken** by attention dropout — that's by design; the renormalisation happens implicitly via the $1/(1-p)$ scale.
3. **Buffer follows the module.** In a Python session: `from mygpt import SingleHeadAttention; m = SingleHeadAttention(4, 4)`; print `m.causal_mask.device`. Should be `cpu`. If you have a CUDA-capable machine, run `m.to("cuda"); print(m.causal_mask.device)` — it should be `cuda:0`. The mask moved with the module without you having to touch it. (A non-buffer tensor would not.)
4. **State-dict round-trip preserves the buffer.** In a Python session: `m1 = SingleHeadAttention(4, 4); torch.save(m1.state_dict(), "/tmp/sha.pt"); m2 = SingleHeadAttention(4, 4); m2.load_state_dict(torch.load("/tmp/sha.pt")); print(torch.equal(m1.causal_mask, m2.causal_mask))`. Should print `True`. The buffer is preserved across save/load just like a parameter.

After each experiment, restore the file you changed before moving on.

---

## 7.9 Exercises

1. **Why `register_buffer` over a plain attribute?** What goes wrong if you replace `self.register_buffer("causal_mask", mask)` with `self.causal_mask = mask`? Try it: build the module, call `m.to("cuda")`, and observe the device of `self.causal_mask`. (Plain attributes do not move; you would silently mix a CPU mask with GPU activations and the addition would crash or fall back to a slow path.)
2. **Why `bias=False`?** All four `nn.Linear` layers still use `bias=False`. Recall the §6.12 hint: softmax is invariant to adding the same constant to every input. Argue specifically why a per-output-channel bias on $Q$ has no effect on the *attention weights* (every column of $QK^\top$ shifts by the same amount, which softmax cancels).
3. **Two dropout layers — could it be one?** We use `attn_drop` after softmax and `out_drop` after `W_O`. Could we merge them into a single layer at the end? Argue why not, by considering what each is regularising. (Hint: `attn_drop` regularises *which positions* a query attends to; `out_drop` regularises *the channels* of the output vector. Different roles, different effects.)
4. **The `max_seq_len = 64` choice.** What is the memory cost of the buffer in bytes, for `max_seq_len = 1024` and `float32`? Generalise to a function of `max_seq_len`. (For comparison, GPT-2's full-context attention masks at 1024 tokens occupy 4 MB per layer × 12 layers ≈ 48 MB — small compared to the 124 M model parameters at 4 bytes each = 496 MB.)

---

## 7.10 What's next

`SingleHeadAttention` now has the engineering properties a real model needs: a registered causal mask, a configurable maximum sequence length, and dropout layers ready to regularise the model in Chapter 14. The maths is unchanged.

In Chapter 8 we generalise to **multi-head attention**: instead of one head with $d_h = C$, run $h$ heads in parallel with $d_h = C / h$. The forward shape changes from $(B, T, C) \to (B, T, C)$ to $(B, T, C) \to (B, h, T, C/h)$ → concatenate → $(B, T, C)$. Multi-head attention is what GPT-2 actually uses, and is the centerpiece operation of every modern transformer.

> **Looking ahead — what to remember from this chapter**
>
> 1. `register_buffer(name, tensor)` adds a non-parameter tensor to a module — moves with `.to()`, persists in `state_dict()`, ignored by the optimiser.
> 2. `nn.Dropout(p)` zeros a fraction $p$ of inputs in train mode and is the identity in eval mode. Toggle with `module.train()` / `.eval()`.
> 3. The refactored `SingleHeadAttention` adds `max_seq_len` and `dropout` arguments; with `dropout=0` it produces output byte-for-byte identical to the Chapter 6 version.
> 4. The total parameter count is unchanged. Buffers and dropout layers are not parameters.

On to [Chapter 8 — Multi-head attention](08_multi_head_attention.md) *(coming soon)*.
