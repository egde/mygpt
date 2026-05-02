---
title: 8. Multi-head attention
nav_order: 9
parent: Part I — LLM Fundamentals
---

# Chapter 8 — Multi-head attention

In Chapter 6 we built a single attention "head" — one $W_Q$, one $W_K$, one $W_V$, all working in $d_h = C$ dimensions. In Chapter 7 we polished it. In this chapter we do the move that defines modern transformers: run **several heads in parallel** with $d_h = C / h$, then concatenate their outputs. (This requires `embed_dim` to be divisible by `num_heads`; the `MultiHeadAttention` constructor raises `ValueError` otherwise.)

By the end you will have:

- understood **why** multi-head attention helps (each head can specialise),
- built `mygpt.MultiHeadAttention(embed_dim, num_heads)` with a clean batched implementation that runs all $h$ heads in one matmul via tensor reshape,
- verified that with `num_heads = 1`, the new module produces output **byte-for-byte identical** to `SingleHeadAttention(embed_dim=C, head_dim=C)` from Chapter 7,
- met the four-axis tensor shape `(B, num_heads, T, head_dim)` that every multi-head transformer uses internally.

There is no new mathematics: the operation inside each head is exactly what Chapter 6 already defined. What is new is *how to run $h$ heads in parallel without a Python loop*.

---

## 8.1 Why multi-head?

A single head learns one function $\mathbb{R}^{T \times C} \to \mathbb{R}^{T \times C}$. To represent that function it has $4 C^2$ parameters (the four $C \times C$ matrices $W_Q, W_K, W_V, W_O$) and one set of attention scores per pair of positions. *One* set of scores means *one* relationship the head can attend on. If you want the model to track, say, both syntactic agreement (subject ↔ verb) and topical coherence (`"AI"` ↔ `"love"`), a single head has to compromise on a representation that captures both.

Multi-head attention lets the model **specialise**. Run $h$ heads in parallel; each gets its own scaled-dot-product attention but with smaller $d_h = C / h$ dimensions. Each head can attend to different relationships:

- Head 0 might learn to attend "previous noun".
- Head 1 might learn to attend "matching subject pronoun".
- Head 2 might learn to attend "topic word".
- And so on.

After all $h$ heads have produced their outputs, we **concatenate** them along the channel axis (giving back the original $C$ dimensions: $h \cdot d_h = C$) and apply a final linear projection. The model decides, via its parameters, what each head specialises in.

Two more facts worth stating up front:

- **Parameter count is unchanged.** With $h$ heads and $d_h = C/h$, each of $W_Q, W_K, W_V$ is $C \times C$ overall, exactly like in single-head — the only thing that changes is *how the columns are split between heads*. The output projection $W_O$ is also $C \times C$. Total: $4 C^2$ parameters either way.
- **The math inside each head is unchanged.** Causal mask, scaled dot-product, softmax, multiply by $V$ — all the same operations Chapter 6 defined. The novelty is purely how we batch them.

---

## 8.2 Setup

This chapter assumes you finished Chapter 7 — `mygpt/` exists with the refactored `SingleHeadAttention` (the one with `register_buffer` and dropout layers).

If you skipped Chapter 7, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 7 ending state from `docs/_state_after_ch07.md` (the `SingleHeadAttention(embed_dim, head_dim, max_seq_len=64, dropout=0.0)` version).

You are ready.

---

## 8.3 The shape transformation: $(B, T, C) \to (B, h, T, d_h)$

The single tensor shape that makes multi-head attention work is

$$
(B, T, C) \;\rightarrow\; (B, h, T, d_h), \qquad C = h \cdot d_h.
$$

We *split* the $C$-axis into $h$ groups of $d_h$ features each, and add a new "head" axis of size $h$ in front of the time axis. The reshape is purely a view — no data is copied — and after it, every subsequent operation can be done with **one** batched matmul instead of a Python loop over heads.

The PyTorch idiom is two operations: `view` to split the channel axis, then `transpose` to bring the head axis next to the batch axis.

```python
# x: (B, T, C)
x = x.view(B, T, num_heads, head_dim)   # (B, T, h, d_h)
x = x.transpose(1, 2)                    # (B, h, T, d_h)
```

Read the transpose carefully: it swaps axes 1 and 2 (which are `T` and `h` after the view). The result is $(B, h, T, d_h)$.

This is the *only* reshape we need. We apply it to $Q$, $K$, $V$ separately, then run the same scaled-dot-product attention as Chapter 6 — but now the leading axes are $(B, h)$ instead of just $(B,)$, so PyTorch's batched matmul (`@`) automatically broadcasts and gives us $h$ independent attention computations in one shot.

---

## 8.4 Per-head attention runs in parallel via batched matmul

With $Q, K, V$ each of shape $(B, h, T, d_h)$:

$$
\text{scores} \;=\; \frac{Q K^\top}{\sqrt{d_h}} \;\in\; \mathbb{R}^{(B, h, T, T)}.
$$

`Q @ K.transpose(-2, -1)` operates on the last two axes; the leading $(B, h)$ are batch axes that PyTorch broadcasts over. Each of the $B \times h$ pairs gets its own $(T, T)$ scores matrix.

Causal masking is unchanged — we add the same $(T, T)$ mask, which broadcasts across the $(B, h)$ batch axes:

```python
scores = scores + self.causal_mask[:T, :T]  # broadcast over (B, h)
weights = F.softmax(scores, dim=-1)         # (B, h, T, T)
out = weights @ V                           # (B, h, T, d_h)
```

`weights @ V` is again a batched matmul: each of the $B \times h$ pairs of $(T, T)$ weights and $(T, d_h)$ values gives a $(T, d_h)$ output. Total output shape: $(B, h, T, d_h)$.

To get back to the conventional $(B, T, C)$ form, we **undo** the reshape and concatenate the heads along the channel axis:

```python
out = out.transpose(1, 2)                     # (B, T, h, d_h)
out = out.contiguous().view(B, T, C)          # (B, T, C), C = h * d_h
```

The `.contiguous()` is necessary because `transpose` produces a non-contiguous view, and `view` requires contiguous memory. After this, `out` looks exactly like the input shape — a single $(B, T, C)$ tensor — and a final $C \times C$ projection $W_O$ gives the module's output.

---

## 8.5 Why a single $C \times C$ projection for each of $W_Q, W_K, W_V$ — and not one per head?

A natural first design is "give each head its own $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ of shape $(C, d_h)$, run them independently, then concatenate". That is mathematically the same as what we are about to do: a single $W_Q$ of shape $(C, C)$ whose columns are conceptually grouped into $h$ blocks of $d_h$ columns each.

The single-projection version has two practical advantages:

- **One matmul instead of $h$.** Even on a GPU, $h$ separate small matmuls are slower than one big matmul because of kernel launch overhead.
- **One parameter tensor per role.** `self.W_Q` is just an `nn.Linear(C, C)`. No `nn.ModuleList`, no per-head bookkeeping in `__init__` or `state_dict`.

The cost: a tiny bit of conceptual unease, because "$W_Q$ has shape $(C, C)$" looks like the single-head version. The trick is the *interpretation* of the columns — after the reshape in §8.4, the first $d_h$ output channels are "head 0's $Q$", the next $d_h$ are "head 1's $Q$", and so on. The matrix is the same; only the axis layout changes.

---

## 8.6 The `MultiHeadAttention` module

Putting it all together — same five-step recipe as Chapter 6, just with a 4-axis tensor in the middle.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 8 the package gains a MultiHeadAttention module that runs
num_heads single-head computations in parallel via tensor reshape, and
combines them through a final output projection.
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
    """Single-head causal self-attention with a registered causal mask and dropout."""

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


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention.

    Inputs:
        x: tensor of shape (B, T, embed_dim).

    Outputs:
        tensor of shape (B, T, embed_dim).

    Constructor arguments:
        embed_dim:    width of the input/output embedding axis (C).
        num_heads:    number of parallel heads (h). Must divide embed_dim.
        max_seq_len:  the largest sequence length the module is willing to
                      process. The causal mask is allocated once with this size
                      in __init__ and sliced down in forward.
        dropout:      probability of zeroing each entry in the attention weights
                      and in the output projection.

    Each head operates in head_dim = embed_dim // num_heads dimensions; the
    heads run in parallel via tensor reshape, and their outputs are
    concatenated along the channel axis before a final embed_dim x embed_dim
    output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
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

        # One C x C projection per role. After the reshape in forward,
        # the first head_dim output channels of W_Q go to head 0, the
        # next head_dim go to head 1, and so on.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        # (B, T, C) -> three (B, T, C) tensors
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split the C axis into (num_heads, head_dim) and move the head axis
        # next to the batch axis: (B, T, C) -> (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention, batched over (B, num_heads).
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, h, T, T)
        scores = scores + self.causal_mask[:T, :T]                   # broadcast (T,T) over (B,h)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V                                            # (B, h, T, head_dim)

        # Undo the reshape, concatenate heads back into the C axis,
        # apply the output projection.
        out = out.transpose(1, 2).contiguous().view(B, T, C)         # (B, T, C)
        return self.out_drop(self.W_O(out))


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    mha = MultiHeadAttention(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0)
    mha.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = mha(x)

    print(f"\nToken ids shape:                {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):       {tuple(x.shape)}")
    print(f"MultiHeadAttention output shape: {tuple(out.shape)}")
    print()
    print(f"num_heads = {mha.num_heads}, head_dim = {mha.head_dim}, embed_dim = {mha.embed_dim}")

    n_te = sum(p.numel() for p in te.parameters())
    n_mha = sum(p.numel() for p in mha.parameters())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"MultiHeadAttention parameters:    {n_mha}")
    print(f"Total parameters:                 {n_te + n_mha}")
```

Run the package entry-point:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4

Token ids shape:                (1, 4)
Embedded shape (B, T, C):       (1, 4, 4)
MultiHeadAttention output shape: (1, 4, 4)

num_heads = 2, head_dim = 2, embed_dim = 4

TokenEmbedding parameters:        16
MultiHeadAttention parameters:    64
Total parameters:                 80
```

Two things to read off:

- **Total parameters: 80** — same as in §7.6. With $C = 4$, going from `SingleHeadAttention(C, head_dim=4)` to `MultiHeadAttention(C, num_heads=2)` does not change the parameter count. Both have $4 C^2 = 64$ parameters in the four $C \times C$ matrices.
- **head_dim = 2** is half the embed_dim, because $C/h = 4/2 = 2$. Each of the two heads operates in 2-D, but the *connectivity* is now factored into two independent attention computations.

---

## 8.7 Verifying $h = 1$ recovers single-head

Sanity check: when `num_heads=1`, `head_dim=embed_dim`, the reshape is a no-op (it inserts a trivial axis of size 1 and then transposes), and the four $C \times C$ matrices play exactly the role $W_Q, W_K, W_V, W_O$ do in `SingleHeadAttention(C, head_dim=C)`. So `MultiHeadAttention(C, 1)` and `SingleHeadAttention(C, C)` should compute the same function.

Built with the same `set_seed(0)` and given the same input, they should produce **byte-for-byte** identical output.

**Save the following to** 📄 `experiments/19_mha_one_head_equivalence.py`:

```python
"""Experiment 19 — MultiHeadAttention(C, num_heads=1) reduces to
SingleHeadAttention(C, head_dim=C).

Builds both with the same seed and same input; confirms torch.equal()
on the two outputs.
"""

import torch

from mygpt import MultiHeadAttention, SingleHeadAttention, set_seed


def main() -> None:
    set_seed(0)
    sha = SingleHeadAttention(embed_dim=4, head_dim=4, max_seq_len=64, dropout=0.0)
    sha.eval()

    set_seed(0)
    mha = MultiHeadAttention(embed_dim=4, num_heads=1, max_seq_len=64, dropout=0.0)
    mha.eval()

    set_seed(42)
    x = torch.randn(1, 4, 4)

    with torch.no_grad():
        out_sha = sha(x)
        out_mha = mha(x)

    print("SingleHeadAttention(4, 4):")
    print(out_sha)
    print()
    print("MultiHeadAttention(4, num_heads=1):")
    print(out_mha)
    print()
    print(f"identical:    {torch.equal(out_sha, out_mha)}")
    print(f"max abs diff: {(out_sha - out_mha).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/19_mha_one_head_equivalence.py
```

**Expected output:**

```text
SingleHeadAttention(4, 4):
tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
         [-0.1713,  0.5772,  0.2182, -0.4687],
         [-0.3211,  0.5328,  0.1321, -0.3144],
         [-0.1588,  0.2404,  0.0839, -0.0570]]])

MultiHeadAttention(4, num_heads=1):
tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
         [-0.1713,  0.5772,  0.2182, -0.4687],
         [-0.3211,  0.5328,  0.1321, -0.3144],
         [-0.1588,  0.2404,  0.0839, -0.0570]]])

identical:    True
max abs diff: 0.000e+00
```

Byte-for-byte identical. The chapter's `MultiHeadAttention` is a strict generalisation of `SingleHeadAttention` — at `num_heads=1` they are mathematically the same module.

(The output matches the §7.7 byte-for-byte equivalence we already confirmed in Chapter 7. So we have a chain `SingleHeadAttention (Ch.6) == SingleHeadAttention (Ch.7) == MultiHeadAttention(num_heads=1)` — three distinct implementations producing identical numbers.)

---

## 8.8 Experiments

1. **Inspect the four-axis attention weights.** In a Python session with the running example, set up `mha = MultiHeadAttention(4, num_heads=2, dropout=0.0); mha.eval(); set_seed(0)`. Build a fresh `(1, 4, 4)` input. Replicate the §6.10 trick of computing the weights manually so you can print the `(B, h, T, T) = (1, 2, 4, 4)` tensor. The two heads should both be lower-triangular but with *different* internal numbers — they are independent computations.
2. **`embed_dim` must be divisible by `num_heads`.** Try `MultiHeadAttention(embed_dim=4, num_heads=3)`. The constructor should raise `ValueError: embed_dim (4) must be divisible by num_heads (3)`. Now try `MultiHeadAttention(embed_dim=6, num_heads=3)` — works, with `head_dim = 2`.
3. **Parameter count is independent of `num_heads`.** Construct `MultiHeadAttention(8, h)` for `h ∈ {1, 2, 4, 8}`. Each should report exactly `4 * 8 * 8 = 256` parameters. The factorisation of `C` into `(h, d_h)` does not change the total.
4. **`max_seq_len` is enforced just like in `SingleHeadAttention`.** Build `MultiHeadAttention(4, 2, max_seq_len=2)`, then call it on `x = torch.randn(1, 4, 4)`. Expected: `ValueError: input length T=4 exceeds max_seq_len=2`.

After each experiment, restore the file you changed before moving on.

---

## 8.9 Exercises

1. **Where does the head axis end up?** In the `forward` method, after `Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)`, $Q$ has shape `(B, num_heads, T, head_dim)`. Why is it useful to put `num_heads` next to the batch axis rather than at the end? (Hint: think about which axes `Q @ K.transpose(-2, -1)` operates on.)
2. **Why `.contiguous()` before `.view()`?** Re-read the line `out = out.transpose(1, 2).contiguous().view(B, T, C)`. What error would PyTorch raise if you removed `.contiguous()`? (Hint: `.transpose` produces a view with non-standard strides; `.view` can only be applied to contiguous memory.)
3. **The output projection is square.** $W_O$ is `(embed_dim, embed_dim)`, mapping from concatenated head outputs back to the embedding dimension. Why do we need it at all — couldn't we skip it and use the concatenated heads directly? (Hint: without $W_O$, the model has no way to *mix* features that head 0 produced with features that head 1 produced. $W_O$ provides exactly that cross-head linear combination.)
4. **GPT-2 small uses what?** GPT-2 small has $C = 768$ and $h = 12$. What is its `head_dim`? How many total parameters does its `MultiHeadAttention` module have, in the same form $4 C^2$? (Answer: $\text{head\_dim} = 64$, parameters $= 4 \cdot 768^2 = 2{,}359{,}296 \approx 2.36$ M per attention module, times 12 layers ≈ 28 M parameters total in GPT-2's attention layers.)

---

## 8.10 What's next

Multi-head attention completes the *attention half* of the transformer. The other half is a position-wise feed-forward network (a 2-layer MLP applied independently to each token's vector). Chapter 9 builds it, along with the **residual connection** that lets information bypass either half.

In Chapter 10 we add **layer normalisation**, the third and final piece of the transformer block. Chapter 11 puts it all together: `TransformerBlock(C, num_heads) = LayerNorm + MultiHeadAttention + LayerNorm + MLP`, with residual connections around both halves. After that chapter we have the entire body of GPT-2.

> **Looking ahead — what to remember from this chapter**
>
> 1. Multi-head attention is single-head attention with the channel axis split into $(h, d_h)$ groups, run in parallel, then concatenated.
> 2. The shape `(B, T, C) → (B, h, T, d_h)` is the only reshape needed; PyTorch's batched matmul handles the rest.
> 3. Parameter count is $4 C^2$ regardless of $h$: $h$ heads of size $d_h = C/h$ have the same total weights as one head of size $C$.
> 4. `MultiHeadAttention(C, 1)` and `SingleHeadAttention(C, C)` produce byte-for-byte identical output — the new module is a strict generalisation.

On to [Chapter 9 — The feed-forward network and residual connections](09_mlp_and_residuals.md) *(coming soon)*.
