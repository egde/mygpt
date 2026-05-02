---
title: 11. The transformer block
nav_order: 12
parent: LLM Fundamentals
---

# Chapter 11 — Putting it together: the transformer block

Eight chapters of building, and here we are. We have all three pieces: `MultiHeadAttention` (Chapters 6–8), `MLP` (Chapter 9), and `LayerNorm` (Chapter 10). This chapter composes them into the unit GPT-2 stacks: a single **transformer block**, with pre-norm and residuals around both halves.

By the end you will have:

- understood the four-line `forward` of a GPT-2 block (`x ← x + mha(ln(x))`, `x ← x + mlp(ln(x))`),
- built `mygpt.TransformerBlock(embed_dim, num_heads)` — `228` parameters for $C=4, h=2$ — and verified it on the running example,
- stacked two of them into a `torch.nn.Sequential` and watched the running-example tensor flow through both,
- met the GPT-2 small numbers ($C = 768, h = 12$, ≈ 7.1 M parameters per block, 12 blocks ≈ 85 M of the 124 M total).

There is no new mathematics or new tensor manipulation in this chapter. The block is composition.

---

## 11.1 Recap: the three pieces and the recipe

After Chapter 10 we have:

| Sub-module                 | Shape                | Parameters (with $C, h$ as before)              |
|----------------------------|----------------------|-------------------------------------------------|
| `MultiHeadAttention(C, h)` | `(B,T,C) → (B,T,C)`  | $4 C^2$                                         |
| `MLP(C)`                   | `(B,T,C) → (B,T,C)`  | $8 C^2 + 5 C$                                   |
| `LayerNorm(C)`             | `(...,C) → (...,C)`  | $2 C$                                           |

GPT-2's transformer block runs each sub-module once, with a `LayerNorm` *before* it and a residual *around* it:

$$
\begin{aligned}
\mathbf{x} &\leftarrow \mathbf{x} + \text{MHA}(\text{LN}_1(\mathbf{x})) \\
\mathbf{x} &\leftarrow \mathbf{x} + \text{MLP}(\text{LN}_2(\mathbf{x}))
\end{aligned}
$$

That's two layer norms, one attention, one MLP, two residuals. In code this is four lines (or two if we use `+=`-style mutation, which we won't, because PyTorch doesn't like it inside `forward`).

The total parameter count of one block is

$$
\underbrace{4 C^2}_{\text{MHA}} + \underbrace{8 C^2 + 5 C}_{\text{MLP}} + \underbrace{2 \cdot 2 C}_{\text{2 × LayerNorm}} \;=\; 12 C^2 + 9 C.
$$

For our $C = 4$ running example: $12 \cdot 16 + 9 \cdot 4 = 192 + 36 = 228$. We will see that number on the screen in §11.4.

---

## 11.2 Setup

This chapter assumes you finished Chapter 10 — `mygpt/` exists with the full Chapter 10 module set: `VOCAB`, `to_ids`, `set_seed`, `TokenEmbedding`, `SingleHeadAttention`, `MultiHeadAttention`, `MLP`, `LayerNorm`.

If you skipped Chapter 10, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 10 ending state from `docs/_state_after_ch10.md`.

You are ready.

---

## 11.3 The `TransformerBlock` class

The class is short. Most of the surface area is in the three sub-modules; the block itself just holds them and runs them in order.

**Append the following class to** 📄 `src/mygpt/__init__.py` (after `LayerNorm`, before `main`):

```python
class TransformerBlock(nn.Module):
    """A single GPT-2 pre-norm transformer block.

    forward(x) computes:
        x = x + mha(ln1(x))      # residual around attention
        x = x + mlp(ln2(x))      # residual around MLP

    Inputs / outputs:
        (B, T, embed_dim) -> (B, T, embed_dim).

    Total parameters: 12 * embed_dim^2 + 9 * embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ln1 = LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

Two design notes:

- **Pre-norm, not post-norm.** The `ln1` and `ln2` are applied *inside* the residual additions: the residual is added to the *un-normalised* input. The motivation came from §10.5 (cleaner gradients through the residual highway). All modern decoder-only transformers do this.
- **Each sub-layer keeps its own dropout.** `MultiHeadAttention` has its `attn_drop` and `out_drop`; `MLP` has its `drop`. The block doesn't add a third dropout — the two sub-modules already cover the standard places GPT-2 drops out.

Then update **`main`** to demonstrate the block:

```python
def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    block = TransformerBlock(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0)
    block.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = block(x)

    print(f"\nToken ids shape:           {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):  {tuple(x.shape)}")
    print(f"Block output shape:        {tuple(out.shape)}")
    print()
    print(block)
    print()

    n_te = sum(p.numel() for p in te.parameters())
    n_block = sum(p.numel() for p in block.parameters())
    print(f"TokenEmbedding parameters:   {n_te}")
    print(f"TransformerBlock parameters: {n_block}")
    print(f"Total parameters:            {n_te + n_block}")
```

---

## 11.4 Verifying the block on the running example

Run the package:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4

Token ids shape:           (1, 4)
Embedded shape (B, T, C):  (1, 4, 4)
Block output shape:        (1, 4, 4)

TransformerBlock(
  (ln1): LayerNorm()
  (mha): MultiHeadAttention(
    (W_Q): Linear(in_features=4, out_features=4, bias=False)
    (W_K): Linear(in_features=4, out_features=4, bias=False)
    (W_V): Linear(in_features=4, out_features=4, bias=False)
    (W_O): Linear(in_features=4, out_features=4, bias=False)
    (attn_drop): Dropout(p=0.0, inplace=False)
    (out_drop): Dropout(p=0.0, inplace=False)
  )
  (ln2): LayerNorm()
  (mlp): MLP(
    (fc1): Linear(in_features=4, out_features=16, bias=True)
    (act): GELU(approximate='none')
    (fc2): Linear(in_features=16, out_features=4, bias=True)
    (drop): Dropout(p=0.0, inplace=False)
  )
)

TokenEmbedding parameters:   16
TransformerBlock parameters: 228
Total parameters:            244
```

Three things to read off:

- **Total block parameters: 228.** That is $12 C^2 + 9 C = 192 + 36 = 228$ for $C = 4$. Matches the §11.1 formula.
- **Output shape `(1, 4, 4)`** equals the input shape — the block is a `(B, T, C) → (B, T, C)` map, which is what makes it stackable.
- **The printed module tree** shows the four submodules (`ln1`, `mha`, `ln2`, `mlp`) with their internal structure. `LayerNorm()` prints with empty parens because our hand-rolled class doesn't define a custom `__repr__`. `MLP` and `MultiHeadAttention` use their default `nn.Module` reprs and show their internal layers.

---

## 11.5 Stacking blocks

The whole point of the block is that it is **stackable**: you can put $N$ of them in series and the input/output shape is the same `(B, T, C)`. GPT-2 small has $N = 12$.

**Save the following to** 📄 `experiments/24_stack_two_blocks.py`:

```python
"""Experiment 24 — Stack two TransformerBlocks and watch the running example
flow through both. Confirms the output shape is unchanged and the parameter
count is exactly 2 × (single-block parameters).
"""

import torch

from mygpt import TokenEmbedding, TransformerBlock, set_seed, to_ids


def main() -> None:
    set_seed(0)
    V, C = 4, 4
    te = TokenEmbedding(V, C)
    blocks = torch.nn.Sequential(
        TransformerBlock(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0),
        TransformerBlock(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0),
    )
    blocks.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = blocks(x)

    print(f"input shape:    {tuple(x.shape)}")
    print(f"after 2 blocks: {tuple(out.shape)}")
    print()

    n_te = sum(p.numel() for p in te.parameters())
    n_blocks = sum(p.numel() for p in blocks.parameters())
    print(f"TokenEmbedding params: {n_te}")
    print(f"2 TransformerBlocks:   {n_blocks}  (= 2 * 228)")
    print(f"Total:                 {n_te + n_blocks}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/24_stack_two_blocks.py
```

**Expected output:**

```text
input shape:    (1, 4, 4)
after 2 blocks: (1, 4, 4)

TokenEmbedding params: 16
2 TransformerBlocks:   456  (= 2 * 228)
Total:                 472
```

Two facts:

- **Stacking is just `nn.Sequential`.** Because every block is a $(B,T,C) \to (B,T,C)$ map, the simplest possible composition (`nn.Sequential` over $N$ blocks) is the right one. GPT-2's `forward` does exactly this — it's a `for` loop over a `nn.ModuleList` of blocks, but mathematically identical to `nn.Sequential`.
- **Parameter count is linear in depth.** Two blocks have exactly twice as many parameters as one. There are no shared weights between blocks — each one has its own $W_Q, W_K, W_V, W_O$ and its own $W_{\text{fc1}}, W_{\text{fc2}}$ and its own $\boldsymbol{\gamma}_1, \boldsymbol{\beta}_1, \boldsymbol{\gamma}_2, \boldsymbol{\beta}_2$.

For GPT-2 small with $C = 768, h = 12, N = 12$:

$$
\begin{aligned}
\text{params per block} &= 12 \cdot 768^2 + 9 \cdot 768 \;\approx\; 7{,}085{,}000 \approx 7.08 \text{ M} \\
\text{12 blocks} &\approx 85 \text{ M parameters}
\end{aligned}
$$

Out of GPT-2 small's 124 M total, ≈ 85 M live in the 12 transformer blocks. The remaining ≈ 39 M is the token + position embeddings (~38.6 M, Chapter 12) and a final layer norm + the language-modelling head (which is *tied* to the token embedding, so it costs zero additional parameters; Chapter 12 explains).

---

## 11.6 Experiments

1. **A 1-head block.** Construct `TransformerBlock(embed_dim=4, num_heads=1)`. Its parameter count should still be `12 * 4 * 4 + 9 * 4 = 228` — `num_heads` does not appear in the formula. Verify by running `sum(p.numel() for p in block.parameters())`. (Recall §8.6: changing `num_heads` factorises the same number of parameters differently; it does not change the total.)
2. **A wider block.** Construct `TransformerBlock(embed_dim=8, num_heads=2)`. Predicted parameter count: $12 \cdot 64 + 9 \cdot 8 = 768 + 72 = 840$. Verify by counting.
3. **The output is *not* the input** even at random init. Run `block(x)` and `print((out - x).abs().max())` for a fresh `block` and a fresh `x = torch.randn(1, 4, 4)`. The max absolute difference should be of order 1 — random initialisation does not produce the identity, even though the residual is in place.
4. **Identity at zero MLP/MHA.** Manually set `block.mha.W_O.weight.data.zero_()` and `block.mlp.fc2.weight.data.zero_()` (and the corresponding biases — `block.mlp.fc2.bias.data.zero_()`). Now both sub-layer outputs are zero, so the block is the identity. Confirm by running and printing `(block(x) - x).abs().max()` — should be ~`0` (within float32 noise).

After each experiment, restore the files you changed before moving on.

---

## 11.7 Exercises

1. **Why pre-norm and not post-norm here?** The original transformer paper (2017) used post-norm: `x ← LN(x + sub(x))`. GPT-2 (2019) switched to pre-norm. Our block follows GPT-2. Argue from §10.5 why pre-norm gives cleaner gradient flow through the residual at the cost of letting the residual stream's scale drift.
2. **Where do biases live in our block?** List every `nn.Linear` and `LayerNorm` in `TransformerBlock` and note whether each has a bias. (Answer: MHA's four linears use `bias=False`; MLP's two linears use `bias=True`; both LayerNorms have a `bias` parameter (they always do). Total bias parameters: `0 + (4*4 + 4) + (4 + 4) = 0 + 20 + 8 = 28` for $C = 4$.)
3. **GPT-2 small parameter count.** Verify the §11.5 claim that one GPT-2 small block has ≈ 7.08 M parameters. With $C = 768$: $12 \cdot 768^2 + 9 \cdot 768 = 7{,}085{,}568$ ≈ $7.09$ M. With 12 blocks: $\approx 85.0$ M.
4. **The block is shape-preserving for any T ≤ max_seq_len.** Construct `TransformerBlock(4, 2, max_seq_len=64)`. Try inputs of shape `(1, T, 4)` for `T ∈ {1, 2, 4, 16, 64}`. Each should produce output of the same shape. Try `T = 65` — you should get a `ValueError` from the inherited `MultiHeadAttention` check.

---

## 11.8 What's next

We have the body of GPT-2. To make a complete model we need two more pieces:

- **Position embeddings.** Self-attention is permutation-invariant — `mha([x_0, x_1, x_2])` and `mha([x_2, x_0, x_1])` would compute the same attention scores, just with rows reordered. To break this symmetry the model adds a position-dependent vector to each token's embedding. Chapter 12 builds it.
- **The language-modelling head.** A final `Linear(C, V)` that projects each output position back to a $V$-dimensional logit vector — one logit per token in the vocabulary. Plus a final `LayerNorm` before the head. Chapter 12 builds this too, including the GPT-2 weight-tying trick that saves $V \cdot C$ parameters.

With those two pieces in place, Chapter 13 will write the forward pass that computes the cross-entropy loss, and Chapter 14 the training loop. Then we can actually train.

> **Looking ahead — what to remember from this chapter**
>
> 1. A `TransformerBlock` is `LayerNorm → MultiHeadAttention → residual → LayerNorm → MLP → residual`. Two layer norms, two residual additions, two sub-layers.
> 2. Parameter count: $12 C^2 + 9 C$. Independent of `num_heads`.
> 3. The block preserves shape `(B, T, C) → (B, T, C)`, so stacking is `nn.Sequential` over $N$ blocks.
> 4. `mygpt.TransformerBlock(C=4, num_heads=2)` has 228 parameters; GPT-2 small's block has ≈ 7.08 M.

On to [Chapter 12 — Position embeddings and the language modeling head](12_positions_and_head.md) *(coming soon)*.
