---
title: 9. MLP and residual connections
nav_order: 10
parent: Part I — LLM Fundamentals
---

# Chapter 9 — The feed-forward network and residual connections

A transformer block has two sub-layers stacked one after the other. Chapters 6–8 built the first one: causal multi-head self-attention. This chapter builds the second one — a small **multi-layer perceptron** (MLP) that is applied independently to each token — and the **residual connection** that wraps both sub-layers.

By the end you will have:

- understood why the MLP is **position-wise** (the same weights are applied to every token's vector, independently),
- met the **GELU** activation (a smoother ReLU, which GPT-2 uses) and computed it by hand for one input,
- built `mygpt.MLP(embed_dim, dropout=0)` — a `Linear → GELU → Linear → Dropout` stack with $8 C^2 + 5C$ parameters for $C = 4$,
- understood the **residual connection** — the simple `x + sublayer(x)` pattern that makes deep networks trainable.

The MLP and the residual connection are the two pieces still missing before we can assemble a full transformer block in Chapter 11.

There is essentially no new mathematics in this chapter. The MLP is two linear layers with a non-linearity in between; the residual connection is one addition.

---

## 9.1 The other half of the transformer block

Each transformer block in GPT-2 has the structure

$$
y = \text{block}(x) \;\;\text{where}\;\; x \leftarrow x + \text{Attention}(\text{LayerNorm}(x)), \;\; x \leftarrow x + \text{MLP}(\text{LayerNorm}(x)).
$$

The two sub-layer outputs are *added* to the input rather than *replacing* it. We have built the attention sub-layer; we have not built the MLP, the layer norm, or wired up the residual additions. This chapter builds the MLP and explains the residual; Chapter 10 builds layer norm; Chapter 11 puts them in a single `TransformerBlock`.

Why do we need an MLP at all? Attention by itself is a *linear* mixer of value vectors — every output position is a (softmax-weighted) linear combination of the values at all positions. Linear combinations of linear functions are still linear, so stacking attention layers without anything else gives the model only the expressive power of one big linear map. The MLP introduces a **non-linearity** (the activation function) between the two linear projections, which lets the model learn arbitrary continuous functions of each token's vector.

The MLP is also where most of the parameter budget goes in a real transformer. At GPT-2 scale ($C = 768$), one MLP has $\approx 8 C^2 = 4.7$ M parameters; one multi-head attention has $4 C^2 = 2.4$ M. The MLP is roughly twice as big as the attention.

---

## 9.2 Setup

This chapter assumes you finished Chapter 8 — `mygpt/` exists with `VOCAB`, `to_ids`, `set_seed`, `TokenEmbedding`, `SingleHeadAttention`, and `MultiHeadAttention`.

If you skipped Chapter 8, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 8 ending state from `docs/_state_after_ch08.md`.

You are ready.

---

## 9.3 Position-wise feed-forward

The MLP we are about to build operates on *each token's vector independently*. Concretely: given an input tensor of shape $(B, T, C)$, every one of the $B \cdot T$ token vectors goes through *the same* two-layer network, and they do not interact. There is no mixing across positions inside the MLP — that mixing already happened in attention.

This means the MLP is just an ordinary 2-layer feed-forward neural network applied to a single vector, broadcast over the batch and time axes by PyTorch's `nn.Linear`. The shape rule:

$$
(B, T, C) \;\xrightarrow{\text{Linear}_1}\; (B, T, 4C) \;\xrightarrow{\text{GELU}}\; (B, T, 4C) \;\xrightarrow{\text{Linear}_2}\; (B, T, C).
$$

The intermediate dimension is **4C**. This is the canonical choice — every transformer paper and reference implementation we know of uses it — and we will not justify it from first principles. The intuition: the model needs room to "expand and recombine" before projecting back. A factor of 4 has been good enough for everyone.

---

## 9.4 GELU: a smoother ReLU

The non-linearity between the two linear layers is the **Gaussian Error Linear Unit** (GELU), introduced by Hendrycks and Gimpel (2016). Its definition is

$$
\text{GELU}(x) \;=\; x \cdot \Phi(x),
$$

where $\Phi$ is the cumulative distribution function (CDF) of the standard normal — that is, $\Phi(x) = P(Z \le x)$ where $Z \sim \mathcal{N}(0, 1)$. In words: GELU multiplies the input by the probability that a standard-normal random variable would be less than it. For very negative $x$, $\Phi(x) \approx 0$ and the output is near zero; for very positive $x$, $\Phi(x) \approx 1$ and the output is near $x$. Around the origin GELU is *smooth* — it has continuous derivatives everywhere, unlike ReLU's kink at zero.

PyTorch ships GELU as `nn.GELU` (or `F.gelu`). The two are interchangeable.

A small experiment to internalise GELU vs ReLU.

**Save the following to** 📄 `experiments/20_gelu_vs_relu.py`:

```python
"""Experiment 20 — GELU and ReLU compared on a few inputs.

GELU is smooth and slightly negative for moderately negative inputs;
ReLU is identically zero for all negative inputs and has a kink at 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    xs = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    print(f"x:    {xs}")
    print(f"gelu: {F.gelu(xs)}")
    print(f"relu: {F.relu(xs)}")
    print()

    g = nn.GELU()
    print(f"nn.GELU and F.gelu identical: {torch.equal(g(xs), F.gelu(xs))}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/20_gelu_vs_relu.py
```

**Expected output:**

```text
x:    tensor([-2.0000, -1.0000, -0.5000,  0.0000,  0.5000,  1.0000,  2.0000])
gelu: tensor([-0.0455, -0.1587, -0.1543,  0.0000,  0.3457,  0.8413,  1.9545])
relu: tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 2.0000])

nn.GELU and F.gelu identical: True
```

Two things to read off:

- **GELU is *not* zero for slightly negative inputs.** $\text{GELU}(-1) \approx -0.16$. The function bends through zero rather than clipping at zero. Many practitioners argue this gives smoother gradients.
- **GELU is approximately ReLU for $|x| > 2$.** $\text{GELU}(2) = 1.95$, $\text{GELU}(-2) = -0.046$. Far from zero, the two functions agree to within a percent.

For our purposes the key fact is: **GELU is what GPT-2 uses**, and it is what we will use. ReLU would also work; it would just put us slightly further from the reference architecture.

---

## 9.5 Residual connections

A **residual connection** wraps a sub-layer with an addition:

$$
y = x + \text{sub}(x).
$$

Instead of asking the sub-layer to compute the *whole* output, we ask it to compute the **change** that should be applied to the input. The original $x$ is preserved as a "highway" that information can travel along untransformed.

Three practical reasons this matters:

1. **Trainability.** Without residuals, gradients computed at the loss have to flow through every layer to reach the early parameters. With residuals, gradients can flow through the *addition* on the way back, which means each layer's parameters receive a healthy gradient even if the layer itself computes near-zero.
2. **Identity at initialisation.** If the sub-layer is initialised to output values near zero (which is typical for $\text{Linear} \to \text{Linear}$ stacks), the block initially behaves like the identity. The model "starts from doing nothing" and learns to add useful transformations.
3. **Composability.** Stacking residual blocks does not exponentially shrink or amplify activations. A pure stack of linear-and-activation layers can drive activations to zero or infinity over depth; residual stacks preserve scale.

GPT-2 wraps **both** sub-layers (attention and MLP) in residuals:

$$
x \leftarrow x + \text{Attn}(\text{LN}(x)), \qquad x \leftarrow x + \text{MLP}(\text{LN}(x)).
$$

In code:

```python
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```

We are not yet ready to assemble that block (we still need layer norm — Chapter 10), but in §9.7 we already use the residual form `out = x + mlp(x)` to demonstrate the pattern.

---

## 9.6 Putting it together: the `MLP` module

We add a third class to `mygpt`: an `MLP` module that wraps a 2-layer feed-forward with GELU. It does *not* include the residual connection — that lives at the call site, where we have access to both `x` and `mlp(x)`.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 9 the package gains an MLP module — the position-wise
feed-forward sub-block of a transformer. Together with the attention
sub-block and (in Chapter 10) layer norm, this is everything we need to
assemble a full transformer block in Chapter 11.
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
    """Multi-head causal self-attention."""

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

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_drop(self.W_O(out))


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear -> Dropout.

    Applied independently to each token's vector. The intermediate width
    is 4 * embed_dim, the canonical choice for transformer MLPs.

    Inputs:
        x: tensor of shape (B, T, embed_dim).
    Outputs:
        tensor of shape (B, T, embed_dim).

    Has 2 * (embed_dim * 4*embed_dim) + (4*embed_dim + embed_dim)
       = 8 * embed_dim^2 + 5 * embed_dim   parameters (with biases).
    """

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


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    mlp = MLP(embed_dim=C, dropout=0.0)
    mlp.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = mlp(x)
    out_residual = x + mlp(x)

    print(f"\nToken ids shape:                {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):       {tuple(x.shape)}")
    print(f"MLP output shape:               {tuple(out.shape)}")
    print(f"x + MLP(x) shape (residual):    {tuple(out_residual.shape)}")
    print()
    print(f"hidden_dim = 4*C = {4*C}, embed_dim = {C}")

    n_te = sum(p.numel() for p in te.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"MLP parameters:                   {n_mlp}")
    print(f"Total parameters:                 {n_te + n_mlp}")
```

(The file omits no previously-introduced classes — `SingleHeadAttention` and `MultiHeadAttention` are still in the file, just not used by `main`. The new lines are the `MLP` class and the bottom of `main`.)

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
MLP output shape:               (1, 4, 4)
x + MLP(x) shape (residual):    (1, 4, 4)

hidden_dim = 4*C = 16, embed_dim = 4

TokenEmbedding parameters:        16
MLP parameters:                   148
Total parameters:                 164
```

Three things to read off:

- **MLP parameters: 148.** That is $2 \cdot C \cdot 4C + 4C + C = 8 \cdot 16 + 16 + 4 = 148$ for $C = 4$. The two `nn.Linear` layers contribute $8C^2$ weights plus $5C$ biases. (The MLP layers use `bias=True`, in contrast to the attention layers that used `bias=False`. The reasoning is in §9.9 ex 2.)
- **Both `mlp(x)` and `x + mlp(x)` have the same shape** as the input, $(1, 4, 4)$. The MLP is a $C \to C$ map at each position; the residual is just an elementwise add of two same-shape tensors.
- **Total parameters jumped from 80 (Ch. 8) to 164.** The MLP's $148$ exceeds the $64$ from a single attention block. At GPT-2 scale, this ratio is preserved: $8C^2 / 4C^2 = 2$. The MLP is consistently the larger sub-layer.

---

## 9.7 Experiment 21 — Residual stabilises depth

The motivating claim of §9.5 — "residuals make depth tractable" — is easy to demonstrate empirically. We compare what happens to the *scale* of activations after stacking 30 randomly-initialised MLPs **with** vs **without** residual connections. Without residuals, the very first MLP shrinks the input scale by an order of magnitude and the rest of the stack runs at that smaller scale; with residuals, every layer's output is *added* to the input, so the scale is preserved across depth (and even drifts upward).

**Save the following to** 📄 `experiments/21_residual_stability.py`:

```python
"""Experiment 21 — Residual connections stabilise deep stacks.

Stacks 30 randomly-initialised MLPs and tracks the std of activations
after each layer, both with and without residual connections.
"""

import torch

from mygpt import MLP, set_seed


def main() -> None:
    set_seed(0)
    C = 16
    n_layers = 30
    mlps = [MLP(embed_dim=C, dropout=0.0).eval() for _ in range(n_layers)]

    x0 = torch.randn(1, 8, C)
    print(f"input std: {x0.std().item():.4f}")
    print()

    print("WITHOUT residuals:")
    with torch.no_grad():
        x = x0.clone()
        for i, mlp in enumerate(mlps):
            x = mlp(x)
            if i in (0, 4, 9, 14, 19, 29):
                print(f"  after layer {i+1:2d}: std = {x.std().item():.6f}")

    print()
    print("WITH residuals (x = x + mlp(x)):")
    with torch.no_grad():
        x = x0.clone()
        for i, mlp in enumerate(mlps):
            x = x + mlp(x)
            if i in (0, 4, 9, 14, 19, 29):
                print(f"  after layer {i+1:2d}: std = {x.std().item():.6f}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/21_residual_stability.py
```

**Expected output (numbers reproduce exactly at seed 0):**

```text
input std: 0.9369

WITHOUT residuals:
  after layer  1: std = 0.218545
  after layer  5: std = 0.075635
  after layer 10: std = 0.083192
  after layer 15: std = 0.072371
  after layer 20: std = 0.077279
  after layer 30: std = 0.096019

WITH residuals (x = x + mlp(x)):
  after layer  1: std = 0.981097
  after layer  5: std = 1.057667
  after layer 10: std = 1.080736
  after layer 15: std = 1.248647
  after layer 20: std = 1.528469
  after layer 30: std = 2.211950
```

Compare the two columns:

- **Without residuals**, the very first MLP shrinks the scale by ~4× (`0.94 → 0.22`), and by layer 5 the activations are at ~1/13th of the input scale. Subsequent layers do not shrink further — they plateau around `0.08` — but they also can't *recover* the original scale. Gradients flowing backward through this stack would be similarly attenuated.
- **With residuals**, the std stays close to the input's own (`0.94`) for the first few layers, then *drifts upward* — by layer 30 it has roughly tripled to `2.21`. The signal is preserved across depth, but unbounded growth is its own problem.

That upward drift is exactly the issue **layer norm** (Chapter 10) is designed to solve: it normalises every token's vector back to mean zero and unit variance, so the post-residual activations stay at a consistent scale no matter how many blocks we stack. With residuals + layer norm — the GPT-2 design — depth stops being a stability problem at all.

---

## 9.8 Experiments

1. **Activation choice doesn't break the world.** In `experiments/20_gelu_vs_relu.py`, modify `mlp.act = nn.GELU()` in your local copy of the `MLP` class to `mlp.act = nn.ReLU()`. The forward pass still runs and the output shape is unchanged; the parameter count is unchanged (activations have no parameters). What does change: the model's expressivity at moderate-magnitude negative inputs. We will not switch back, but it is good to know it costs nothing to try.
2. **Verify the MLP is position-wise.** Build `mlp = MLP(4); mlp.eval()`. Construct `x1 = torch.randn(1, 4, 4)`. Run `out1 = mlp(x1)`. Now permute the *time* axis: `x2 = x1[:, [3, 2, 1, 0], :]`. Run `out2 = mlp(x2)`. Verify that `torch.equal(out2, out1[:, [3, 2, 1, 0], :])`. Conclusion: the MLP commutes with permutations of the time axis — it really is independent across positions.
3. **The 4× expansion is a hyperparameter.** Modify the `MLP` class so the intermediate dimension is `2 * embed_dim` instead of `4 * embed_dim`. Recompute the parameter count: it should drop from $8C^2 + 5C$ to $4C^2 + 3C$ — exactly half plus a small bias term. (Real GPT-2 uses 4×; some efficient variants like ALBERT use smaller ratios.)
4. **Without bias, the parameter count is exactly $8C^2$.** Modify both `nn.Linear` calls to use `bias=False`. The MLP now has $8C^2 = 128$ parameters for $C = 4$, instead of $148$. Some recent transformers (e.g. PaLM) use bias-free MLPs to simplify the parallel-computation kernels.

After each experiment, restore the file you changed before moving on.

---

## 9.9 Exercises

1. **Position-wise = batched.** A single `nn.Linear(C, 4C)` applied to a tensor of shape $(B, T, C)$ produces a tensor of shape $(B, T, 4C)$. Argue that this is *exactly* the same as applying the same $\mathbb{R}^C \to \mathbb{R}^{4C}$ function to each of the $B \cdot T$ vectors independently. (Hint: write out the matrix multiplication along the last axis.)
2. **Why bias=True in MLP but bias=False in attention?** Recall §7.9 ex 2: in attention, a per-output-channel bias on $Q$ shifts every column of $Q K^\top$ by the same amount, which softmax cancels. So the bias is redundant. Argue that this argument **does not apply** to the MLP: there is no softmax, and shifting the input by a constant materially changes the output. Hence the MLP keeps its biases and attention does not.
3. **Parameter count for GPT-2 small.** GPT-2 small has $C = 768$. How many parameters does *one* MLP module have? (Answer: $8 \cdot 768^2 + 5 \cdot 768 = 4{,}722{,}432 \approx 4.72$ M.) GPT-2 has 12 layers × 1 MLP per layer = 12 MLPs total. What fraction of the 124 M total parameters is in the MLPs? (Answer: about 56.7 M / 124 M ≈ 46%.)
4. **The residual identity.** If `mlp(x)` is initialised to the zero function (e.g. by setting all weights to 0), then `x + mlp(x) = x`. Argue from the chain rule that the gradient of the loss with respect to a parameter inside `mlp` is non-zero only if `mlp(x)` is non-zero somewhere in its forward pass. (This is why we don't initialise the second linear layer to zero: it would be stuck at the identity forever.)

---

## 9.10 What's next

The transformer block has three pieces and we have built two. Chapter 10 introduces the third: **layer normalisation**, the operation that runs *before* each sub-layer's forward pass and re-scales every token's vector to zero mean / unit variance per token. With layer norm in hand, Chapter 11 wires up `LayerNorm + MultiHeadAttention + LayerNorm + MLP` with residuals around both halves into a single `TransformerBlock` class — the unit that GPT-2 stacks 12 times to make a complete model.

> **Looking ahead — what to remember from this chapter**
>
> 1. The MLP is `Linear(C, 4C) → GELU → Linear(4C, C) → Dropout`, applied independently to every token. It carries roughly twice as many parameters as one attention block.
> 2. GELU is a smooth approximation of ReLU; it is what GPT-2 uses, but ReLU would also work.
> 3. A residual connection is `out = x + sublayer(x)`. It preserves activation scale across depth and lets gradients flow past the sublayer at training time.
> 4. `mygpt.MLP(embed_dim, dropout=0.0)` has $8 C^2 + 5 C$ parameters; for $C = 4$ that is 148 parameters.

On to [Chapter 10 — Layer normalization](10_layer_norm.md) *(coming soon)*.
