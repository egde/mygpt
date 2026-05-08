---
title: 6. Single-head self-attention
nav_order: 7
parent: Part I — LLM Fundamentals
---

# Chapter 6 — Single-head self-attention from scratch

After Chapter 5 every token is a vector. That is necessary but not sufficient: the model still treats each position independently. We need a way for position $t$'s representation to depend on the positions before it — `"love"` should be allowed to look at `"I"`, but not at `"AI"` (which has not been generated yet at the moment we predict the next token).

This chapter introduces the operation that does that: **causal self-attention**. We build it once, by hand, on a tiny three-token example, then wrap it as `mygpt.SingleHeadAttention`. By the end you will have:

- defined **softmax** for the first time and computed one by hand;
- understood **query, key, value** as three views of the same input, and what each one is for;
- computed a $(T, T)$ matrix of **attention scores** and applied a **causal mask**;
- run the running example `"I love AI !"` through the full attention layer and observed the output shape `(B, T, C)`.

There is more maths in this chapter than in any previous one. None of it goes beyond matrix multiplication and the chain rule.

---

## 6.1 The motivation: position-aware token mixing

After Chapter 5, our running example is a tensor of shape $(T, C) = (4, 4)$ — one row per token, four numbers per row. Call it $X$. Each row is *independent* of the others: nothing about row 1 reflects what row 0 contains.

A language model has to do better. To predict the token after `"I love"`, the representation at position 1 (`"love"`) needs to *know about* position 0 (`"I"`). Otherwise no information about the prefix can flow into the prediction.

We need an operation $\text{mix}: (T, C) \to (T, C)$ that produces a new $(T, C)$ tensor, where each row is a learned combination of the rows of $X$. **Self-attention** is one such operation. Two design constraints make it specifically suited to language modelling:

1. **Causality.** Position $t$ may only depend on positions $0, 1, \ldots, t$ — not on positions $t+1, \ldots, T-1$. We are training the model to predict the next token; it must not be allowed to peek ahead.
2. **Content-based weighting.** The amount of mixing between positions $i$ and $j$ should depend on what is *at* those positions, not just their indices. `"love"` should attend strongly to `"I"` because `"I love"` is a coherent fragment, not because position 1 is one step after position 0.

Self-attention satisfies both. We build it in five steps: softmax, Q/K/V projections, attention scores, causal mask, output. Each step is a single linear-algebra operation.

---

## 6.2 Setup

This chapter assumes you finished Chapter 5 — `mygpt/` exists with `VOCAB`, `to_ids`, `set_seed`, and `TokenEmbedding`. All commands below run from inside the `mygpt/` project root.

If you skipped Chapter 5, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 5 ending state:

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time."""

import torch
import torch.nn as nn


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")


def to_ids(tokens: list[str]) -> torch.Tensor:
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)
```

You are ready.

---

## 6.3 Softmax: from logits to probabilities

Before attention, we need one new operation. Given a vector of real numbers $\mathbf{z} = (z_0, z_1, \ldots, z_{n-1})$, the **softmax** function turns it into a probability distribution:

$$
\text{softmax}(\mathbf{z})_i \;=\; \frac{e^{z_i}}{\sum_{j=0}^{n-1} e^{z_j}}.
$$

Three properties to read off the formula:

- **Non-negative.** $e^{z_i} > 0$ for all real $z_i$, so every output coordinate is positive.
- **Sums to 1.** The denominator is the sum of all numerators, so the outputs sum to 1.
- **Order-preserving.** Larger $z_i$ → larger softmax output. Softmax monotonically *amplifies* the largest entries (because of the exponential) while preserving their ranking.

The inputs $z_i$ are called **logits** — unnormalised real numbers we hand to softmax. We will produce logits and call softmax in three places: at the end of attention (this chapter), inside cross-entropy loss (Chapter 13), and at generation time (Chapter 15).

A small numerical example. For $\mathbf{z} = (1, 2, 3)$:

$$
e^1 \approx 2.7183, \quad e^2 \approx 7.3891, \quad e^3 \approx 20.0855, \quad \text{sum} \approx 30.1929.
$$

Dividing gives

$$
\text{softmax}(\mathbf{z}) \approx (0.0900, \; 0.2447, \; 0.6652),
$$

which sums to 1. Notice that even though $z_2 = 3$ is only $1.5\times$ as large as $z_0 = 1$, the softmax output for index 2 is over $7\times$ as large as for index 0. The exponential disproportionately favours the largest input. This is exactly what we want when we are about to use the softmax outputs as **mixing weights** — they will concentrate attention on the highest-scoring positions.

PyTorch ships softmax as `torch.nn.functional.softmax(z, dim=-1)`. The `dim=-1` argument tells it to softmax along the *last* axis; in a tensor of attention scores of shape $(B, T, T)$, that is the right axis to normalise — each row of the $(T, T)$ score matrix becomes a probability distribution over the $T$ source positions.

**Save the following to** 📄 `experiments/13_softmax.py`:

```python
"""Experiment 13 — Softmax by hand and by torch."""

import math

import torch
import torch.nn.functional as F


def main() -> None:
    z = torch.tensor([1.0, 2.0, 3.0])
    print(f"z = {z}")

    # By hand
    exp_z = torch.exp(z)
    print(f"exp(z) = {exp_z}")
    total = exp_z.sum().item()
    print(f"sum of exp(z) = {total:.4f}")
    by_hand = exp_z / total
    print(f"softmax(z) by hand = {by_hand}")
    print()

    # By torch
    by_torch = F.softmax(z, dim=-1)
    print(f"softmax(z) by torch = {by_torch}")
    print(f"sum by torch = {by_torch.sum().item():.4f}")
    print(f"identical: {torch.allclose(by_hand, by_torch)}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/13_softmax.py
```

**Expected output:**

```text
z = tensor([1., 2., 3.])
exp(z) = tensor([ 2.7183,  7.3891, 20.0855])
sum of exp(z) = 30.1929
softmax(z) by hand = tensor([0.0900, 0.2447, 0.6652])

softmax(z) by torch = tensor([0.0900, 0.2447, 0.6652])
sum by torch = 1.0000
identical: True
```

---

## 6.4 Query, key, value: three views of the same input

Self-attention's core idea is that every input row plays *three* roles at once:

- as a **query** — asking *"what am I looking for?"*
- as a **key** — answering *"what am I offering?"*
- as a **value** — carrying *"what content do I contribute, if attention picks me?"*

Concretely, we project the input $X \in \mathbb{R}^{T \times C}$ through three learned linear maps:

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V,
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{C \times d_h}$ are the **weight matrices** of three `nn.Linear` layers (no bias), and $d_h$ is the **head dimension**. In this chapter we set $d_h = C$ for simplicity; in Chapter 8 we will set $d_h < C$ and run several heads in parallel.

Why three projections of the *same* input? Because the role a token plays as a query is generally different from its role as a key or value. Letting the model learn three separate $C \times d_h$ matrices gives it the freedom to express all three independently.

The shapes after projection:

$$
X: (T, C), \qquad Q, K, V: (T, d_h).
$$

For the running example with $T=4, C=4, d_h = 4$, each of $Q, K, V$ is a $4 \times 4$ matrix.

---

## 6.5 Attention scores and the scaled dot product

Given $Q$ and $K$, we want a $T \times T$ matrix `scores` where `scores[i, j]` measures how much position $i$ should attend to position $j$. The natural choice — and the one that defines self-attention — is the **dot product** of the query at $i$ with the key at $j$:

$$
\text{scores}[i, j] \;=\; Q_i \cdot K_j \;=\; \sum_{k=0}^{d_h - 1} Q_{i,k} \, K_{j,k}.
$$

In matrix form this is just $Q K^\top$:

$$
\text{scores} \;=\; Q K^\top \;\in\; \mathbb{R}^{T \times T}.
$$

If $Q_i$ and $K_j$ are similar (high cosine similarity, large magnitudes), the score is large — and we will pay more attention to position $j$. If they are dissimilar or small, the score is near zero, and position $j$ contributes little.

**Why divide by $\sqrt{d_h}$?** Scores are sums of $d_h$ products. If $Q$ and $K$ have entries of variance $\sigma^2$, then $Q_i \cdot K_j$ has variance proportional to $d_h \sigma^2$. As $d_h$ grows, scores get large in magnitude, and softmax pushed by large logits saturates — most of its output mass concentrates on a single index, gradients vanish, the model can no longer learn nuanced attention patterns. Dividing by $\sqrt{d_h}$ brings the variance of scores back to $\sigma^2$, regardless of $d_h$:

$$
\text{scaled\_scores} \;=\; \frac{Q K^\top}{\sqrt{d_h}}.
$$

This $\sqrt{d_h}$ factor is the only place the attention formula uses a square root, and it is non-negotiable.

---

## 6.6 Causal masking

Without intervention, every position would attend to every other position — including future ones. For language modelling that is wrong: at position $t$ we are predicting token $t+1$, and the model must not see tokens $\geq t+1$ during training.

The fix is to set the scores at "future" positions to $-\infty$ *before* softmax. Since $e^{-\infty} = 0$, those positions contribute zero to the softmax denominator and zero to the softmax output — they are exactly removed.

Concretely, we add a $T \times T$ **causal mask** matrix $M$ to the scaled scores:

$$
M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

The mask is the upper-triangular part of an all-`-inf` matrix (excluding the diagonal). In PyTorch:

```python
mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
```

For $T = 4$:

$$
M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}.
$$

After `scaled + mask` and softmax row-wise, the attention-weights matrix will be **lower-triangular** with rows that sum to 1.

---

## 6.7 The full attention computation

Putting the five steps together:

$$
\boxed{\text{Attention}(Q, K, V) \;=\; \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}} + M\right) V.}
$$

This is the equation Vaswani et al. wrote in 2017. In code (with `dim=-1` softmax along the source axis):

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_h)   # (T, T)
scores = scores + mask                              # causal
weights = F.softmax(scores, dim=-1)                 # (T, T), row-stochastic
out = weights @ V                                    # (T, d_h)
```

A final linear projection $W_O \in \mathbb{R}^{d_h \times C}$ takes the output back from $d_h$ to $C$ dimensions:

$$
\text{out\_final} = \text{Attention}(Q, K, V) \, W_O \;\in\; \mathbb{R}^{T \times C}.
$$

For $d_h = C$, $W_O$ is a square matrix; for $d_h < C$ (multi-head, Chapter 8) it concatenates and projects.

---

## 6.8 By hand on a tiny example

Math is easier to trust when you can reproduce it on paper. Take the simplest case where the math is hand-doable: $T = 3$ tokens, $C = 2$, and **identity projections** so $Q = K = V = X$:

$$
X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}.
$$

**Step 1 — scores $= QK^\top = XX^\top$.** Compute pairwise dot products:

$$
XX^\top = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}.
$$

(Read it: $X_0 \cdot X_2 = 1 \cdot 1 + 0 \cdot 1 = 1$, $X_2 \cdot X_2 = 1 + 1 = 2$, etc.)

**Step 2 — divide by $\sqrt{d_h} = \sqrt{2} \approx 1.4142$:**

$$
\text{scaled} \approx \begin{pmatrix} 0.7071 & 0 & 0.7071 \\ 0 & 0.7071 & 0.7071 \\ 0.7071 & 0.7071 & 1.4142 \end{pmatrix}.
$$

**Step 3 — add causal mask** (set entries above the diagonal to $-\infty$):

$$
\text{masked} = \begin{pmatrix} 0.7071 & -\infty & -\infty \\ 0 & 0.7071 & -\infty \\ 0.7071 & 0.7071 & 1.4142 \end{pmatrix}.
$$

**Step 4 — softmax row-wise.** Row 0: only entry 0 is finite, so softmax = $(1, 0, 0)$. Row 1: $\text{softmax}(0, 0.7071) = (1/(1+e^{0.7071}), e^{0.7071}/(1+e^{0.7071})) \approx (0.3302, 0.6698)$. Row 2: $\text{softmax}(0.7071, 0.7071, 1.4142)$. The two equal entries get equal mass; $e^{1.4142} \approx 4.113$ vs $e^{0.7071} \approx 2.028$, sum $\approx 8.169$, giving $\approx (0.2483, 0.2483, 0.5034)$.

$$
\text{weights} \approx \begin{pmatrix} 1 & 0 & 0 \\ 0.3302 & 0.6698 & 0 \\ 0.2483 & 0.2483 & 0.5035 \end{pmatrix}.
$$

Read off two facts: the matrix is **lower-triangular** (causal mask did its job), and **rows sum to 1** (softmax did its job).

**Step 5 — output $= \text{weights} \, V = \text{weights} \, X$.**

$$
\text{out}_0 = 1 \cdot X_0 = (1, 0).
$$
$$
\text{out}_1 = 0.3302 \cdot X_0 + 0.6698 \cdot X_1 = (0.3302, 0.6698).
$$
$$
\text{out}_2 = 0.2483 \cdot X_0 + 0.2483 \cdot X_1 + 0.5035 \cdot X_2 = (0.7517, 0.7517).
$$

So

$$
\text{out} \approx \begin{pmatrix} 1.0000 & 0.0000 \\ 0.3302 & 0.6698 \\ 0.7517 & 0.7517 \end{pmatrix}.
$$

We now reproduce that calculation in PyTorch.

**Save the following to** 📄 `experiments/14_attention_by_hand.py`:

```python
"""Experiment 14 — Causal self-attention by hand on a 3x2 input.

Uses identity projections (Q = K = V = X) so the scores are X X^T and
the math is checkable on paper. Compare the printed weights and output
to the by-hand calculation in §6.8.
"""

import math

import torch
import torch.nn.functional as F


def main() -> None:
    X = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    T, d = X.shape
    print(f"X (T={T}, C={d}):")
    print(X)
    print()

    # Step 1: scores = X X^T
    scores = X @ X.T
    print("scores = X X^T:")
    print(scores)
    print()

    # Step 2: scaled by 1/sqrt(d)
    scaled = scores / math.sqrt(d)
    print(f"scaled by 1/sqrt({d}) = 1/{math.sqrt(d):.4f}:")
    print(scaled)
    print()

    # Step 3: causal mask
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    masked = scaled + mask
    print("scaled + causal mask:")
    print(masked)
    print()

    # Step 4: softmax row-wise
    weights = F.softmax(masked, dim=-1)
    print("attention weights (lower-triangular, rows sum to 1):")
    print(weights)
    print(f"row sums: {weights.sum(dim=-1)}")
    print()

    # Step 5: output = weights @ V (V = X)
    out = weights @ X
    print("attention output = weights @ V:")
    print(out)


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/14_attention_by_hand.py
```

**Expected output:**

```text
X (T=3, C=2):
tensor([[1., 0.],
        [0., 1.],
        [1., 1.]])

scores = X X^T:
tensor([[1., 0., 1.],
        [0., 1., 1.],
        [1., 1., 2.]])

scaled by 1/sqrt(2) = 1/1.4142:
tensor([[0.7071, 0.0000, 0.7071],
        [0.0000, 0.7071, 0.7071],
        [0.7071, 0.7071, 1.4142]])

scaled + causal mask:
tensor([[0.7071,   -inf,   -inf],
        [0.0000, 0.7071,   -inf],
        [0.7071, 0.7071, 1.4142]])

attention weights (lower-triangular, rows sum to 1):
tensor([[1.0000, 0.0000, 0.0000],
        [0.3302, 0.6698, 0.0000],
        [0.2483, 0.2483, 0.5035]])
row sums: tensor([1., 1., 1.])

attention output = weights @ V:
tensor([[1.0000, 0.0000],
        [0.3302, 0.6698],
        [0.7517, 0.7517]])
```

Every printed number agrees with the by-hand calculation above. PyTorch is doing exactly what we did on paper.

---

## 6.9 Extending `mygpt`: `SingleHeadAttention`

Time to wrap the five steps as a learnable `nn.Module`. The differences between this code and the by-hand version are:

- **Three learnable projections** $W_Q, W_K, W_V$ replace the identity.
- **An output projection** $W_O$ at the end (allows the model to mix the head's output back into the embedding space; for $d_h = C$ it is a square matrix and could in principle be folded into $W_V$, but we keep it separate so the structure matches the multi-head version we will build in Chapter 8).
- **Operates in a batched $(B, T, C)$ shape**: PyTorch's `@` and `transpose(-2, -1)` handle the leading batch axis automatically.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 6 the package knows about: the running-example vocabulary,
how to convert tokens to id tensors (Chapter 3), how to seed PyTorch's RNG
(Chapter 4), a TokenEmbedding module (Chapter 5), and a single-head
causal self-attention module (this chapter).
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
    """Single-head causal self-attention.

    Inputs:
        x: tensor of shape (B, T, embed_dim).

    Outputs:
        tensor of shape (B, T, embed_dim).

    Has three learnable projections (W_Q, W_K, W_V) of shape
    (embed_dim, head_dim) and a final output projection W_O of shape
    (head_dim, embed_dim). For single-head we set head_dim = embed_dim;
    for multi-head (Chapter 8) head_dim < embed_dim.
    """

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
        Q = self.W_Q(x)                          # (B, T, head_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, T, T)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)                          # (B, T, T)
        out = weights @ V                                             # (B, T, head_dim)
        return self.W_O(out)                                          # (B, T, embed_dim)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(vocab_size=V, embed_dim=C)
    attn = SingleHeadAttention(embed_dim=C, head_dim=C)

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)   # (1, T)
    x = te(ids)                                           # (1, T, C)
    out = attn(x)                                         # (1, T, C)

    print(f"\nToken ids shape:           {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):  {tuple(x.shape)}")
    print(f"Attention output (B, T, C): {tuple(out.shape)}")

    n_te = sum(p.numel() for p in te.parameters())
    n_attn = sum(p.numel() for p in attn.parameters())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"SingleHeadAttention parameters:   {n_attn}")
    print(f"Total:                            {n_te + n_attn}")
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
Total:                            80
```

The 64 parameters of `SingleHeadAttention` are exactly the four $C \times d_h = 4 \times 4 = 16$ matrices ($W_Q, W_K, W_V, W_O$), giving $4 \times 16 = 64$.

---

## 6.10 End-to-end: attention on `"I love AI !"`

We can now run the full pipeline — embed the running example, then attend — and observe the attention weights as the model sees them at initialisation. (At init the projections are random, so the weights are not "meaningful" yet; they will become meaningful only after Chapter 14's training. For now we observe their *shape* and the causal-mask structure.)

**Save the following to** 📄 `experiments/15_attention_running_example.py`:

```python
"""Experiment 15 — End-to-end self-attention on the running example.

Embeds 'I love AI !', runs SingleHeadAttention, and prints the
intermediate attention weights so you can see the lower-triangular
structure imposed by the causal mask.
"""

import math

import torch
import torch.nn.functional as F

from mygpt import VOCAB, SingleHeadAttention, TokenEmbedding, set_seed, to_ids


def main() -> None:
    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(vocab_size=V, embed_dim=C)
    attn = SingleHeadAttention(embed_dim=C, head_dim=C)

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)

    # Re-run attention manually so we can pull out the weights matrix
    Q = attn.W_Q(x)
    K = attn.W_K(x)
    V_proj = attn.W_V(x)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(attn.head_dim)
    mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
    scores = scores + mask
    weights = F.softmax(scores, dim=-1)

    print("Attention weights (B=1, T=4, T=4) — first batch element:")
    torch.set_printoptions(precision=4)
    print(weights[0].detach())
    print()
    print(f"row sums: {weights[0].sum(dim=-1).detach()}")
    print()

    # Standard module output
    out = attn(x)
    print(f"Final attention output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/15_attention_running_example.py
```

**Expected output (your numerical weights at seed 0 will exactly match — they are deterministic):**

```text
Attention weights (B=1, T=4, T=4) — first batch element:
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.2526, 0.7474, 0.0000, 0.0000],
        [0.3018, 0.2972, 0.4011, 0.0000],
        [0.2149, 0.3615, 0.1914, 0.2321]])

row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000])

Final attention output shape: (1, 4, 4)
```

Three things to notice:

- The attention-weights matrix is **lower-triangular** — every entry above the diagonal is zero. The causal mask is doing its job.
- **Each row sums to 1**, regardless of how many positions are unmasked. Softmax does this for free.
- The **first row is `(1, 0, 0, 0)`** — position 0 has no other positions to attend to (it is the first token), so all of its attention goes to itself. This is structural, not learned.

---

## 6.11 Experiments

1. **Symmetry of identity attention.** In `experiments/14_attention_by_hand.py`, the input $X$ is *not* symmetric (rows are different), yet the matrix `scores = X X^T` is symmetric. Why? Verify by adding `print(torch.equal(scores, scores.T))` after the scores are computed.
2. **Remove the scale factor.** In `experiments/15_attention_running_example.py`, change `/ math.sqrt(attn.head_dim)` to nothing (just `Q @ K.transpose(-2, -1)`). Re-run. With $d_h = 4$ the change is small; with $d_h = 64$ (try it by setting `C = 64` in the script) the unscaled scores become large enough that some attention rows put almost all mass on a single column. The chapter's claim "scaled prevents softmax saturation" is exactly this effect.
3. **Remove the causal mask.** Set `mask = torch.zeros(4, 4)` instead of the upper-triangular `-inf`. Re-run experiment 15. The weights are now *full*, not lower-triangular: position 0 attends to all four positions. Confirm row sums are still 1 (softmax doesn't care about the mask shape, only the input values).
4. **Confirm weights become uniform when scores are equal.** In `experiments/14_attention_by_hand.py`, replace `X` with `torch.zeros(3, 2)`. With identity projections the scores matrix is all zeros; softmax over zeros gives uniform weights $(1/n, \ldots, 1/n)$ in each row. With causal masking, row 0 = $(1, 0, 0)$, row 1 = $(0.5, 0.5, 0)$, row 2 = $(1/3, 1/3, 1/3)$. Verify.

After each experiment, restore the file you changed before moving on.

---

## 6.12 Exercises

1. **Why $\sqrt{d_h}$ specifically?** Show that if $Q$ and $K$ have entries of mean 0 and variance $\sigma^2$, then the dot product $Q_i \cdot K_j$ has variance $d_h \sigma^4$ (assume entries are independent). Conclude that dividing by $\sqrt{d_h}$ brings the standard deviation of scores back to $\sigma^2$, independent of $d_h$.
2. **Parameter count of `SingleHeadAttention(C, C)`.** In terms of $C$, write down the total parameter count (no biases). Generalise to `SingleHeadAttention(C, d_h)` for $d_h \neq C$.
3. **Why bias=False?** All four `nn.Linear` layers in our `SingleHeadAttention` use `bias=False`. Argue from the formula why an additive bias on $Q$ or $K$ alone has no effect on the *softmax* output (hint: softmax is invariant to adding the same constant to every input — see the next chapter for a proof).
4. **The first-row degeneracy.** In §6.10 we noted the first row of attention weights is always $(1, 0, \ldots, 0)$ regardless of input. Argue that this is a consequence of the causal mask having only one unmasked entry on row 0, plus the fact that softmax over a single value always returns 1.

---

## 6.13 What's next

Chapter 7 turns the inline self-attention computation into a small, reusable module that we can stack — and Chapter 8 generalises to **multi-head** attention, where several heads run in parallel with $d_h < C$ and their outputs are concatenated. Together they finish Part II; from Chapter 9 we wire attention into the full transformer block (residual connection + MLP + layer norm).

> **Looking ahead — what to remember from this chapter**
>
> 1. Self-attention is a $(T, C) \to (T, C)$ operation; each output row is a learned, content-dependent mixture of the input rows.
> 2. The five-step recipe is: project to $Q, K, V$, compute scaled dot-product scores, apply causal mask, softmax, multiply by $V$.
> 3. Softmax over a vector with $-\infty$ in some positions zeroes those positions exactly — that is how the causal mask "removes" future positions.
> 4. The $\sqrt{d_h}$ scale factor prevents softmax saturation as $d_h$ grows; without it, large-$d_h$ models cannot learn good attention patterns.
> 5. `mygpt.SingleHeadAttention(C, d_h)` has $4 \cdot C \cdot d_h$ parameters — one $C \times d_h$ matrix each for $Q, K, V$ and one $d_h \times C$ for the output projection.

On to [Chapter 7 — A reusable attention module](07_reusable_attention.md).
