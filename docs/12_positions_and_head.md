---
title: 12. Position embeddings and the LM head
nav_order: 13
parent: Part I — LLM Fundamentals
---

# Chapter 12 — Position embeddings and the language modeling head

After Chapter 11 we have the body of GPT-2: an embedding lookup, a stack of transformer blocks, and a (B, T, C) tensor at the top. Two things still missing before we have a complete model:

1. **Position embeddings** — a way to tell the model that token `"I"` at position 0 is different from token `"I"` at position 3. Without this, attention is permutation-invariant: it would happily produce the same logits for `"I love AI !"` and `"AI I ! love"`.
2. **The language-modelling head** — a final projection from the $C$-dimensional output of the last block to a $V$-dimensional logit vector at every position. Plus a final layer norm before the projection. GPT-2 uses a clever **weight-tying** trick that lets the head share weights with the token embedding, saving $V \cdot C$ parameters.

By the end you will have:

- understood why self-attention is permutation-invariant and how a single `nn.Embedding(max_seq_len, C)` lookup fixes it,
- met the **language-modelling head** as the final $C \to V$ linear projection that produces logits,
- seen the **weight-tied head** trick that saves $V \cdot C$ parameters in real GPT-2,
- assembled `mygpt.GPT(vocab_size, embed_dim, num_heads, num_layers)` — the full model — and verified it produces output of shape `(B, T, V)`.

After this chapter we have a model. Chapter 13 wires up the loss; Chapter 14 trains it.

---

## 12.1 The permutation-invariance problem

Self-attention, as we built it in Chapters 6–8, is **permutation-invariant** in a precise sense: if you permute the rows of the input, you get the same output rows in the corresponding permuted order. Concretely, if $X$ is a $(T, C)$ input and $P$ is any permutation matrix, then

$$
\text{Attention}(P X) \;=\; P \cdot \text{Attention}(X).
$$

Apply that to the language-modelling task: `mha([x_0, x_1, x_2])` and `mha([x_2, x_0, x_1])` produce the same set of output vectors, just reordered. The model has no way to tell whether a token came first, second, or third — it only sees content, not position.

That breaks language modelling. The whole point is that "love AI" and "AI love" are *different* — sequence order matters.

(Two technical caveats. The **causal mask** is a per-position constraint, but it does not actually break the symmetry: a permuted input would just see a different *prefix* at each position, with the same content-based attention pattern within that prefix. The MLP and residual sub-layers are also position-wise, so they don't break it either. The whole network is fully permutation-equivariant from $X$ to the final output.)

The fix is mechanical and elegant: **add a position-dependent vector** to every token embedding, *before* feeding it to the transformer body. Then "token 3 at position 0" and "token 3 at position 3" become different inputs from the model's point of view, even though they share the token embedding.

---

## 12.2 Setup

This chapter assumes you finished Chapter 11 — `mygpt/` exists with the full Chapter 11 module set ending in `TransformerBlock`.

If you skipped Chapter 11, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 11 ending state from `docs/_state_after_ch11.md`.

You are ready.

---

## 12.3 Learned position embeddings

GPT-2 uses **learned** position embeddings: a separate `nn.Embedding(max_seq_len, embed_dim)` lookup, indexed by the integer position $0, 1, \ldots, T-1$. The vectors are *learned* during training, just like the token embeddings — there is nothing built-in about how they work, no sine/cosine formula. The model figures out, on its own, what each position should contribute.

Concretely:

```python
self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

# In forward:
positions = torch.arange(T, device=ids.device)   # (T,)
x = self.token_embedding(ids) + self.position_embedding(positions)   # (B, T, C)
```

Three things to notice:

- **`positions` is a 1-D tensor `[0, 1, ..., T-1]`.** Same for every example in the batch — broadcasting handles the batch axis when we do the addition.
- **The two embeddings are added, not concatenated.** Adding keeps the output shape $(B, T, C)$ — same as the token embedding alone — so the rest of the model is unchanged. Concatenating would double the channel axis, requiring every downstream layer to rewire.
- **Position embeddings have $\text{max\_seq\_len} \cdot C$ parameters.** For our running example with $\text{max\_seq\_len}=64, C=4$, that is $256$ parameters — already bigger than the token embedding's $V \cdot C = 16$. For GPT-2 small ($\text{max\_seq\_len}=1024, C=768$), it is $786{,}432 \approx 0.79$ M — small compared to attention/MLP, but not negligible.

Let's see the effect empirically: with position embeddings, the same token at different positions gets a different vector.

**Save the following to** 📄 `experiments/25_position_breaks_invariance.py`:

```python
"""Experiment 25 — Position embeddings break self-attention's permutation invariance.

Without position embeddings, token 3 at position 3 has the same vector as
token 3 at position 0. With position embeddings added, they differ.
"""

import torch
import torch.nn as nn

from mygpt import TokenEmbedding, set_seed


def main() -> None:
    set_seed(0)
    V, C, max_seq = 4, 4, 8
    te = TokenEmbedding(V, C)
    pe = nn.Embedding(max_seq, C)

    # Two id sequences that share the token "3" at different positions
    ids1 = torch.tensor([0, 1, 2, 3])
    ids2 = torch.tensor([3, 2, 1, 0])

    # Without position embedding: token 3 always has the same vector
    print("Without position embedding (token 3 row):")
    print(f"  ids1 position 3: {te(ids1)[3]}")
    print(f"  ids2 position 0: {te(ids2)[0]}")
    print(f"  identical: {torch.equal(te(ids1)[3], te(ids2)[0])}")
    print()

    # With position embedding: same token at different positions differs
    def with_pos(ids):
        T = ids.shape[-1]
        positions = torch.arange(T)
        return te(ids) + pe(positions)

    v1 = with_pos(ids1)
    v2 = with_pos(ids2)
    print("With position embedding (token 3 at different positions):")
    print(f"  ids1 position 3 (token 3 at pos 3): {v1[3]}")
    print(f"  ids2 position 0 (token 3 at pos 0): {v2[0]}")
    print(f"  identical: {torch.equal(v1[3], v2[0])}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/25_position_breaks_invariance.py
```

**Expected output:**

```text
Without position embedding (token 3 row):
  ids1 position 3: tensor([ 0.1198,  1.2377,  1.1168, -0.2473], grad_fn=<SelectBackward0>)
  ids2 position 0: tensor([ 0.1198,  1.2377,  1.1168, -0.2473], grad_fn=<SelectBackward0>)
  identical: True

With position embedding (token 3 at different positions):
  ids1 position 3 (token 3 at pos 3): tensor([ 1.5092,  2.8240,  2.0631, -1.0910], grad_fn=<SelectBackward0>)
  ids2 position 0 (token 3 at pos 0): tensor([-1.2328, -0.4583,  1.6834,  0.5462], grad_fn=<SelectBackward0>)
  identical: False
```

The two rows of the *with-position* output are different (`identical: False`), even though both encode the same token id. The position embedding is doing its job.

---

## 12.4 The language-modelling head

After the last transformer block, every token's vector still has $C$ channels. To predict the next token we need to project from $C$ to $V$ — one logit per token in the vocabulary. That projection is the **language-modelling head**:

$$
\text{logits} \;=\; \text{LayerNorm}_\text{final}(\text{block\_output}) \cdot W_\text{head}^\top, \qquad W_\text{head} \in \mathbb{R}^{V \times C}.
$$

Three details:

- **A final `LayerNorm`** runs *before* the head, normalising the output of the last block. This is GPT-2's design — without it, the residual stream's drift (which we measured in §10.6) would feed directly into the head and push some logits to extreme values.
- **The head is `nn.Linear(C, V, bias=False)`** in a typical implementation. Its weight matrix is $W_\text{head}$ of shape $(V, C)$; output shape is $(B, T, V)$.
- **No softmax inside the head.** The head produces *logits* — unnormalised real numbers. Softmax happens later, either inside the cross-entropy loss (Chapter 13) or at generation time (Chapter 15). Producing logits is the right contract because cross-entropy is more numerically stable when computed directly from logits than from probabilities.

---

## 12.5 Weight tying: the GPT-2 trick

The head's weight $W_\text{head} \in \mathbb{R}^{V \times C}$ has the *exact same shape* as the token embedding's matrix $E \in \mathbb{R}^{V \times C}$. GPT-2 ties them: $W_\text{head} = E$. The head is then *not a separate parameter* — its forward pass uses `x @ E^T` directly, with $E$ being the same tensor that the token embedding looks up.

Two reasons this is appealing:

- **Saves $V \cdot C$ parameters.** For GPT-2 small ($V=50{,}257, C=768$) that is $\approx 38.6$ M — about 31% of the model. (Recall §5.10 ex 2: the embedding was already 31% of the total. Tying makes the head free.)
- **Aligns input and output spaces.** A token's input embedding $E[i]$ and its output prediction $W_\text{head}[i]$ now point in the same direction in $\mathbb{R}^C$. Empirically, this helps training: the model has fewer redundant degrees of freedom.

In code, we do not allocate an `nn.Linear` for the head at all. We just use the token embedding's weight matrix transposed:

```python
# Tied head — no separate Linear layer
logits = x @ self.token_embedding.embedding.weight.T  # (B, T, V)
```

`self.token_embedding.embedding.weight` is the $V \times C$ tensor inside `nn.Embedding`. Its transpose is $C \times V$. Multiplying `x: (B, T, C)` by it gives `(B, T, V)` — the logit shape we want.

---

## 12.6 Building `mygpt.GPT`

Time to assemble the full model. `GPT` is the top-level class that ties everything together, so it gets its own module file (`src/mygpt/model.py`).

**Append the following class to** 📄 `src/mygpt/model.py`:

```python
class GPT(nn.Module):
    """The full GPT-2-style decoder-only transformer.

    Inputs:
        ids: long tensor of shape (B, T) with values in [0, vocab_size).

    Outputs:
        logits: float tensor of shape (B, T, vocab_size), unnormalised.

    Architecture:
        token_embedding   (V, C) parameters
      + position_embedding (max_seq_len, C) parameters
      → embed_drop
      → N x TransformerBlock(C, num_heads)
      → ln_f (final LayerNorm)
      → head (tied to token_embedding.embedding.weight; no extra params)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(embed_dim)
        # No separate head: we reuse self.token_embedding.embedding.weight in forward.

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(T, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)  # (B, T, C)
        x = self.embed_drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        # Tied head: logits = x @ E^T, where E is the token-embedding matrix.
        logits = x @ self.token_embedding.embedding.weight.T  # (B, T, V)
        return logits
```

Then update **`main`** to construct a small GPT and run it on the running example:

```python
def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C, h, N = len(VOCAB), 4, 2, 2
    gpt = GPT(vocab_size=V, embed_dim=C, num_heads=h, num_layers=N,
              max_seq_len=64, dropout=0.0)
    gpt.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    logits = gpt(ids)

    print(f"\nToken ids shape:  {tuple(ids.shape)}")
    print(f"Logits shape:     {tuple(logits.shape)}  (B, T, V)")
    print()

    n_te = sum(p.numel() for p in gpt.token_embedding.parameters())
    n_pe = sum(p.numel() for p in gpt.position_embedding.parameters())
    n_blocks = sum(p.numel() for p in gpt.blocks.parameters())
    n_ln_f = sum(p.numel() for p in gpt.ln_f.parameters())
    n_total = sum(p.numel() for p in gpt.parameters())
    print(f"Token embedding       (V*C):       {n_te:>5}")
    print(f"Position embedding (max_seq*C):    {n_pe:>5}")
    print(f"{N} TransformerBlocks  (N*228):     {n_blocks:>5}")
    print(f"Final LayerNorm       (2*C):        {n_ln_f:>5}")
    print(f"Tied head            (0 extra):     {0:>5}")
    print(f"Total parameters:                  {n_total:>5}")
```

Run:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4

Token ids shape:  (1, 4)
Logits shape:     (1, 4, 4)  (B, T, V)

Token embedding       (V*C):          16
Position embedding (max_seq*C):      256
2 TransformerBlocks  (N*228):       456
Final LayerNorm       (2*C):            8
Tied head            (0 extra):         0
Total parameters:                    736
```

Three things to read off:

- **Logits shape `(1, 4, 4)`.** That is $(B, T, V) = (1, 4, 4)$. Every position gets a vector of $V = 4$ logits — one prediction for each of `"I"`, `"love"`, `"AI"`, `"!"` as the *next* token. Chapter 13 will use these to compute cross-entropy loss.
- **Total parameters: 736.** The breakdown is $16 + 256 + 456 + 8 = 736$ (no contribution from the tied head). The position embedding alone (256) is bigger than every other piece except the transformer blocks.
- **Tied head costs zero parameters.** `gpt.parameters()` walks the registered submodules; the head reuses `token_embedding.embedding.weight`, which is already counted under `token_embedding`. There is nothing else to count.

---

## 12.7 Experiments

1. **Wider model.** Construct `GPT(vocab_size=4, embed_dim=8, num_heads=2, num_layers=2, max_seq_len=64)`. Predicted parameter count: $V \cdot C + \text{max\_seq\_len} \cdot C + N \cdot (12 C^2 + 9 C) + 2 C = 32 + 512 + 1680 + 16 = 2240$. Verify by counting.
2. **Deeper model.** Construct `GPT(vocab_size=4, embed_dim=4, num_heads=2, num_layers=4)` (twice as many blocks). Predicted parameter count: $16 + 256 + 4 \cdot 228 + 8 = 1192$. Verify.
3. **Position-embedding hyperparameter.** Construct `GPT(vocab_size=4, embed_dim=4, num_heads=2, num_layers=2, max_seq_len=8)`. The position embedding now has $8 \cdot 4 = 32$ parameters instead of $256$ — a much smaller fraction of the model. Total: $16 + 32 + 456 + 8 = 512$. Verify.
4. **The tied head really is tied.** After constructing `gpt = GPT(...)`, change one entry of `gpt.token_embedding.embedding.weight.data` (e.g. `gpt.token_embedding.embedding.weight.data[0, 0] = 999.0`). Run `gpt(ids)` again and confirm the logits at *every* position have changed in their column 0 — because column 0 of the head's effective weight is row 0 of the token embedding, and we just modified it.

After each experiment, restore the file you changed before moving on.

---

## 12.8 Exercises

1. **Why max_seq_len affects parameter count but not training data size.** Increasing `max_seq_len` from 64 to 1024 grows the position embedding from $64 C$ to $1024 C$ — adding parameters. But it does *not* require more training data; the same data is just *available* to use up to length 1024. Argue why, and what the cost is at inference time. (Hint: per-step compute scales like $T^2$ in attention.)
2. **Why aren't position embeddings *outside* the transformer blocks like a "+pe" hack?** They are inside the model because they are learnable parameters that need gradients. Argue from the chain rule that putting `+pe` in `forward` (before the first block) gives `pe` a non-trivial gradient on every example. (Hint: $\frac{\partial \mathcal{L}}{\partial \text{pe}_t}$ flows back through every block to position $t$.)
3. **GPT-2 small parameter accounting.** Compute the number of parameters for GPT-2 small with $V=50{,}257, C=768, h=12, N=12, \text{max\_seq\_len}=1024$, with weight tying. (Answer: $V C + \text{max\_seq\_len} \cdot C + N \cdot (12 C^2 + 9 C) + 2 C = 38{,}597{,}376 + 786{,}432 + 85{,}026{,}816 + 1{,}536 = 124{,}412{,}160 \approx 124$ M. ✓ matches the published 124 M figure.)
4. **What does the tied head buy you in terms of generalisation?** The token embedding $E[i]$ and the head row $W_\text{head}[i]$ both represent token $i$ — one as input, one as output. With tying, training the model to predict token $i$ updates $E[i]$ in a useful way, *and* training the model to consume token $i$ updates $W_\text{head}[i]$ in a useful way. Argue informally why this should help small-sample generalisation.

---

## 12.9 What's next

We have a complete model. `mygpt.GPT(vocab_size=4, embed_dim=4, num_heads=2, num_layers=2)` has 736 parameters and produces a `(1, 4, 4)` logit tensor on the running example.

**Chapter 13** wires up the loss. Given logits and the *true* next-token ids, we compute the **cross-entropy loss** — the scalar that gradient descent will minimise. We will also write the forward-pass-with-loss method that real training loops use.

**Chapter 14** trains the model. We bring back the SGD loop from Chapter 4, give it a small text dataset, and watch the loss go down. After Chapter 14 we have a *trained* GPT.

**Chapter 15** generates text. Given a prompt and a trained model, we sample tokens one at a time, autoregressively, until we hit a stop condition.

> **Looking ahead — what to remember from this chapter**
>
> 1. Self-attention is permutation-invariant — without position information, `mha([x_0, x_1, x_2])` and `mha([x_2, x_0, x_1])` give the same set of output rows. Position embeddings break this symmetry by giving each position its own learned vector to add to the token embedding.
> 2. The language-modelling head is a $C \to V$ projection. GPT-2 ties its weight to the token embedding, saving $V \cdot C$ parameters.
> 3. A final `LayerNorm` runs before the head to renormalise the residual stream.
> 4. `mygpt.GPT(V=4, C=4, h=2, N=2, max_seq_len=64)` has 736 parameters and outputs logits of shape `(B, T, V)`.

On to [Chapter 13 — The forward pass with loss](13_forward_pass_with_loss.md).
