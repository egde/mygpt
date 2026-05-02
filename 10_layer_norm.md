---
title: 10. Layer normalization
nav_order: 11
parent: LLM Fundamentals
---

# Chapter 10 — Layer normalization

In Chapter 9 we built the MLP and saw that residual connections preserve activation scale across depth — but with an upward drift (`std = 0.94` at layer 1 → `2.21` at layer 30). This chapter introduces the third and final piece of the transformer block, **layer normalization**, which keeps every sub-layer's input at a controlled scale.

By the end you will have:

- understood the **per-token** mean/std normalisation that defines layer norm,
- built `mygpt.LayerNorm(embed_dim)` from scratch and verified it matches PyTorch's `nn.LayerNorm` to within float-32 precision,
- met the **pre-norm** convention GPT-2 uses (`out = x + sub(LN(x))`) and its trade-off vs the original transformer's **post-norm** (`LN(x + sub(x))`),
- watched LayerNorm reduce the §9.7 drift from `2.21` to `1.69` over 30 layers.

After this chapter, every piece of GPT-2's transformer block is in place. Chapter 11 wires them up.

---

## 10.1 What layer norm is for

A `Linear` layer with weights drawn from any reasonable distribution can produce outputs whose *scale* is unrelated to its inputs. After Chapter 9 we saw the consequences: stack 30 of them with residual connections and the activation std drifts from `0.94` to `2.21`. The operations are individually reasonable; the cumulative effect is not.

**Layer normalisation** breaks the cumulative drift by *re-normalising* the input to each sub-layer. For every token's vector $\mathbf{x} \in \mathbb{R}^C$, layer norm computes the mean and (population) variance over the $C$ channels and applies

$$
\text{LN}(\mathbf{x}) \;=\; \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu(\mathbf{x})}{\sqrt{\sigma^2(\mathbf{x}) + \varepsilon}} \;+\; \boldsymbol{\beta},
$$

where

- $\mu(\mathbf{x}) = \frac{1}{C} \sum_c x_c$ is the per-token mean,
- $\sigma^2(\mathbf{x}) = \frac{1}{C} \sum_c (x_c - \mu(\mathbf{x}))^2$ is the per-token (population) variance,
- $\varepsilon \approx 10^{-5}$ is a small constant that prevents divide-by-zero when a token's vector is constant,
- $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^C$ are **learnable** scale and bias vectors.

After the normalisation step, every token's vector has mean 0 and variance 1 (per token, over the $C$ axis). The learnable $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ then re-scale and shift each channel; with the default initialisation $\boldsymbol{\gamma} = \mathbf{1}, \boldsymbol{\beta} = \mathbf{0}$, layer norm starts as the pure normalisation, and the model learns any deviation from that during training.

Two adjectives worth knowing:

- **Per-token.** The mean and variance are computed independently for every $(b, t)$ position. There is *no* mixing across positions or across the batch. This contrasts with **batch norm**, which averages across the batch — a problem for autoregressive generation, where the batch is not always large or stable.
- **Learnable.** The two parameters $\boldsymbol{\gamma}, \boldsymbol{\beta}$ have shape $(C,)$ each, contributing $2C$ parameters per `LayerNorm` module. For $C = 4$ that is 8 parameters; for GPT-2's $C = 768$ it is 1,536 — negligible compared to the millions of parameters in attention and MLP.

---

## 10.2 Setup

This chapter assumes you finished Chapter 9 — `mygpt/` exists with `VOCAB`, `to_ids`, `set_seed`, `TokenEmbedding`, `SingleHeadAttention`, `MultiHeadAttention`, and `MLP`.

If you skipped Chapter 9, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 9 ending state from `docs/_state_after_ch09.md`.

You are ready.

---

## 10.3 By hand: per-token mean, variance, normalise

Let's compute layer norm by hand on a tiny example, then check it in PyTorch. Take a single token vector

$$
\mathbf{x} = (1, 2, 3, 4) \in \mathbb{R}^4.
$$

**Mean:** $\mu = (1 + 2 + 3 + 4) / 4 = 2.5$.

**Centred values:** $\mathbf{x} - \mu = (-1.5, -0.5, 0.5, 1.5)$.

**Variance** (population, divisor $n$, not $n-1$): $\sigma^2 = (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 4 = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25$.

**Standard deviation:** $\sigma = \sqrt{1.25} \approx 1.1180$.

**Normalised values:** $(\mathbf{x} - \mu) / \sigma \approx (-1.342, -0.447, 0.447, 1.342)$.

Sanity check: the four normalised values average to 0 and have a (population) variance of 1 — confirm by hand or trust the algebra.

With $\boldsymbol{\gamma} = \mathbf{1}, \boldsymbol{\beta} = \mathbf{0}$ (the default initialisation), the normalised values are also the layer-norm output. That is the version we verify against PyTorch next.

**Save the following to** 📄 `experiments/22_layernorm_by_hand.py`:

```python
"""Experiment 22 — Layer norm by hand on x = (1, 2, 3, 4)."""

import torch
import torch.nn as nn


def main() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    mean = x.mean()
    var = x.var(unbiased=False)         # divisor n, matches LayerNorm
    eps = 1e-5
    x_normed = (x - mean) / torch.sqrt(var + eps)

    print(f"x:           {x}")
    print(f"mean:        {mean.item():.6f}")
    print(f"var:         {var.item():.6f}")
    print(f"std:         {torch.sqrt(var).item():.6f}")
    print(f"x_normed:    {x_normed}")
    print(f"normed mean: {x_normed.mean().item():.6f}  (should be ~0)")
    print(f"normed std:  {x_normed.std(unbiased=False).item():.6f}  (should be ~1)")
    print()

    # Compare with torch's LayerNorm (with default gamma=1, beta=0)
    ln = nn.LayerNorm(4)
    ln.eval()
    out_torch = ln(x)
    print(f"nn.LayerNorm(4)(x):  {out_torch}")
    print(f"matches our by-hand: {torch.allclose(x_normed, out_torch, atol=1e-5)}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/22_layernorm_by_hand.py
```

**Expected output (the trailing `999996` on `normed std` is float32 noise — see below):**

```text
x:           tensor([1., 2., 3., 4.])
mean:        2.500000
var:         1.250000
std:         1.118034
x_normed:    tensor([-1.3416, -0.4472,  0.4472,  1.3416])
normed mean: 0.000000  (should be ~0)
normed std:  0.999996  (should be ~1)

nn.LayerNorm(4)(x):  tensor([-1.3416, -0.4472,  0.4472,  1.3416],
       grad_fn=<NativeLayerNormBackward0>)
matches our by-hand: True
```

Two things to read off:

- **`normed std = 0.999996`, not exactly 1.** The $\varepsilon = 10^{-5}$ regulariser inside `(x - mean) / sqrt(var + eps)` makes the divisor slightly larger than $\sigma$, so the post-normalisation std is slightly smaller than 1 in float32. For any non-degenerate input the deviation is on the order of $\varepsilon / \sigma^2$ — invisible at GPT-2 scale, visible only when you print 6 decimals.
- **PyTorch's `nn.LayerNorm` agrees with our by-hand calculation** to within a few units in the last decimal (`torch.allclose(..., atol=1e-5) == True`). The only practical difference between the two is efficiency: `nn.LayerNorm` uses fused C++/CUDA kernels.

---

## 10.4 Building `mygpt.LayerNorm`

We add a hand-rolled `LayerNorm` to `mygpt`, mirroring the same pattern we used for `SingleHeadAttention` in Chapter 6 — implement it from scratch so the math is visible, and check that it agrees with PyTorch's built-in.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with the Chapter 9 file plus this new class** (and updated `main`):

```python
class LayerNorm(nn.Module):
    """Per-token layer normalisation with learnable scale and bias.

    Inputs:
        x: tensor of shape (..., embed_dim).

    Outputs:
        tensor of the same shape, with each (..., :)-slice normalised to
        mean 0 and (population) variance 1, then scaled by `weight` and
        shifted by `bias`.

    Parameters: weight (embed_dim,) and bias (embed_dim,), both
    initialised to ones and zeros respectively (so the module starts as
    a pure normalisation).
    """

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
```

(The rest of `__init__.py` is the Chapter 9 ending state — `VOCAB`, `to_ids`, `set_seed`, `TokenEmbedding`, `SingleHeadAttention`, `MultiHeadAttention`, `MLP` — all preserved. `LayerNorm` goes in after `MLP`.)

Then update **`main`** at the bottom of the file to:

```python
def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    ln = LayerNorm(C)
    mlp = MLP(embed_dim=C, dropout=0.0)
    ln.eval(); mlp.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    x_normed = ln(x)
    out = x + mlp(ln(x))   # GPT-2 pre-norm pattern: residual around mlp(ln(x))

    print(f"\nInput x       shape={tuple(x.shape)}")
    print(f"After LN(x)   shape={tuple(x_normed.shape)}")
    print(f"After residual+MLP+LN  shape={tuple(out.shape)}")
    print()

    # Per-token mean/std of normalised tensor
    print(f"LN(x) per-token means (4 positions): {ln(x).mean(dim=-1).flatten().tolist()}")
    print(f"LN(x) per-token stds  (4 positions): {ln(x).std(dim=-1, unbiased=False).flatten().tolist()}")
    print()

    n_te = sum(p.numel() for p in te.parameters())
    n_ln = sum(p.numel() for p in ln.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"TokenEmbedding parameters:   {n_te}")
    print(f"LayerNorm parameters:        {n_ln}  (= 2 * embed_dim)")
    print(f"MLP parameters:              {n_mlp}")
    print(f"Total parameters:            {n_te + n_ln + n_mlp}")
```

Run:

```bash
uv run mygpt
```

**Expected output (the per-token means are tiny float32 numbers near zero, not exact zeros; the per-token stds are ≈ 1 with a small float32 deviation from the same `eps` discussed above):**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4

Input x       shape=(1, 4, 4)
After LN(x)   shape=(1, 4, 4)
After residual+MLP+LN  shape=(1, 4, 4)

LN(x) per-token means (4 positions): [1.341104507446289e-07, 0.0, 1.4901161193847656e-08, 5.960464477539063e-08]
LN(x) per-token stds  (4 positions): [0.9999693036079407, 0.9999963641166687, 0.99998939037323, 0.9999875426292419]

TokenEmbedding parameters:   16
LayerNorm parameters:        8  (= 2 * embed_dim)
MLP parameters:              148
Total parameters:            172
```

Three things worth noticing:

- **Per-token means are not literally `0.0`.** They are float32 numbers like `1.34e-7`. Mathematically the mean of `(x - mean(x))` is zero; in float32 the subtraction has rounding error of order $\sigma \cdot 2^{-23} \sim 10^{-7}$ for our inputs. Treat the printed values as "zero up to float32 precision".
- **Per-token stds are ≈ 0.99996, not 1.00000.** Same `eps` story as in §10.3: the divisor is $\sqrt{\sigma^2 + \varepsilon}$, slightly larger than $\sigma$, giving a post-LN std slightly smaller than 1. Negligible at any practical scale.
- **`LayerNorm` adds 8 parameters** (2 × `embed_dim` = 2 × 4). That's `weight = ones(4)` and `bias = zeros(4)`. Tiny compared to the 148 of `MLP`. At GPT-2 scale, every layer norm contributes 1,536 parameters; with 25 layer norms in the model that is ~38k total — about 0.03% of the 124M total.

---

## 10.5 Pre-norm vs post-norm

There are two places to put the layer norm relative to the residual:

- **Post-norm** (original transformer, 2017): `x ← LN(x + sub(x))`. The normalisation is the *last* operation; the residual stream itself is renormalised at every block.
- **Pre-norm** (GPT-2 onwards): `x ← x + sub(LN(x))`. The normalisation is the *first* operation; only the input *to* the sub-layer is normalised; the residual stream accumulates magnitude.

The empirical consensus since around 2019 is that pre-norm is easier to train deeply — gradients flow more cleanly through the residual highway, because the residual is added *after* the normalisation rather than being normalised away. Every modern decoder-only transformer (GPT-2, GPT-3, Llama, etc.) uses pre-norm. We will follow GPT-2 and use **pre-norm** in Chapter 11.

The cost of pre-norm is exactly the drift we saw in §9.7 and will see attenuated in §10.6: the residual stream itself isn't normalised, so its scale grows over depth. Real models accept this — they add a final `LN` before the language-modelling head to renormalise once at the end.

---

## 10.6 Verifying that LayerNorm controls the §9.7 drift

The motivating claim of this chapter is empirical: insert a `LayerNorm` before each sub-layer in the §9.7 stack, and the activation drift should be reduced. Let's check.

**Save the following to** 📄 `experiments/23_layernorm_drift.py`:

```python
"""Experiment 23 — LayerNorm reduces the §9.7 residual drift.

Stacks 30 randomly-initialised MLPs with residual connections, with and
without a LayerNorm before each MLP. With pre-LN, the residual stream's
scale grows more slowly across depth.
"""

import torch

from mygpt import MLP, LayerNorm, set_seed


def main() -> None:
    set_seed(0)
    C = 16
    n_layers = 30
    mlps = [MLP(embed_dim=C, dropout=0.0).eval() for _ in range(n_layers)]
    lns = [LayerNorm(embed_dim=C).eval() for _ in range(n_layers)]

    x0 = torch.randn(1, 8, C)
    print(f"input std: {x0.std().item():.4f}")
    print()

    print("WITH residuals + pre-LayerNorm (x = x + mlp(ln(x))):")
    with torch.no_grad():
        x = x0.clone()
        for i, (mlp, ln) in enumerate(zip(mlps, lns)):
            x = x + mlp(ln(x))
            if i in (0, 4, 9, 14, 19, 29):
                print(f"  after layer {i+1:2d}: std = {x.std().item():.6f}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/23_layernorm_drift.py
```

**Expected output:**

```text
input std: 0.9369

WITH residuals + pre-LayerNorm (x = x + mlp(ln(x))):
  after layer  1: std = 0.992292
  after layer  5: std = 1.069066
  after layer 10: std = 1.146214
  after layer 15: std = 1.275711
  after layer 20: std = 1.454677
  after layer 30: std = 1.690578
```

Compare with §9.7 (residual but no LayerNorm): `2.21` at layer 30. With pre-LN the same stack ends at `1.69` — the drift is reduced by about 30% over 30 layers.

This is *not* a complete fix. Pre-norm normalises the *inputs* to each sub-layer, which keeps the linear/MLP layers operating at a stable scale, but it does not normalise the **residual stream** itself. If you want the residual stream to also stay at unit scale, you need post-norm (`LN(x + mlp(x))`), which keeps the post-block activations at exactly unit variance — at the cost of training stability for very deep models. GPT-2's choice of pre-norm trades a modest scale drift for cleaner gradients; we follow that choice.

---

## 10.7 Experiments

1. **The constant-input degenerate case.** Build a `ln = LayerNorm(4)` and call it on `x = torch.zeros(4)`. The mean is 0, the variance is 0 — without the $\varepsilon = 10^{-5}$ regulariser, division by zero would produce `nan`. With $\varepsilon$, the output is the (vanishing) numerator divided by $\sqrt{\varepsilon}$ — i.e., `0 / sqrt(eps) = 0`. Verify by running and confirming the output is `tensor([0., 0., 0., 0.])`.
2. **Weight and bias affect output linearly.** Build a `ln = LayerNorm(4)`. Set `ln.weight.data = torch.tensor([2.0, 2.0, 2.0, 2.0])` and `ln.bias.data = torch.tensor([1.0, 1.0, 1.0, 1.0])` (in eval mode). Run on `x = torch.tensor([1.0, 2.0, 3.0, 4.0])`. Predicted output: `(2 * normalised) + 1` — element-wise. The actual output is `tensor([-1.6833, 0.1056, 1.8944, 3.6833])`, matching `[2 * (-1.3416) + 1, 2 * (-0.4472) + 1, 2 * 0.4472 + 1, 2 * 1.3416 + 1]` to within float32 precision.
3. **Permutation-invariance of LayerNorm.** Like the MLP, layer norm is a per-token operation; it commutes with permutations of the time axis. Build `x1 = torch.randn(1, 4, 4)`, run `out1 = ln(x1)`, then permute and run `out2 = ln(x1[:, [3, 2, 1, 0], :])`. Verify `torch.equal(out2, out1[:, [3, 2, 1, 0], :])`.
4. **`nn.LayerNorm` agrees with ours.** In a Python session, build both `mygpt.LayerNorm(8)` and `torch.nn.LayerNorm(8)` (both default-init to `gamma=1, beta=0`). Apply both to the same random input. Use `torch.allclose(out_ours, out_torch, atol=1e-5)` to confirm. (`torch.equal` will fail because PyTorch's fused kernel uses a slightly different floating-point reduction order; `allclose` with `atol=1e-5` will succeed.)

After each experiment, restore the file you changed before moving on.

---

## 10.8 Exercises

1. **Why $\varepsilon$ matters.** In the §10.7 ex 1 case, what would be the LayerNorm output for `torch.zeros(4)` if we set $\varepsilon = 0$? (Answer: `nan` everywhere, because `0 / sqrt(0) = 0/0 = nan`.) Argue that for any non-degenerate input, increasing $\varepsilon$ from `0` to `1e-5` changes the output by a relative amount of order $\varepsilon / \sigma^2$, which is negligible.
2. **Parameter count.** A `LayerNorm(embed_dim)` has `2 * embed_dim` parameters. Compute this for GPT-2 small ($C = 768$). How many `LayerNorm` modules does GPT-2 have, given 12 transformer blocks (each with 2 layer norms, before attention and before MLP) plus 1 final layer norm? (Answer: `2 * 768 = 1,536` per layer norm; `12 * 2 + 1 = 25` layer norms; `25 * 1,536 ≈ 38,400 ≈ 0.04` M parameters.)
3. **Pre-norm vs post-norm for a single block.** Build a small example with `x = torch.randn(1, 4, 4)`, `ln = LayerNorm(4)`, `mlp = MLP(4)` (eval mode for both). Compute pre-norm `out_pre = x + mlp(ln(x))` and post-norm `out_post = ln(x + mlp(x))`. Argue, in your own words, why the post-norm output's *individual tokens* will all have mean 0 and std ≈ 1 (because `ln` is the last operation), while the pre-norm output's tokens *won't*.
4. **Why `unbiased=False`?** PyTorch's `tensor.var()` defaults to `unbiased=True` (divisor $n - 1$, Bessel's correction). LayerNorm uses `unbiased=False` (divisor $n$, the population variance). Why? (Hint: think about what happens when $C = 1$. The unbiased variance of one number is undefined; the population variance is 0, which combined with $\varepsilon$ keeps the math finite.)

---

## 10.9 What's next

We have all three pieces of the transformer block: multi-head causal self-attention (Chapters 6–8), the position-wise MLP (Chapter 9), and layer normalisation (this chapter). Chapter 11 wires them up:

```python
# A pre-norm GPT-2 transformer block:
def block(x):
    x = x + mha(ln1(x))      # residual around attention
    x = x + mlp(ln2(x))      # residual around MLP
    return x
```

That is the body of one GPT-2 layer. After Chapter 11 we just need position embeddings (Chapter 12) and a language-modelling head (Chapter 12) to have a complete GPT-2.

> **Looking ahead — what to remember from this chapter**
>
> 1. Layer norm normalises every token's vector to mean 0, std ≈ 1 over the channel axis — *per token*, *not* across the batch.
> 2. The learnable `weight` and `bias` are shape `(embed_dim,)`, contributing `2 * embed_dim` parameters.
> 3. GPT-2 uses **pre-norm**: `x = x + sub(LN(x))`. The residual stream's scale drifts (we measured 1.69 over 30 layers, vs 2.21 without LN), but gradients flow cleanly.
> 4. `mygpt.LayerNorm` is a hand-rolled module that matches `nn.LayerNorm` to float-32 precision (`torch.allclose(..., atol=1e-5) == True`).

On to [Chapter 11 — Putting it together: the transformer block](11_transformer_block.md) *(coming soon)*.
