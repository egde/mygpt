---
title: 13. The forward pass with loss
nav_order: 14
parent: LLM Fundamentals
---

# Chapter 13 — The forward pass with loss

After Chapter 12 we have a model that turns token ids into logits. To **train** the model we need the other side of Chapter 4's gradient-descent recipe: a scalar **loss** to minimise. For language modelling that loss is **cross-entropy** between the model's predicted next-token distribution and the actual next token — exactly the cross-entropy of Ch.4 §4.10's promise, finally cashed in.

By the end you will have:

- understood the **next-token prediction** task: given tokens $0, 1, \ldots, t$, predict token $t+1$,
- met **cross-entropy** loss and computed it by hand on one $(logits, target)$ pair,
- understood the **target shift** trick: input is the sequence; targets are the same sequence shifted left by one,
- updated `mygpt.GPT.forward(ids, targets=None)` to return both logits and (optionally) loss,
- watched a random-init GPT-2-small-shaped model produce a loss of about `4.26` on the running example, and understood why it is higher than the ideal $\log V \approx 1.39$.

There is one small new concept (cross-entropy) and a useful new convention (the `(logits, loss)` return tuple). After this chapter we are exactly one Chapter 4-style training loop away from a *trained* model.

---

## 13.1 The next-token prediction task

Recall §1.5: a language model factors $P(x_1, \ldots, x_T)$ as

$$
P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdot \ldots \cdot P(x_T \mid x_1, \ldots, x_{T-1}).
$$

Training optimises every conditional simultaneously. For our model, after running `forward(ids)` we have **logits of shape `(B, T, V)`** — one length-$V$ logit vector per `(b, t)` position. The contract:

- `logits[b, t, :]` is the model's prediction of the **next** token after `ids[b, 0..t]` — that is, of `ids[b, t+1]`.

So position $t = 0$ predicts position 1, position 1 predicts position 2, …, position $T-1$ predicts position $T$. Position $T$ has no prediction target — it would predict the (non-existent) $(T+1)$-th token.

The cleanest way to feed this into a loss is to **shift**: input is `ids[:, :-1]` (length $T-1$), targets are `ids[:, 1:]` (length $T-1$, the same tokens shifted left by one). Both tensors then have shape `(B, T-1)`, and every `logits[b, t, :]` has a matching `target[b, t]`.

For our running example `"I love AI !"` = `[0, 1, 2, 3]` (length 4):

- input: `[0, 1, 2]`     ← `"I love AI"`     (length 3)
- targets: `[1, 2, 3]` ← `"love AI !"`   (length 3)

Position 0 of the input is `0` (`"I"`); the target at position 0 is `1` (`"love"`) — the model is learning that `"love"` is a likely next token after `"I"`. And so on for the other two positions.

---

## 13.2 Setup

This chapter assumes you finished Chapter 12 — `mygpt/` exists with the `GPT` class.

If you skipped Chapter 12, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 12 ending state from `docs/_state_after_ch12.md`.

You are ready.

---

## 13.3 Cross-entropy loss

Suppose the model produces logit vector $\mathbf{z} \in \mathbb{R}^V$ for one position, and the true next token is index $y \in \{0, \ldots, V-1\}$. The cross-entropy loss is

$$
\mathcal{L}(\mathbf{z}, y) \;=\; -\log\!\Big(\frac{e^{z_y}}{\sum_{j=0}^{V-1} e^{z_j}}\Big) \;=\; -\log \text{softmax}(\mathbf{z})_y \;=\; -z_y + \log \sum_{j=0}^{V-1} e^{z_j}.
$$

Three properties to read off:

- **Non-negative.** Softmax outputs are in $(0, 1]$, so $-\log \text{softmax}$ is in $[0, \infty)$.
- **Zero only when $\text{softmax}(\mathbf{z})_y = 1$.** That requires the model to put *all* its mass on the correct token — perfect prediction.
- **Equals $\log V$ when the logits are all equal.** Uniform softmax gives $\frac{1}{V}$ at every index, so $-\log \frac{1}{V} = \log V$. For our $V = 4$ that is $\log 4 \approx 1.386$ — the **random-guessing baseline**.

For a batch of predictions, we average the per-position losses:

$$
\mathcal{L}_\text{batch} \;=\; \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(\mathbf{z}_i, y_i),
$$

where $N = B \cdot (T-1)$ is the number of (input, target) pairs in the batch.

PyTorch ships this as `F.cross_entropy(logits, targets)` and applies the softmax internally — for numerical stability, the log-sum-exp form on the right is computed in a numerically-stable way that never explicitly forms `softmax`. The expected input shapes are:

- `logits`: shape `(N, V)` — each row is the logit vector for one prediction.
- `targets`: shape `(N,)` — each entry is the true class index.

Our `(B, T, V)` logits and `(B, T)` targets need to be flattened: `logits.view(B*T, V)` and `targets.view(B*T)`.

A worked example. Take logits $\mathbf{z} = (1.0, 2.0, 0.5, -1.0)$ and target $y = 1$:

$$
\begin{aligned}
e^{1.0} \approx 2.7183, \;\; e^{2.0} \approx 7.3891, \;\; e^{0.5} \approx 1.6487, \;\; e^{-1.0} \approx 0.3679 \\
\sum e^{z_j} \approx 12.1240, \qquad \text{softmax}(\mathbf{z})_1 \approx 7.3891 / 12.1240 \approx 0.6095 \\
\mathcal{L} \;=\; -\log(0.6095) \;\approx\; 0.4952.
\end{aligned}
$$

We will reproduce that with `F.cross_entropy` next.

**Save the following to** 📄 `experiments/26_cross_entropy_by_hand.py`:

```python
"""Experiment 26 — Cross-entropy by hand on (logits=(1.0, 2.0, 0.5, -1.0), target=1)."""

import math

import torch
import torch.nn.functional as F


def main() -> None:
    logits = torch.tensor([1.0, 2.0, 0.5, -1.0])
    target = torch.tensor(1)

    # By hand
    exp_logits = torch.exp(logits)
    sum_exp = exp_logits.sum().item()
    softmax_target = exp_logits[target].item() / sum_exp
    loss_by_hand = -math.log(softmax_target)

    print(f"logits:                   {logits}")
    print(f"target:                   {target.item()}")
    print(f"exp(logits):              {exp_logits}")
    print(f"sum exp:                  {sum_exp:.6f}")
    print(f"softmax[target]:          {softmax_target:.6f}")
    print(f"-log(softmax[target]):    {loss_by_hand:.6f}")
    print()

    # By F.cross_entropy
    loss_torch = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    print(f"F.cross_entropy:          {loss_torch.item():.6f}")
    print(f"matches by-hand:          {abs(loss_by_hand - loss_torch.item()) < 1e-6}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/26_cross_entropy_by_hand.py
```

**Expected output:**

```text
logits:                   tensor([ 1.0000,  2.0000,  0.5000, -1.0000])
target:                   1
exp(logits):              tensor([2.7183, 7.3891, 1.6487, 0.3679])
sum exp:                  12.123940
softmax[target]:          0.609460
-log(softmax[target]):    0.495182

F.cross_entropy:          0.495182
matches by-hand:          True
```

Two things to read off:

- **The loss is `0.4952`, not zero.** The model assigns 60.95% probability to the correct token. Since softmax of `2.0` (the biggest logit) does not equal `1.0`, the loss is non-zero. Perfect prediction would require the logit at the target to be infinitely larger than the others.
- **PyTorch's loss equals our by-hand computation.** `F.cross_entropy` applies log-softmax internally; we did the same arithmetic explicitly.

---

## 13.4 Updating `GPT.forward` to return loss

The standard convention in nanoGPT-style code is for `forward` to take an optional `targets` argument. If `targets is None`, return only the logits (this is how you call it at *generation* time, when there are no labels). If `targets is not None`, also return the cross-entropy loss.

Replace the `forward` method of `GPT` with:

```python
def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute logits and (optionally) cross-entropy loss.

    Inputs:
        ids:     long tensor of shape (B, T) with values in [0, vocab_size).
        targets: optional long tensor of shape (B, T); if given, every
                 logits[b, t, :] is scored against targets[b, t] via
                 cross-entropy.

    Outputs:
        (logits, loss). `loss` is None if `targets` is None.
    """
    B, T = ids.shape
    if T > self.max_seq_len:
        raise ValueError(
            f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
        )

    positions = torch.arange(T, device=ids.device)
    x = self.token_embedding(ids) + self.position_embedding(positions)
    x = self.embed_drop(x)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = x @ self.token_embedding.embedding.weight.T  # (B, T, V)

    if targets is None:
        return logits, None

    # F.cross_entropy expects (N, V) logits and (N,) targets.
    loss = F.cross_entropy(
        logits.view(B * T, -1),
        targets.view(B * T),
    )
    return logits, loss
```

(You also need to add `import math` at the top of `__init__.py` if it's not already there — it is, from Chapter 6.)

Then update `main` to demonstrate the (input, targets) → (logits, loss) flow:

```python
def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C, h, N = len(VOCAB), 4, 2, 2
    gpt = GPT(vocab_size=V, embed_dim=C, num_heads=h, num_layers=N,
              max_seq_len=64, dropout=0.0)
    gpt.eval()

    # Next-token prediction setup: input is the first T-1 tokens, targets
    # are the same sequence shifted left by 1.
    full = to_ids(["I", "love", "AI", "!"])
    input_ids = full[:-1].unsqueeze(0)   # (1, 3) = [[0, 1, 2]]   = "I love AI"
    targets = full[1:].unsqueeze(0)      # (1, 3) = [[1, 2, 3]]   = "love AI !"

    logits, loss = gpt(input_ids, targets)
    print(f"\nInput ids (B, T):    {tuple(input_ids.shape)}  {input_ids.tolist()}")
    print(f"Targets   (B, T):    {tuple(targets.shape)}  {targets.tolist()}")
    print(f"Logits    (B, T, V): {tuple(logits.shape)}")
    print(f"Loss:                {loss.item():.4f}")

    import math as _math
    print(f"\nReference: log(V) = log({V}) = {_math.log(V):.4f}")
    print("(Random-init loss is typically a small multiple of log(V); training drives it down.)")
```

(The `import math as _math` inside `main` avoids shadowing the outer `math` import if you have `from mygpt import math` elsewhere — a defensive paranoia, not strictly needed here.)

Run it:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4

Input ids (B, T):    (1, 3)  [[0, 1, 2]]
Targets   (B, T):    (1, 3)  [[1, 2, 3]]
Logits    (B, T, V): (1, 3, 4)
Loss:                4.2588

Reference: log(V) = log(4) = 1.3863
```

Three things to read off:

- **Loss = 4.2588 at random init.** That is *higher* than the random-guess baseline of `log(V) = 1.39`. Why? A randomly-initialised network with N(0,1) weights produces logits of moderate magnitude, and softmax on moderate logits puts disproportionate mass on whichever logit happens to be biggest. If the biggest logit happens to be at the *wrong* token (which it usually is, since training has not occurred), the model is *confidently wrong*, and confident-wrongness gives loss > log(V). With weight initialisation tuned to start the network closer to identity (which real implementations do), the random-init loss is closer to log(V).
- **`logits.shape = (1, 3, 4)`.** Three positions of prediction (the input length), each producing a length-4 logit vector. After softmax the row sums would be 1; after argmax, each row would be the model's most-likely next token at that position.
- **`loss` is a scalar.** `F.cross_entropy` returns a single number — the average over the $B \cdot T = 3$ pairs. This is the scalar gradient descent will minimise in Chapter 14.

---

## 13.5 The `forward` contract: what does and doesn't return loss

Two regimes:

| Use case               | Call                                  | Returns          |
|------------------------|---------------------------------------|------------------|
| Inference / generation | `logits, _ = gpt(input_ids)`          | `(logits, None)` |
| Training               | `logits, loss = gpt(input_ids, targets)` | `(logits, loss)` |

This pattern appears in every nanoGPT-style codebase. The key reason for the optional `targets`: at generation time, you don't have a target — you are *creating* the next token, not comparing against a known one. Returning `None` for `loss` lets the same forward function serve both regimes.

A subtle nuance: because Python's `tuple` unpacking is positional, you can write `logits, loss = gpt(ids)` even when `targets` is `None` — the second element is `None`, which assigns fine. This is the canonical idiom.

---

## 13.6 Experiments

1. **Confirm uniform-logits gives `log(V)`.** In a Python session, set `logits = torch.zeros(1, 3, 4)` and `targets = torch.tensor([[1, 2, 3]])`. Compute `loss = F.cross_entropy(logits.view(3, 4), targets.view(3))`. The result should be exactly `log(4) ≈ 1.3863` — uniform logits are exactly the random-guess baseline.
2. **A fully-confident correct prediction has loss 0.** Set `logits = torch.tensor([[[0.0, 100.0, 0.0, 0.0]]])` and `targets = torch.tensor([[1]])`. Compute the loss. The answer is essentially 0 (because softmax of a +100 logit is essentially 1.0).
3. **A fully-confident wrong prediction has very high loss.** Same logits as above, but `targets = torch.tensor([[0]])` (target is the position with logit 0, not the position with logit 100). The loss is approximately 100 — the negative log of the tiny softmax probability assigned to the wrong target. Cross-entropy *punishes* over-confidence on wrong predictions strongly.
4. **The forward `targets=None` regime.** Call `gpt(input_ids)` (without targets). The return is `(logits, None)`. Extract just logits with `logits, _ = gpt(input_ids)`. This is what generation will look like in Chapter 15.

After each experiment, restore the file you changed before moving on (only exp 4 modifies anything, and only ephemerally).

---

## 13.7 Exercises

1. **The mean over what?** When we call `F.cross_entropy(logits.view(B*T, V), targets.view(B*T))`, PyTorch averages over all $B \cdot T$ predictions — *both* the batch and the time axes. Why does that matter for batch-size invariance? (Hint: if instead PyTorch summed, doubling the batch size would double the loss, and gradients would be twice as big — effectively halving the learning rate.)
2. **Why log-sum-exp instead of explicit softmax?** PyTorch's `F.cross_entropy` does not actually compute `softmax(logits)` and then `log` it — it computes `-z_y + log(sum exp(z))` directly via the log-sum-exp identity. Argue why this is more numerically stable, by considering what happens for $z_j = 1000$. (Hint: `exp(1000)` overflows; `log(sum exp(z))` doesn't, when computed via the trick.)
3. **What is the highest possible loss?** For a $V$-class cross-entropy where the model puts probability $\epsilon$ on the true class, the loss is $-\log \epsilon$. As $\epsilon \to 0$, the loss $\to \infty$. There is no upper bound. Argue informally why this property is useful for training (gradient signal is strongest where the model is most wrong).
4. **The shift trick formalised.** Given a sequence `tokens` of length $T$, write the input/target slicing in Python (`tokens[:-1]` and `tokens[1:]`). Confirm both have length $T-1$. What is the relationship between the input at position $t$ and the target at position $t$? (Answer: target at position $t$ is `tokens[t+1]`; the model is predicting the token at position $t+1$ from the prefix `tokens[0..t]`.)

---

## 13.8 What's next

We have a model and a loss. Chapter 14 closes the loop: a Chapter-4-style training loop applied to a real text file. We will

1. tokenise the file into a `(N,)` long tensor of ids,
2. sample random `(input, target)` pairs of length `(B, T)` from it,
3. run `logits, loss = gpt(input, target)`, `loss.backward()`, `optimizer.step()`,
4. watch the loss decrease over many iterations.

After Chapter 14 we have a *trained* `mygpt.GPT` — and Chapter 15 will sample text from it.

> **Looking ahead — what to remember from this chapter**
>
> 1. Cross-entropy on logits and an integer target is `-z_y + log(sum exp(z))`. PyTorch ships it as `F.cross_entropy`, applying log-sum-exp for numerical stability.
> 2. The *uniform-logits baseline* is `log(V)`. A random-init network typically produces a loss several times higher because it is confidently wrong on most positions.
> 3. The next-token shift: input is `tokens[:-1]`, targets are `tokens[1:]`. Position $t$ predicts position $t+1$.
> 4. `mygpt.GPT.forward(ids, targets=None)` returns `(logits, loss)` where `loss` is `None` for inference and a scalar for training.

On to [Chapter 14 — Training loop: gradient descent in practice](14_training_loop.md) *(coming soon)*.
