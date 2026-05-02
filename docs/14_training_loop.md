---
title: 14. Training loop
nav_order: 15
parent: LLM Fundamentals
---

# Chapter 14 — Training loop: gradient descent in practice

Eleven chapters back in §4.5 we saw the four-line training loop: `zero_grad`, forward, `backward`, `step`. Twelve chapters back in §1.10 we promised that the package would, by the end, train and generate text. This chapter cashes in.

By the end you will have:

- built a tiny corpus by repeating the running example `"I love AI !"` 64 times,
- met `get_batch(data, B, T)` — the canonical sampler that draws random `(input, target)` pairs from a 1-D token tensor,
- met **AdamW**, the optimiser modern transformers use; understood why it works better than SGD for these models,
- run a 200-step training loop on a tiny GPT and watched the loss drop from `5.27` to `0.0035` (well below `log(V) = 1.39`),
- verified the trained model gives the right `argmax` next-token prediction at every step of the cycle.

After this chapter we have a *trained* `mygpt.GPT`. Chapter 15 generates text from it.

---

## 14.1 What we are about to do

Recap of the recipe:

1. **Data.** Turn text into a 1-D `long` tensor of token ids. We have `mygpt.to_ids` for that already.
2. **Sampler.** A function `get_batch(data, B, T)` that returns random `(B, T)` (input, target) pairs from the data tensor. Every step of training calls this once.
3. **Model.** `mygpt.GPT(vocab_size, embed_dim, num_heads, num_layers)`. Returns `(logits, loss)` from `forward(ids, targets)`.
4. **Optimiser.** `torch.optim.AdamW(model.parameters(), lr=1e-2)`. AdamW is a per-parameter, momentum-aware variant of SGD; modern transformers use it almost universally.
5. **Loop.** For each step: `get_batch` → `zero_grad` → `forward` → `backward` → `step`. Print the loss occasionally.

We will run 200 steps on a small `GPT(V=4, C=8, h=2, N=2)` and watch it memorise the cycle `"I love AI ! I love AI ! ..."`. With only 4 distinct tokens in a periodic pattern, the model has just enough latent capacity to fit the data, and we can drive the loss to essentially zero.

---

## 14.2 Setup

This chapter assumes you finished Chapter 13 — `mygpt/` exists with `GPT.forward(ids, targets=None)` returning `(logits, loss)`.

If you skipped Chapter 13, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 13 ending state from `docs/_state_after_ch13.md`.

You are ready.

---

## 14.3 The corpus and the batch sampler

The simplest possible "training corpus" we can build with `mygpt.VOCAB`: take `"I love AI !"`, repeat it 64 times to get 256 tokens.

```python
data = torch.tensor([0, 1, 2, 3] * 64, dtype=torch.long)
# data.shape == (256,)
```

This is a *strongly periodic* corpus. The next-token-prediction task is: at every position $t$, predict the token that comes one step later in the cycle. The four "rules" the model must learn are:

- after `0` (`"I"`) comes `1` (`"love"`),
- after `1` (`"love"`) comes `2` (`"AI"`),
- after `2` (`"AI"`) comes `3` (`"!"`),
- after `3` (`"!"`) comes `0` (`"I"`) — wrapping around.

A minimal corpus, but enough to test the whole pipeline.

The **batch sampler** picks `B` random starting positions and slices `(B, T+1)` chunks; the first `T` tokens are the input, the last `T` (shifted by 1) are the targets:

```python
def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len]   for i in ix])    # (B, T)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])  # (B, T) shifted
    return x, y
```

The `len(data) - seq_len - 1` upper bound on `ix` ensures we never run past the end of the data when slicing `i+seq_len+1`.

This is the same sampler used by Karpathy's nanoGPT — it is the canonical pattern for transformer training on a flat token stream.

We add `get_batch` to `mygpt` so future chapters can re-use it.

**Append the following function to** 📄 `src/mygpt/__init__.py` (after `set_seed`, before `TokenEmbedding`):

```python
def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, target) pairs from a 1-D token tensor.

    Inputs:
        data: 1-D long tensor of token ids (the entire corpus).
        batch_size: B, the number of independent sequences in the batch.
        seq_len: T, the length of each sequence.

    Outputs:
        x: (B, T) long tensor — the input ids.
        y: (B, T) long tensor — the same ids shifted left by one (targets).
    """
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y
```

A small experiment to see what `get_batch` returns.

**Save the following to** 📄 `experiments/27_corpus_and_batch.py`:

```python
"""Experiment 27 — Build a tiny corpus and sample one batch."""

import torch

from mygpt import VOCAB, get_batch, set_seed


def main() -> None:
    set_seed(0)

    # Build a corpus by repeating the running example 64 times
    data = torch.tensor([VOCAB.index(t) for t in (["I", "love", "AI", "!"] * 64)], dtype=torch.long)
    print(f"Corpus length: {len(data)} tokens")
    print(f"First 12 tokens: {data[:12].tolist()}")
    print()

    # One batch
    set_seed(42)
    x, y = get_batch(data, batch_size=4, seq_len=8)
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print()
    print("x (first row):", x[0].tolist())
    print("y (first row):", y[0].tolist())
    print("(y is x shifted left by one — the next-token target.)")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/27_corpus_and_batch.py
```

**Expected output:**

```text
Corpus length: 256 tokens
First 12 tokens: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

x shape: (4, 8)
y shape: (4, 8)

x (first row): [0, 1, 2, 3, 0, 1, 2, 3]
y (first row): [1, 2, 3, 0, 1, 2, 3, 0]
(y is x shifted left by one — the next-token target.)
```

You can read off the cycle: every entry of `y[0]` is `(x[0] + 1) % 4`. The model's job is to learn that rule.

---

## 14.4 AdamW: the optimiser transformers actually use

In Chapter 4 we used `torch.optim.SGD`, vanilla gradient descent. SGD works for the toy quadratic and the linear-fit problem, but for transformers it converges *much* slower than the modern alternative: **AdamW** (Loshchilov & Hutter 2019).

AdamW differs from SGD in three ways. We won't derive any of it; the practical takeaway is what matters:

1. **Per-parameter step size.** AdamW maintains a running estimate of the *magnitude* of each parameter's gradient and divides the update by that magnitude. Big-gradient parameters get small steps; small-gradient parameters get bigger steps. The single global learning rate `lr` is more like a cap than a fixed step size.
2. **Momentum.** AdamW also maintains an exponentially-weighted average of recent gradients, and updates parameters using that average instead of the latest gradient. This smooths out the trajectory and helps the optimiser cross flat regions.
3. **Weight decay.** AdamW directly shrinks every parameter toward zero by a small factor each step. This is a regularisation technique that prevents weights from drifting too large.

Concretely: on the §14.5 loss curve below, AdamW reaches loss < 0.02 by step 100 where SGD takes hundreds more (you'll verify this in §14.7 exp 1). The per-parameter step size and the momentum are doing most of that work.

The PyTorch API is identical to `SGD`:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
```

For our tiny problem we can use `weight_decay=0.0` to keep the demonstration cleanest. Real transformers typically use `weight_decay=0.1`.

(All three SGD failure modes from §4.8 — too-high lr diverges, no zero_grad accumulates, etc — apply equally to AdamW. AdamW is *better*, not magical.)

---

## 14.5 Putting it together: the training loop

The full loop, in 14 lines:

```python
set_seed(0)
model = GPT(vocab_size=4, embed_dim=8, num_heads=2, num_layers=2,
            max_seq_len=64, dropout=0.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
data = torch.tensor([0, 1, 2, 3] * 64, dtype=torch.long)

set_seed(42)
B, T = 4, 8
for step in range(1, 201):
    x, y = get_batch(data, B, T)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if step in (1, 10, 50, 100, 200):
        print(f"step {step:>3}: loss = {loss.item():.4f}")
```

Eight of those lines are setup; the loop body is exactly the four lines from §4.5: `zero_grad`, forward, `backward`, `step`. The `if step in ...` is just for printing.

A naming note: each pass through the loop is a **step** (one `optimizer.step()` call). Chapter 4 used **iteration** for the same idea; transformer-training literature prefers "step", so we use that from here on.

**Save the following to** 📄 `experiments/28_train_gpt.py`:

```python
"""Experiment 28 — Train a tiny GPT on a repeating-cycle corpus.

200 AdamW steps with B=4, T=8. Loss drops from ~5.27 (random init,
confidently wrong) to ~0.0035 (essentially memorised the cycle).
"""

import math

import torch

from mygpt import GPT, get_batch, set_seed


def main() -> None:
    set_seed(0)
    model = GPT(
        vocab_size=4,
        embed_dim=8,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.tensor([0, 1, 2, 3] * 64, dtype=torch.long)

    set_seed(42)
    B, T = 4, 8
    print(f"Training on a length-{len(data)} corpus, batch_size={B}, seq_len={T}")
    print(f"Reference: log(V) = log(4) = {math.log(4):.4f}\n")

    for step in range(1, 201):
        x, y = get_batch(data, B, T)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step in (1, 10, 50, 100, 200):
            print(f"step {step:>3}: loss = {loss.item():.4f}")

    # Save the trained model so Chapter 15 can load it
    torch.save(model.state_dict(), "trained_gpt.pt")
    print("\nSaved trained model to trained_gpt.pt")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/28_train_gpt.py
```

**Expected output:**

```text
Training on a length-256 corpus, batch_size=4, seq_len=8
Reference: log(V) = log(4) = 1.3863

step   1: loss = 5.2739
step  10: loss = 1.6107
step  50: loss = 0.3453
step 100: loss = 0.0162
step 200: loss = 0.0035

Saved trained model to trained_gpt.pt
```

Read this row by row:

- **Step 1: loss = 5.27.** Random-init, way above `log(V) = 1.39`. The model is *confidently wrong* (recall §13.4).
- **Step 10: loss ≈ 1.61.** After 10 batches of gradient descent, the model has stopped being confidently wrong. It is around the random-guessing baseline `log(V) = 1.39` — i.e. its predictions are about as informative as picking a token at random. The next 40 steps will teach it the cycle.
- **Step 50: loss ≈ 0.35.** A factor-of-4 reduction below random guessing. The model is now beginning to lock onto the periodic structure.
- **Step 100: loss ≈ 0.016.** Two orders of magnitude below random guessing. The model is essentially predicting the right next token most of the time.
- **Step 200: loss ≈ 0.004.** Three orders of magnitude below random guessing — effectively memorised. Further training would push the loss closer to 0 but with diminishing returns.

That is what *training a transformer* looks like in numbers. A real GPT-2 training run does the same thing on a much bigger corpus, with millions of steps, and a loss curve that looks qualitatively similar — sharp drop early, slow grind later.

(`trained_gpt.pt` is a bare `state_dict` — just the model's weights. Chapter 18 will replace this with a self-contained `.ckpt` that bundles the tokenizer and architecture config alongside the weights, so a single file is enough to reload.)

---

## 14.6 Verifying that the model has learned the cycle

The loss has dropped to `0.0035`. To make that concrete, let's check that `argmax(logits)` at every position predicts the *correct* next token.

**Save the following to** 📄 `experiments/29_eval_trained.py`:

```python
"""Experiment 29 — Verify the trained GPT predicts the cycle correctly.

For every prefix of the cycle, computes argmax(logits[-1]) and compares
to the true next token.
"""

import torch

from mygpt import GPT, VOCAB, set_seed


def main() -> None:
    # Re-build a fresh model with the same architecture and load the trained
    # weights from experiment 28.
    set_seed(0)  # only affects model init, which load_state_dict overwrites
    model = GPT(
        vocab_size=4,
        embed_dim=8,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
        dropout=0.0,
    )
    model.load_state_dict(torch.load("trained_gpt.pt"))
    model.eval()

    print("Trained model's argmax next-token prediction:\n")
    for L in range(1, 9):
        # Build a length-L prefix of the cycle starting from token 0
        prefix = torch.tensor([(i % 4) for i in range(L)]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(prefix)
        pred = logits[0, -1, :].argmax().item()
        actual = (prefix[0, -1].item() + 1) % 4
        ok = "OK" if pred == actual else "WRONG"
        print(f"  prefix={prefix[0].tolist()!s:<28}  predicted={VOCAB[pred]!r:<7}  expected={VOCAB[actual]!r:<7}  {ok}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/29_eval_trained.py
```

**Expected output:**

```text
Trained model's argmax next-token prediction:

  prefix=[0]                           predicted='love'   expected='love'   OK
  prefix=[0, 1]                        predicted='AI'     expected='AI'     OK
  prefix=[0, 1, 2]                     predicted='!'      expected='!'      OK
  prefix=[0, 1, 2, 3]                  predicted='I'      expected='I'      OK
  prefix=[0, 1, 2, 3, 0]               predicted='love'   expected='love'   OK
  prefix=[0, 1, 2, 3, 0, 1]            predicted='AI'     expected='AI'     OK
  prefix=[0, 1, 2, 3, 0, 1, 2]         predicted='!'      expected='!'      OK
  prefix=[0, 1, 2, 3, 0, 1, 2, 3]      predicted='I'      expected='I'      OK
```

Every row is `OK`. The trained model has learned the cycle perfectly.

This is the moment the tutorial's promise from §1.10 becomes reality: a Python package with $2{,}240$ parameters, trained for 200 steps on 256 tokens, reliably predicts the next token in the running example.

---

## 14.7 Experiments

1. **AdamW vs SGD.** In `experiments/28_train_gpt.py`, change `torch.optim.AdamW(...)` to `torch.optim.SGD(model.parameters(), lr=1e-1)`. (We bump `lr` from `1e-2` to `1e-1` because SGD takes bigger steps when there is no per-parameter scaling.) Re-run. SGD also converges, but more slowly: at step 50 loss is around 0.76 (vs AdamW's 0.35), and at step 200 around 0.008 (vs AdamW's 0.0035). On bigger transformers the gap is much larger.
2. **Crank the learning rate.** Set AdamW `lr=1.0` (100× the default). Re-run. The loss spikes upward in the first few steps (around `8.16` at step 10, *higher* than the random init at step 1 — the optimiser is overshooting on every step), then settles at roughly `log(V) = 1.39` for the rest of training. The model has been knocked off its random initialisation and cannot find its way back to the periodic pattern. AdamW does *not* immune the model to a bad LR; it just makes the choice less brittle than SGD.
3. **Batch size matters less than you think.** Set `B=1` (one example per step). Re-run. The loss at step 200 is still close to zero — the model trains fine on a smaller batch, just with noisier gradient estimates. Now try `B=16` (4× larger). Same end-state. For our tiny problem batch size only affects gradient noise, not whether training converges.
4. **Restore the 4-line loop.** Replace `experiments/28_train_gpt.py`'s loop body with literally:

   ```python
   optimizer.zero_grad()
   _, loss = model(x, y)
   loss.backward()
   optimizer.step()
   ```

   Confirm the loss output is unchanged. The four lines are the entire training step; everything else around them is bookkeeping.

After each experiment, restore the file you changed before moving on.

---

## 14.8 Exercises

1. **Why subtract `seq_len + 1` in `get_batch`?** The line `torch.randint(0, len(data) - seq_len - 1, ...)` ensures we never read past the end. Argue that without the `-1` term the slicing `data[i+1 : i+seq_len+1]` could go out of bounds when `i = len(data) - seq_len - 1`.
2. **Loss vs perplexity.** A common metric reported alongside loss is **perplexity** = `exp(loss)`. Argue that a perplexity of 4 corresponds to "the model is as confused as if it were uniformly guessing among 4 tokens" (because `log(4) = 1.386` is the loss at uniform softmax, and `exp(log(4)) = 4`). Compute the perplexity of our step-200 loss `0.0035`. (Answer: about `1.0035` — almost as good as the perfect `1.0`.)
3. **What if the corpus had 5 distinct tokens in a cycle of period 5?** Set up a corpus `data = torch.tensor([0,1,2,3,4] * 64, dtype=torch.long)` and train a `GPT(vocab_size=5, ...)`. The cycle is now period 5; can a 2-block, embed-dim-8 model still learn it? (Empirically yes, but you may need more steps.)
4. **The training data fits the model — what's missing?** Our 2240-parameter model "memorised" 256 tokens. With more parameters than data, this is *overfitting* in the strict sense. Why does it not matter for this demo? (Answer: because the data is *the* generator of the language we are trying to model. There is no separate "test set" with unseen patterns. In real LLM training, the corpus is a sample of a much larger distribution, and overfitting *would* hurt — which is why we use dropout, weight decay, and held-out evaluation sets.)

---

## 14.9 What's next

The model trains. The next step is to **generate** text from it: given a prompt, repeatedly sample the next token from the model's predicted distribution, append it to the prompt, and continue until we hit a stop condition.

Chapter 15 builds the generation loop. We will see:

- the autoregressive sampling pattern,
- **temperature** (a knob on the softmax to make sampling more or less greedy),
- **top-k sampling** (truncating the distribution to its top-k entries),
- a `mygpt.generate(model, prompt_ids, max_new_tokens, ...)` function.

Then we have a full generation pipeline: `mygpt` package + trained model → text out.

> **Looking ahead — what to remember from this chapter**
>
> 1. The training loop is the same four-line ritual from Ch.4. What changes is the optimiser (AdamW), the model (GPT), and the data sampling (`get_batch`).
> 2. AdamW has per-parameter step sizes, momentum, and weight decay. It is faster than SGD on transformers and is the default for nearly every modern training run.
> 3. Loss starts high (random init can be far above `log(V)`), drops past `log(V)` quickly, then grinds toward zero on a learnable corpus.
> 4. With our 2240-parameter `GPT(V=4, C=8, h=2, N=2)` and 200 AdamW steps, the loss on the cycle corpus drops from `5.27` to `0.0035` — and `argmax` predicts every token in the cycle correctly.
> 5. `mygpt.get_batch(data, B, T)` is the canonical sampler we will use in Chapter 15 too.

On to [Chapter 15 — Generation: sampling text from a trained model](15_generation.md) *(coming soon)*.
