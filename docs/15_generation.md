---
title: 15. Generation
nav_order: 16
parent: LLM Fundamentals
---

# Chapter 15 — Generation: sampling text from a trained model

After Chapter 14 we have a *trained* `mygpt.GPT`. To turn it into a text generator we need a small loop that:

1. runs the model on the current sequence,
2. picks one token from the model's predicted next-token distribution,
3. appends that token to the sequence,
4. repeats until we hit a stop condition.

That is **autoregressive generation**. By the end of this chapter you will have:

- understood greedy decoding (always pick `argmax`) and **sampling** (draw from the softmax distribution),
- met **temperature**, the single scalar that interpolates between confident-greedy and uniform-random,
- met **top-k** sampling, which restricts the distribution to the $k$ most-likely next tokens before sampling,
- added `mygpt.generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None)` to the package,
- watched the trained model from Chapter 14 generate `"I love AI ! I love AI !"` continuously, and observed how generation degrades once the sequence is longer than the model's trained context.

§1.10 promised: "by the end of this tutorial you will have a Python package called `mygpt` that you can train on a text file and use to generate text." Chapter 14 cashed in the *training* half. This chapter cashes in the *generation* half.

---

## 15.1 The autoregressive loop

In code, the generation loop is six lines:

```python
ids = prompt_ids
for _ in range(max_new_tokens):
    logits, _ = model(ids)            # (B, T, V)
    next_logits = logits[:, -1, :]    # (B, V) — only the last position's prediction
    next_ids = pick_one(next_logits)  # (B, 1) — sampling strategy goes here
    ids = torch.cat([ids, next_ids], dim=1)   # (B, T+1)
```

Three things to read off:

- **We only use the *last* position's logits.** `model(ids)` returns logits at every position; the last one is the prediction of the next token after the prompt. The other logits are useful for training (their loss tells us how well the model predicts each known position) but irrelevant at generation time.
- **The prompt is just a starting sequence.** It can be a single token, a sentence, or a paragraph. As long as it is at most `max_seq_len` tokens (the model's context window), the model can consume it.
- **We append, not overwrite.** Each step grows the sequence by one token. After `max_new_tokens` steps the sequence has length `len(prompt) + max_new_tokens`.

The "sampling strategy" — what `pick_one` does — is the *only* design choice in the loop. The rest of this chapter is about the four standard sampling strategies and how they interact.

---

## 15.2 Setup

This chapter assumes you finished Chapter 14 — `mygpt/` exists with `get_batch`, the `GPT` class, and a `trained_gpt.pt` checkpoint produced by `experiments/28_train_gpt.py`.

If you skipped Chapter 14, recreate the state from a clean directory and re-train the model:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Overwrite **`src/mygpt/__init__.py`** with the Chapter 14 ending state from `docs/_state_after_ch14.md`. Then re-create and run **`experiments/28_train_gpt.py`** from §14.5 to produce `trained_gpt.pt`.

You are ready.

---

## 15.3 Greedy decoding

The simplest sampling strategy: pick the token with the highest logit. This is the **argmax** of the softmax distribution; equivalently the argmax of the raw logits, since softmax is monotonic.

```python
next_ids = next_logits.argmax(dim=-1, keepdim=True)
```

Greedy decoding is **deterministic**: for the same model and prompt, you get the same continuation every time.

It is also typically **repetitive**: at every position, "the most likely token" is a stable local optimum, so the model can get stuck in a loop. (For our trained model on a periodic corpus this is a feature, not a bug — the cycle is exactly what we want it to repeat.)

**Save the following to** 📄 `experiments/30_greedy.py`:

```python
"""Experiment 30 — Greedy generation from the trained GPT.

argmax(logits) at every step, no randomness, prompt='I', 7 new tokens.
"""

import torch

from mygpt import GPT, VOCAB, set_seed


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
    model.load_state_dict(torch.load("trained_gpt.pt"))
    model.eval()

    ids = torch.tensor([[VOCAB.index("I")]])  # (1, 1)
    print(f"prompt: {[VOCAB[i] for i in ids[0].tolist()]}")

    with torch.no_grad():
        for _ in range(7):
            logits, _ = model(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)

    print(f"output: {[VOCAB[i] for i in ids[0].tolist()]}")


if __name__ == "__main__":
    main()
```

Run it (after you have run experiment 28 to produce `trained_gpt.pt`):

```bash
uv run python experiments/30_greedy.py
```

**Expected output:**

```text
prompt: ['I']
output: ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!']
```

The model produces the cycle perfectly: 7 new tokens after the 1-token prompt = 8 total tokens, exactly matching the cycle length the model trained on. (We will see in §15.9 what happens past 8.)

---

## 15.4 Sampling from the softmax distribution

Greedy decoding loses information: the model's softmax distribution is `(0.9, 0.07, 0.02, 0.01)` and `(0.5, 0.3, 0.15, 0.05)` are equally well-described by "argmax = 0", but they say very different things about how confident the model is. **Sampling** uses the full distribution: draw the next token from $\text{softmax}(\text{logits})$ as a categorical distribution.

```python
probs = F.softmax(next_logits, dim=-1)
next_ids = torch.multinomial(probs, num_samples=1)
```

`torch.multinomial(probs, n)` draws `n` independent samples from a categorical distribution defined by `probs` (which must be non-negative and sum to 1 along the last axis). For our `(B, V)` probability tensor, it returns `(B, n)` of token ids.

For our highly-trained, high-confidence model, sampling and greedy give nearly identical results within the trained context — the softmax is so peaked (probabilities like `0.998, 0.001, 0.001, 0.000`) that `multinomial` almost always picks the most-likely token. To see sampling actually do something different we need to *flatten* the distribution, which is what temperature is for.

---

## 15.5 Temperature

Temperature is one scalar that interpolates between greedy ($\tau \to 0$) and uniform ($\tau \to \infty$). The recipe: **divide logits by $\tau$ before softmax**.

$$
\text{softmax}_\tau(\mathbf{z}) \;=\; \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}.
$$

Read the limits:

- $\tau = 1$: standard softmax. No change.
- $\tau \to 0^{+}$: dividing by something tiny blows up the gap between the largest and second-largest logits. Softmax becomes almost one-hot at the argmax — equivalent to greedy decoding.
- $\tau \to \infty$: dividing by something huge collapses every logit toward 0. Softmax becomes uniform across the vocabulary — pure random sampling.

Practical defaults: $\tau \in [0.7, 1.0]$ for "creative but coherent" sampling; $\tau \in [0, 0.5]$ for "stay close to the most likely choice"; $\tau > 1.5$ produces noticeably more random output.

Two properties:

- **Temperature does not change the rank order.** Dividing every logit by the same positive number preserves which is biggest. So argmax is the same; only the *spread* of probabilities changes.
- **Temperature does not change the model's parameters.** It is a sampling-time knob, applied to the model's output logits. The same model can be sampled at different temperatures.

A worked example. Take logits $(2.0, 1.0, 0.0)$:

| $\tau$ | softmax probabilities                           |
|--------|-------------------------------------------------|
| 1.0    | $(0.665, 0.245, 0.090)$                         |
| 0.5    | $(0.867, 0.117, 0.016)$                         |
| 2.0    | $(0.506, 0.307, 0.186)$                         |
| 100.0  | $(0.337, 0.333, 0.330)$ — nearly uniform        |

We will let an experiment confirm those numbers.

**Save the following to** 📄 `experiments/31_temperature.py`:

```python
"""Experiment 31 — Temperature reshapes the softmax distribution.

For logits=(2.0, 1.0, 0.0), show softmax at several temperatures.
"""

import torch
import torch.nn.functional as F


def main() -> None:
    logits = torch.tensor([2.0, 1.0, 0.0])
    print(f"logits: {logits}\n")
    for tau in (1.0, 0.5, 2.0, 100.0):
        probs = F.softmax(logits / tau, dim=-1)
        print(f"  tau = {tau:>5.1f}: softmax = {[f'{p:.3f}' for p in probs.tolist()]}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/31_temperature.py
```

**Expected output:**

```text
logits: tensor([2., 1., 0.])

  tau =   1.0: softmax = ['0.665', '0.245', '0.090']
  tau =   0.5: softmax = ['0.867', '0.117', '0.016']
  tau =   2.0: softmax = ['0.506', '0.307', '0.186']
  tau = 100.0: softmax = ['0.337', '0.333', '0.330']
```

The columns shift exactly as predicted: low τ concentrates on the largest logit, high τ flattens toward uniform.

---

## 15.6 Top-k sampling

Even at moderate $\tau$, the long tail of the distribution — the unlikely tokens — sometimes contributes garbage. **Top-k sampling** truncates: keep only the $k$ largest logits, set the rest to $-\infty$, then softmax and sample.

```python
v, _ = torch.topk(logits, k)             # the k largest logits (descending)
logits[logits < v[:, [-1]]] = -float('inf')
# now softmax over the modified logits
```

After the assignment, every logit not in the top-$k$ is $-\infty$, so its softmax probability is exactly 0. Sampling now draws *only* from the top-$k$ tokens, with their relative probabilities preserved.

Two boundary cases:

- $k = 1$: keep only the argmax. Equivalent to greedy decoding.
- $k = V$ (or $k$ ≥ vocabulary size): keep everything. No truncation; same as plain sampling.

Top-k is GPT-2's default sampling strategy; the original generation paper used $k = 40$ for a vocabulary of 50,257.

For our $V = 4$ vocabulary, top-k with $k = 2$ is the most we can do before $k = V$ degenerates to plain sampling. We will see top-k more clearly when we play with bigger vocabularies in Part V.

---

## 15.7 `mygpt.generate`: the unified function

We package greedy / sampling / temperature / top-k into one function with sensible defaults:

```python
def generate(
    model,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
):
    """Autoregressively generate max_new_tokens after prompt_ids."""
    ids = prompt_ids
    for _ in range(max_new_tokens):
        # Crop to context window if needed
        if ids.shape[1] > model.max_seq_len:
            ids_cond = ids[:, -model.max_seq_len:]
        else:
            ids_cond = ids
        with torch.no_grad():
            logits, _ = model(ids_cond)
        logits = logits[:, -1, :] / temperature             # (B, V), scaled
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
        ids = torch.cat([ids, next_ids], dim=1)
    return ids
```

Three design choices worth flagging:

- **`temperature=1.0` and `top_k=None` is plain sampling.** Greedy is `top_k=1` (or, near-equivalently, `temperature=0.001`).
- **The context-window crop in the first three lines** is GPT-2's standard behaviour: if the sequence has grown longer than what the model can process (`max_seq_len`), we keep only the *last* `max_seq_len` tokens. The earlier tokens are forgotten. (Real models use much larger context windows so this only matters in very long generation.)
- **`torch.no_grad()` matters.** Generation does not need gradients; turning autograd off speeds up the forward pass and avoids holding onto intermediate activations.

**Append the following to** 📄 `src/mygpt/__init__.py` (after the `GPT` class, before `main`):

```python
def generate(
    model: "GPT",
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively generate max_new_tokens after prompt_ids.

    Inputs:
        model:           a trained GPT (or compatible Module returning logits).
        prompt_ids:      long tensor of shape (B, T_prompt).
        max_new_tokens:  how many tokens to append.
        temperature:     softmax temperature; <1 sharpens, >1 flattens.
        top_k:           if given, restrict sampling to the top_k most-likely
                         tokens at each step.

    Output:
        long tensor of shape (B, T_prompt + max_new_tokens).
    """
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        ids_cond = (
            ids[:, -model.max_seq_len :] if ids.shape[1] > model.max_seq_len else ids
        )
        with torch.no_grad():
            logits, _ = model(ids_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_ids], dim=1)
    return ids
```

---

## 15.8 Generating from the trained model

Now the satisfying experiment: load `trained_gpt.pt`, prompt with `"I"`, and generate the full cycle. We will use `top_k=1` for fully deterministic, reproducible output.

**Save the following to** 📄 `experiments/32_generate_from_trained.py`:

```python
"""Experiment 32 — Use mygpt.generate on the trained model.

prompt='I', generate 7 new tokens with top_k=1 (greedy). Stays within
the trained T=8 context window — see experiment 33 for what happens
beyond it.
"""

import torch

from mygpt import GPT, VOCAB, generate, set_seed, to_ids


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
    model.load_state_dict(torch.load("trained_gpt.pt"))

    prompt = to_ids(["I"]).unsqueeze(0)
    out = generate(model, prompt, max_new_tokens=7, top_k=1)

    print(f"prompt: {[VOCAB[i] for i in prompt[0].tolist()]}")
    print(f"output: {[VOCAB[i] for i in out[0].tolist()]}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/32_generate_from_trained.py
```

**Expected output:**

```text
prompt: ['I']
output: ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!']
```

That is GPT-2 in miniature: a transformer with 2,240 parameters, trained for 200 AdamW steps on 256 tokens of synthetic text, generating the cycle the corpus encoded. The promise from §1.10 — "you will have a Python package called `mygpt` that you can ... use to generate text from the command line" — is delivered.

---

## 15.9 What happens beyond the trained context

Our model was trained with `seq_len = 8`. The position embedding was thus exposed to positions 0..7 during training. For positions 8 and beyond, the position embedding has *never received a gradient* — its weights are still at the random initialisation. Generation past position 7 is therefore extrapolating into untrained territory, and the model's predictions become unreliable.

**Save the following to** 📄 `experiments/33_beyond_context.py`:

```python
"""Experiment 33 — Generation degrades beyond the trained T=8 context.

prompt='I', generate 12 tokens with top_k=1. The first 8 tokens (positions
0..7) are the perfect cycle; the rest (positions 8..) drift, because
the position embeddings at those positions were never trained.
"""

import torch

from mygpt import GPT, VOCAB, generate, set_seed, to_ids


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
    model.load_state_dict(torch.load("trained_gpt.pt"))

    prompt = to_ids(["I"]).unsqueeze(0)
    out = generate(model, prompt, max_new_tokens=12, top_k=1)

    tokens = [VOCAB[i] for i in out[0].tolist()]
    cycle = ["I", "love", "AI", "!"] * 4   # what we'd expect if the cycle held
    print(f"output:   {tokens}")
    print(f"expected: {cycle[:len(tokens)]}")
    print()
    for t, (got, want) in enumerate(zip(tokens, cycle)):
        ok = "OK" if got == want else "DRIFT"
        print(f"  position {t}: got={got!r:<7} expected={want!r:<7} {ok}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/33_beyond_context.py
```

**Expected output:**

```text
output:   ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!', 'I', 'AI', '!', '!', 'love']
expected: ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!', 'I', 'love', 'AI', '!', 'I']

  position 0: got='I'     expected='I'     OK
  position 1: got='love'  expected='love'  OK
  position 2: got='AI'    expected='AI'    OK
  position 3: got='!'     expected='!'     OK
  position 4: got='I'     expected='I'     OK
  position 5: got='love'  expected='love'  OK
  position 6: got='AI'    expected='AI'    OK
  position 7: got='!'     expected='!'     OK
  position 8: got='I'     expected='I'     OK
  position 9: got='AI'    expected='love'  DRIFT
  position 10: got='!'     expected='AI'    DRIFT
  position 11: got='!'     expected='!'     OK
  position 12: got='love'  expected='I'     DRIFT
```

Position 0–8 are clean (the model trained at every position $\leq 7$, plus position 8 is "the natural next token after a length-8 input" which the model effectively saw during training). From position 9 onward, the position embedding is in untrained territory and the model drifts.

This is the practical reason real GPT models train on long sequences (1024 tokens for GPT-2, much more for newer models): so that the position embeddings at those positions are *also* trained. Our `max_seq_len=64` could in principle accept inputs that long, but only the first 8 positions actually have useful position embeddings.

The fix, if we wanted to generate longer cycles reliably, is to retrain with a larger `seq_len` so all positions up to `max_seq_len` see gradient updates. We will not retrain here; the lesson is the *concept* — context window + training context length are not the same thing, and confusing them gets you garbled output.

---

## 15.10 Experiments

1. **Compare temperature on the trained model.** Edit `experiments/32_generate_from_trained.py` to call `generate(model, prompt, max_new_tokens=7, temperature=2.0)`. The model is *so* confident inside the trained context (the gap between the top logit and the others is huge) that at `temperature=2.0` (and even at `3.0`) the output is still the cycle. Bump it to `temperature=5.0` and you may see one or two tokens drift; only at extreme settings like `temperature=100.0` does the output become noticeably random.
2. **Compare top_k.** With prompt `"I"` and `max_new_tokens=7`, try `top_k=1` (greedy), `top_k=2`, `top_k=4` (no truncation, since $V = 4$). Within the trained context all three give the cycle, because the most-likely token is always correct. Top-k matters more when the model is *uncertain* — a regime our overtrained tiny model rarely enters.
3. **Different prompt.** Try `prompt = to_ids(["love", "AI", "!"]).unsqueeze(0)` and `max_new_tokens=4`. Should the model continue with `"I love AI !"`? Yes — and within the trained context (length $3 + 4 = 7 \leq 8$) it does. Verify.
4. **Forget the context crop.** Try the §15.7 `generate` with the line `ids_cond = ids[:, -model.max_seq_len:]` removed (so the full sequence is always passed in). Generate 100 tokens. Watch what happens when the input length exceeds `model.max_seq_len = 64`: the model raises `ValueError: input length T=65 exceeds max_seq_len=64`. The crop in `generate` is what prevents this.

After each experiment, restore the file you changed before moving on.

---

## 15.11 Exercises

1. **Greedy is `top_k=1`, almost.** Argue that `generate(..., top_k=1)` is mathematically identical to *deterministic argmax decoding*, except for the case where two logits are exactly equal at the same position. (Hint: `torch.topk(logits, 1)` plus `multinomial` over a one-hot distribution always returns that one position; ties are broken by `argmax` order in PyTorch.)
2. **Why divide by $\tau$ before softmax, not after?** Argue that `softmax(z / τ)` and `(softmax(z))^(1/τ)` (followed by renormalisation) are *not* the same, and that only the first form has the clean limiting properties (uniform at $\tau \to \infty$, one-hot at $\tau \to 0$). (Hint: try $z = (1, 1, 1)$ and notice that `(softmax(z))^(1/τ)` is degenerate for any $\tau$.)
3. **Top-p sampling.** A different sampling strategy, called **top-p** or *nucleus* sampling, keeps the smallest set of top tokens whose cumulative probability is at least $p$. Argue why top-p adapts to the *shape* of the distribution (uniformly-spread distributions keep more tokens; sharply-peaked distributions keep fewer) while top-k always keeps exactly $k$. (We do not implement top-p here; nanoGPT does.)
4. **What if max_new_tokens overflows max_seq_len?** Our `generate` function crops to `max_seq_len` at every step. Argue why this means generating 100 tokens with `max_seq_len=64` is fine but the model only ever conditions on the last 64 tokens during the second half of generation. (This is GPT-2's "sliding window" decoder; the early prompt is forgotten.)

---

## 15.12 What's next

We can train. We can generate. The remaining three chapters move from "tutorial-scale toy" to "could-actually-train-a-real-corpus":

- **Chapter 16** replaces the four-token vocabulary with a **character-level tokenizer** that can handle real text. We will scan a text file, build a vocabulary of every distinct character, and produce `to_ids` / `from_ids` functions.
- **Chapter 17** runs an actual training run on a real text file (Tiny Shakespeare, ~1 MB) — a more realistic scale for a laptop CPU.
- **Chapter 18** wraps everything in a CLI with checkpoints: `mygpt train file.txt`, `mygpt generate --checkpoint ckpt.pt --prompt "..."`. The end product of the tutorial.

> **Looking ahead — what to remember from this chapter**
>
> 1. The autoregressive loop is six lines: forward, take last position's logits, pick a token, append, repeat. The "pick a token" step is the only design choice.
> 2. Greedy decoding (argmax) is deterministic; sampling (multinomial on softmax) is not. Temperature scales the spread of the softmax; top-k zeros out the tail.
> 3. `mygpt.generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None)` packages all four strategies into one function with sensible defaults.
> 4. Generation past the *trained* context length (here, T=8 because we trained on 8-token chunks) drifts because position embeddings at higher indices were never trained. Real models avoid this by training at the full max_seq_len.

On to [Chapter 16 — A reusable character tokenizer](16_character_tokenizer.md) *(coming soon)*.
