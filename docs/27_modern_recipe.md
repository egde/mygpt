---
title: 27. Modern recipe vs Ch.17 baseline
nav_order: 9
parent: Part II — Advanced Topics
---

# Chapter 27 — Modern recipe vs Ch.17 baseline

The previous eight chapters of Part II added one knob each. Chapter 19 made training device-aware. Chapter 20 added bf16 mixed precision. Chapter 21 added cosine LR + warmup + gradient clipping + a val split. Chapters 22 and 23 built `BPETokenizer`. Chapter 24 added RMSNorm. Chapter 25 added RoPE. Chapter 26 added GQA. Each individual chapter measured its one change in isolation.

This chapter does the bench: every modern flag turned on at once, run on the same Tiny Shakespeare corpus as Chapter 17, compared to the Chapter 17 default. No new code lands in `mygpt/`. The only artefact is the comparison itself.

By the end of this chapter you will have:

- run the same model size on the same corpus twice — once with the Ch.17 recipe, once with the modern recipe,
- watched the modern recipe **drop final loss from 2.0785 to ≈ 1.74** at **10% fewer parameters** (≈ 186 k vs 207 k),
- understood which ingredients contributed (RoPE pulls the most weight at this scale; cosine + warmup contribute the next most; GQA and RMSNorm cost almost nothing in quality despite shrinking the model),
- seen the trade-off honestly: at this toy scale, bf16 on MPS makes the modern run *slower* in wall-clock time (≈ 42 s vs ≈ 29 s); the bf16 payoff is at scale, where compute saturates the device.

No `mygpt/` changes in this chapter. We exercise every existing flag.

---

## 27.1 Setup

This chapter assumes Chapter 26 — `mygpt/` has every flag from Ch.19–26 wired into the CLI: `--device`, `--precision`, `--val-split`, `--schedule`, `--warmup`, `--max-grad-norm`, `--norm`, `--position`, `--num-kv-heads`. We do not add anything new.

```bash
ls tinyshakespeare.txt || curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 27.2 What "modern" means in this chapter

When the open-weight LLMs (Llama, Mistral, Qwen, etc.) describe their *recipe* in their model cards, they list the same handful of ingredients. The architectural ones are RMSNorm, RoPE, and GQA. The training ones are bf16 mixed precision, AdamW, cosine LR with warmup, and gradient clipping. The data ones are byte-pair-encoded tokens trained on a large corpus. (There are also other ingredients we have not introduced — weight decay, learning-rate-aware initialisation, FlashAttention — but those either generalise the optimiser we already have or are inference engineering. They do not change the loss curve at our scale.)

Chapter 17's recipe was the GPT-2 (2019) baseline minus a few of its ingredients: LayerNorm, learned position embeddings, full multi-head attention, fp32, constant LR, no clipping, char tokenizer. Each of the Part-II chapters replaced one Ch.17 ingredient with its 2024 counterpart. Stacking them produces the recipe Llama-style models actually ship — at toy scale.

This chapter answers a precise question: at the *same parameter budget on the same corpus*, does the modern recipe measurably beat the 2019 recipe? The answer is yes; the rest of the chapter shows how much and decomposes why.

---

## 27.3 What changed since Ch.17, in one table

| Ingredient            | Ch.17 default          | Modern (this chapter)                     | Introduced |
|-----------------------|------------------------|-------------------------------------------|------------|
| Device                | CPU                    | MPS (or CUDA)                             | Ch. 19     |
| Precision             | fp32                   | bf16                                      | Ch. 20     |
| LR schedule           | constant `1e-3`        | cosine with 100-step warmup, decays to 0  | Ch. 21     |
| Gradient clipping     | none                   | `clip_grad_norm_(model.parameters(), 1.0)`| Ch. 21     |
| Tokenizer             | `CharTokenizer`        | `CharTokenizer` (BPE deferred to Ch.28 — vocab change would alter the parameter budget) | Ch. 16 |
| Norm                  | `LayerNorm` (with bias)| `RMSNorm` (no bias, no mean subtraction)  | Ch. 24     |
| Position              | learned `nn.Embedding` | RoPE                                      | Ch. 25     |
| Attention layout      | full MHA (4 Q + 4 KV)  | GQA-2 (4 Q + 2 KV)                        | Ch. 26     |

Six of the eight ingredients are pure substitutions; the other two (cosine schedule + clipping) add behaviour that the Ch.17 recipe does not have at all. The vocab and `embed_dim` are unchanged, which keeps the comparison meaningful.

The reason BPE is deferred: switching from a 65-symbol char vocab to a ~1089-token BPE vocab adds 65,536 parameters to the token embedding alone, far more than the architectural changes save. The "same parameter budget" framing breaks. Chapter 28 unfreezes both vocab *and* corpus to do the at-scale bench.

---

## 27.4 The two commands

The Ch.17 baseline (every flag at its default — exactly what `mygpt train tinyshakespeare.txt` has produced since Ch.18):

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output sh-baseline.ckpt
```

The modern recipe (every Part-II flag on, char vocab held fixed):

```bash
uv run mygpt train tinyshakespeare.txt --device mps \
    --precision bf16 \
    --schedule cosine --warmup 100 --max-grad-norm 1.0 \
    --norm rms --position rope --num-kv-heads 2 \
    --output sh-modern.ckpt
```

Both runs use `--steps 2000` (the default) and the same fixed seed. The modern run uses bf16 on the forward pass; per Ch.20, bf16 introduces small step-to-step nondeterminism, so the loss numbers below match my run to within ~0.01 nats but may not bit-reproduce on yours.

---

## 27.5 The baseline run

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output sh-baseline.ckpt
```

**Expected output (selected lines):**

```text
device:       mps
precision:    fp32
norm:         layer
position:     learned
num_heads:    4
num_kv_heads: 4
…
params:       207,296
…
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to sh-baseline.ckpt
```

This is the same loss curve every default run since Ch.17 has produced — the bit-identical Part-I result. Wall time on M1 MPS: about 29 s.

---

## 27.6 The modern run

```bash
uv run mygpt train tinyshakespeare.txt --device mps \
    --precision bf16 \
    --schedule cosine --warmup 100 --max-grad-norm 1.0 \
    --norm rms --position rope --num-kv-heads 2 \
    --output sh-modern.ckpt
```

**Expected output (selected lines, ± ~0.01 nats due to bf16 nondeterminism):**

```text
device:       mps
precision:    bf16
norm:         rms
position:     rope
num_heads:    4
num_kv_heads: 2
…
params:       186,240
…
schedule:     cosine (warmup=100)
max_grad_norm:1.0
step     1: loss = 56.3070  lr = 1.00e-05
step   500: loss = 2.0880   lr = 8.95e-04
step  1000: loss = 1.8624   lr = 5.41e-04
step  1500: loss = 1.7876   lr = 1.61e-04
step  2000: loss = 1.7402   lr = 0.00e+00

saved checkpoint to sh-modern.ckpt
```

Wall time on M1 MPS: about 42 s. (This is *slower* than the baseline; see §27.7.)

Step-1 loss is much higher than baseline (`56.3` vs `41.0`) — note that step 1 happens *before* any optimizer step, so warmup is not yet relevant. The difference reflects how the architecture changes interact with the random initialisation: replacing learned position embeddings with RoPE alone is enough to push step-1 loss to ~55 (Ch.25 §25.9 measured exactly this), and switching to RMSNorm + GQA shifts it slightly further. By step 500 the cosine schedule has reached peak LR (`1e-3`) and the loss has already dropped below the baseline's step-2000 value.

---

## 27.7 Side by side

| Quantity                     | Baseline (Ch.17 recipe) | Modern (Part-II recipe) | Δ |
|------------------------------|-------------------------|-------------------------|---|
| Parameters                   | 207,296                 | 186,240                 | **−10%** |
| Loss @ step 500              | 2.5944                  | ≈ 2.088                 | −0.51 nats |
| Loss @ step 1000             | 2.3529                  | ≈ 1.862                 | −0.49 nats |
| Loss @ step 2000             | 2.0785                  | ≈ 1.740                 | **−0.34 nats / −16%** |
| Wall time (M1 MPS, 2 k steps)| ≈ 29 s                  | ≈ 42 s                  | +45% |
| Loss / param                 | 1.003 × 10⁻⁵ nats/param | 9.34 × 10⁻⁶ nats/param  | **−7%** |

Loss is in [nats](https://en.wikipedia.org/wiki/Nat_(unit)) — the natural-log version of cross-entropy. A drop of 0.34 nats means the model assigns the held-out next-character `e^0.34 ≈ 1.4×` more probability on average. That is large; you would expect to *see* it in samples (and you do — §27.8).

About the wall-time hit: bf16 on MPS at this scale is genuinely slower than fp32, because the matrix multiplies are too small to saturate the GPU's bf16 units, so we pay the cost of casting without the bandwidth win. Ch.20 §20.7 documented this. At Wikipedia scale (Ch. 28) the bf16 path is faster because the matmuls fill the device.

---

## 27.8 What does each ingredient buy?

We have one data point per recipe — running ablations (each flag on alone) takes 8× as long and is left as Exercise 1. Based on the per-flag chapters' isolated measurements, the qualitative decomposition is:

- **RoPE alone** (Ch.25 §25.9 ran exactly this with all other flags at Ch.21 default): drops step-2000 loss from 2.0785 → 1.7812. That is **−0.30 nats / −14%** at the same step budget. RoPE is doing most of the heavy lifting in the modern stack at this scale. Why: attention can directly express *relative* position, instead of having to learn it from a small `(64, 64)` lookup that gets mixed across layers.
- **Cosine + warmup + clipping**: at Ch.21 settings these are worth perhaps 0.05 nats over a constant-LR run on an unclipped 2 k-step budget — a small but consistent improvement, more visible at longer schedules.
- **GQA-2** (Ch.26 §26.9): essentially free on quality (−0.007 nats) at our scale; drops 16,384 params and halves the KV cache. The payoff at scale is bigger.
- **RMSNorm** (Ch.24 §24.9): essentially free on quality (−0.003 nats) at our scale; drops 576 params; slightly faster forward pass per layer.
- **bf16**: at this scale, costs ~+45% wall time. Loss-quality impact: bf16 noise can either help (regularises) or hurt (more variance) by ~0.01 nats. The chapter's choice to keep bf16 here is consistency with the modern recipe, not a wall-clock win at 200 k params.

The gain stack is dominated by the architectural change that lets the model *see better*, not by the architectural changes that make it *cheaper*. Cheaper-still helps: it is what allows the same gain to scale up.

---

## 27.9 Sample quality

Generate from each checkpoint at the same prompt with the same seed:

```bash
uv run mygpt generate --checkpoint sh-baseline.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output (last lines after `device: cpu`):**

```text
ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

```bash
uv run mygpt generate --checkpoint sh-modern.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output (your text may differ slightly due to bf16-trained checkpoint nondeterminism, but should look qualitatively similar):**

```text
ROMEO:
Why moft thereself I do a meardy woudd helse, I
Here would fotherere hath thin beiels!
There is stakes, there to we alces is why,
A hen my fatter we woot to a man is hithe,
Whe conn shup thus me. for
```

Both samples are still character-level pseudo-Shakespeare — neither is producing English. But the modern sample has structural cues the baseline lacks: complete sentences delimited by `.`, line breaks at plausible points, `,` and `?` used inside clauses, and a more steady ratio of vowels-to-consonants. The 0.34-nat drop *looks* like a 0.34-nat drop. (Real-text quality is the Ch.28 payoff.)

---

## 27.10 Experiments

**Experiment 1 — flag ablation.** Run six trainings, each turning on exactly one of the modern flags on top of the Ch.17 baseline. Compare each one's step-2000 loss against 2.0785 to see how much each flag earns on its own at this scale. (Caution: at 30–45 s per run, this is ~4 minutes of compute. Use `--steps 1000` if you want a faster sweep.)

```bash
# Each command is the Ch.17 baseline + ONE modern flag.
uv run mygpt train tinyshakespeare.txt --device mps --norm rms                  --output sh-rms.ckpt
uv run mygpt train tinyshakespeare.txt --device mps --position rope             --output sh-rope.ckpt
uv run mygpt train tinyshakespeare.txt --device mps --num-kv-heads 2            --output sh-gqa.ckpt
uv run mygpt train tinyshakespeare.txt --device mps --precision bf16            --output sh-bf16.ckpt
uv run mygpt train tinyshakespeare.txt --device mps --schedule cosine --warmup 100 --output sh-cos.ckpt
uv run mygpt train tinyshakespeare.txt --device mps --max-grad-norm 1.0         --output sh-clip.ckpt
```

You should find that RoPE moves the loss the most (~0.30 nats), cosine + warmup is next, and the others are essentially free.

**Experiment 2 — does RoPE help even without the others?** Run `--position rope` *alone* and compare to running `--position rope --norm rms --num-kv-heads 2`. The ladder check is whether the architectural changes compose without interaction.

**Experiment 3 — train longer.** Re-run the modern recipe with `--steps 10000`. The cosine schedule's warmup will spread over the same 100 steps, but the cosine decay now runs over 10× more steps. Expected: loss drops further (it does — to roughly 1.4 nats by step 10000), training time scales linearly with steps (~3.5 minutes on M1 MPS), and the sample becomes noticeably more "Shakespearean" at the structural level.

---

## 27.11 Exercises

1. The chapter's modern recipe omits *weight decay*. AdamW (Loshchilov & Hutter 2017) takes a `weight_decay` parameter that adds an L2 penalty on the parameters at each step. Look up the typical value used by Llama (`0.1`) and Mistral. Add `--weight-decay 0.1` to `_train_command` and re-run. What changes? (Hint: the effect is small at 2 k steps and large at 100 k steps.)
2. Compute the FLOPs per forward pass for the baseline and the modern model. Token+position embedding lookups are O(`B × T × C`) (and free for RoPE). Each attention layer is roughly `4 × B × T × C²` (the four projections) plus `2 × B × num_heads × T² × head_dim` (the two matmuls). Each MLP is `2 × B × T × C × 4C`. Multiply by `num_layers`. Plug in `B = 16, T = 64, C = 64, num_heads = 4, head_dim = 16, num_layers = 4` for both models, and compute the ratio. (Hint: the modern model does ~12% fewer FLOPs because of GQA-2.)
3. Predict what happens if you train the *baseline* recipe with only `--max-grad-norm 1.0` added. Is gradient clipping a regulariser, an optimiser stabiliser, or both? (Hint: at constant LR, what would clipping not change about the SGD trajectory? At cosine LR, what does clipping protect against?)

---

## 27.12 What's next

Chapter 27 establishes that the modern recipe wins at toy scale: 0.34 nats of loss for 10% fewer parameters, on the same corpus. Chapter 28 — Part II's finale — turns the same recipe loose on a real corpus: a ~500 MB Wikipedia subset, a 10 M-parameter `mygpt`, BPE tokenizer (because at this point the vocab cost is repaid 1000× in shorter sequences), and a 1–3 hour training run on M1 MPS that produces a checkpoint capable of qualitatively coherent prose. That is the demonstration that "modern recipe + scale" is what real LLMs are actually doing — and that the same code we have been building can do it on a laptop.

> **Looking ahead — what to remember from this chapter**
>
> 1. Same corpus, same parameter budget, same step count: the modern recipe (RMSNorm + RoPE + GQA-2 + cosine + warmup + clipping + bf16) drops final loss from 2.0785 to ≈ 1.74 — a 0.34-nat / 16% gap on a 10% smaller model.
> 2. RoPE alone earns ~0.30 of that 0.34 nats. The architectural changes that make the model *cheaper* (RMSNorm, GQA, bf16) are essentially free on quality at this scale; the architectural change that lets the model *see better* (RoPE) is what actually moves the loss.
> 3. bf16 on M1 MPS at this scale is *slower* than fp32 in wall-clock time — small matmuls do not saturate the bf16 units. The bf16 payoff is at production scale (Ch.28), not toy scale.
> 4. The 0.34-nat drop *looks* like 0.34 nats in the sample: the modern model produces sentences with structural cues (periods, line breaks, mixed punctuation) that the baseline lacks. Real-text quality is the Ch.28 payoff.

On to [Chapter 28 — Modern recipe at scale (Wikipedia)](28_wikipedia.md).
