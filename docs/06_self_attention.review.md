# Student walkthrough report: Chapter 6 — Single-head self-attention from scratch

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/06_self_attention.md`
- Total sections walked: 11 of 13 (§6.1, §6.13 are prose; the remaining 11 sections walk cleanly)
- Files created: 4 — `src/mygpt/__init__.py` (rewritten twice), `experiments/13_softmax.py`, `experiments/14_attention_by_hand.py`, `experiments/15_attention_running_example.py`
- Shell commands run: 12 (core path + four §6.11 experiments)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly (including the seeded attention-weights matrix in §6.10, all five steps of the §6.8 by-hand walk, and every §6.11 experiment prediction).

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch06-rev1-20260502-052522/mygpt`

## Walkthrough

### Section: §6.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.5 ending state — VOCAB, to_ids, set_seed, TokenEmbedding)
- Commands run: `uv init mygpt --package`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes** — recapitulating the Ch.5 setup runs cleanly.
- Issues raised here: none

### Section: §6.3 Softmax
- Files written: `experiments/13_softmax.py`
- Commands run: `uv run python experiments/13_softmax.py`
- Output:
  ```text
  z = tensor([1., 2., 3.])
  exp(z) = tensor([ 2.7183,  7.3891, 20.0855])
  sum of exp(z) = 30.1929
  softmax(z) by hand = tensor([0.0900, 0.2447, 0.6652])

  softmax(z) by torch = tensor([0.0900, 0.2447, 0.6652])
  sum by torch = 1.0000
  identical: True
  ```
- Expected output match: **yes** — exact match. The by-hand calculation in §6.3 ($e^1 \approx 2.7183$, sum ≈ 30.1929, softmax ≈ (0.0900, 0.2447, 0.6652)) reproduces verbatim.
- Issues raised here: none

### Section: §6.8 By hand on a tiny example
- Files written: `experiments/14_attention_by_hand.py`
- Commands run: `uv run python experiments/14_attention_by_hand.py`
- Expected output match: **yes** — every one of the five steps matches the chapter's Expected-Output block exactly:
  - scores = `[[1,0,1],[0,1,1],[1,1,2]]`
  - scaled by 1/sqrt(2) = 0.7071
  - masked: upper triangle `-inf`
  - weights: `[[1.0000, 0, 0], [0.3302, 0.6698, 0], [0.2483, 0.2483, 0.5035]]`
  - output: `[[1.0000, 0], [0.3302, 0.6698], [0.7517, 0.7517]]`
  - The by-hand text in §6.8 (e.g. $e^{0.7071} \approx 2.028$, $e^{1.4142} \approx 4.113$, sum 8.169) reproduces.
- Issues raised here: none

### Section: §6.9 Extending mygpt: SingleHeadAttention
- Files written: `src/mygpt/__init__.py` (Ch.6 version with `SingleHeadAttention`)
- Commands run: `uv run mygpt`
- Output:
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
- Expected output match: **yes** — exact match. The 64 parameters of `SingleHeadAttention` confirm the chapter's formula $4 \cdot C \cdot d_h = 4 \cdot 4 \cdot 4 = 64$.
- Issues raised here: none

### Section: §6.10 End-to-end on the running example
- Files written: `experiments/15_attention_running_example.py`
- Commands run: `uv run python experiments/15_attention_running_example.py`
- Output:
  ```text
  Attention weights (B=1, T=4, T=4) — first batch element:
  tensor([[1.0000, 0.0000, 0.0000, 0.0000],
          [0.2526, 0.7474, 0.0000, 0.0000],
          [0.3018, 0.2972, 0.4011, 0.0000],
          [0.2149, 0.3615, 0.1914, 0.2321]])

  row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000])

  Final attention output shape: (1, 4, 4)
  ```
- Expected output match: **yes** — exact byte-for-byte match. The lower-triangular structure, the row-sum-to-1 property, and the structural `(1, 0, 0, 0)` first row are all confirmed.
- Issues raised here: none

### Section: §6.11 Experiments
- Files written: temporary edits to `experiments/14_attention_by_hand.py` and `experiments/15_attention_running_example.py` (all restored after).
- Commands run:
  ```bash
  python -c "X = torch.tensor([[1,0],[0,1],[1,1]]); ..."   # exp 1
  uv run python experiments/15_attention_running_example.py    # exp 2 unscaled, d=4
  uv run python experiments/15_attention_running_example.py    # exp 2 unscaled, d=64
  uv run python experiments/15_attention_running_example.py    # exp 3 no mask
  uv run python experiments/14_attention_by_hand.py             # exp 4 zeros input
  ```
- Output (exp 1): `scores symmetric: True` — chapter's prediction holds (which it always does, since $XX^\top$ is symmetric for any $X$).
- Output (exp 2, d=4 unscaled): row 1 weights become `[0.1026, 0.8974]` instead of `[0.2526, 0.7474]` — a small but visible shift toward the higher-scoring column. Matches the chapter's prediction "the change is small at d=4".
- Output (exp 2, d=64 unscaled): rows 1, 2, 3 become `[0.9024, 0.0976]`, `[0.0147, 0.0358, 0.9495]`, `[0.0284, 0.0018, 0.9558, 0.0141]`. Each row puts ≥ 0.90 mass on one column. Matches the chapter's prediction "almost all mass on a single column".
- Output (exp 3, no mask): weights are full, not lower-triangular. Row 0 = `[0.2571, 0.2206, 0.3235, 0.1988]` (now attends to all 4 positions). Row sums still = 1. Matches.
- Output (exp 4, zeros input): weights = `[[1.0, 0, 0], [0.5, 0.5, 0], [0.3333, 0.3333, 0.3333]]`. Matches the chapter's prediction exactly (uniform-with-causal-mask).
- Expected output match: **yes** for all four experiments.
- Issues raised here: none

### Section: §6.12 Exercises
- Files written: none (these are all reflective / proof-style exercises).
- Commands run: none.
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§6.2–6.11 inside a fresh `/tmp/code-along-runs/ch06-rev1-20260502-052522/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every numerical claim in the chapter — the by-hand softmax of `(1, 2, 3)`, the by-hand attention output `[[1, 0], [0.3302, 0.6698], [0.7517, 0.7517]]` on the 3×2 input, the seeded `(1, 4, 4)` attention weights matrix in §6.10 (which the author explicitly re-captured during pre-review after spotting a mismatch), the parameter count $4 \cdot C \cdot d_h$, and the four §6.11 experiment predictions — held exactly.

A few things worth noting on persona check, none of them findings:

1. **Softmax is introduced before being used** (§6.3 → §6.7). ✓
2. **Query / key / value are introduced as a triple before being used in the formula** (§6.4 → §6.5). ✓
3. **Causal mask is introduced before the full formula** (§6.6 → §6.7). ✓
4. **The pre-review pass at `docs/_review_lessons.md` already caught one finding before this review** (§6.10's hallucinated attention-weights matrix; fixed before invoking the reviewer). The pre-review checklist appears to be paying off — this is the second consecutive chapter to land at 0 findings on review #1.

Pedagogically, this is the densest chapter so far: softmax + Q/K/V + scaled dot product + causal mask + output projection are five new operations introduced in one chapter, but they are all introduced in order and each is justified before being used. A math-literate ML-naive student following the chapter end-to-end has every concept defined before it is invoked.
