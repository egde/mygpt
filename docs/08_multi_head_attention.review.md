# Student walkthrough report: Chapter 8 — Multi-head attention

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/08_multi_head_attention.md`
- Total sections walked: 7 of 10 (§8.1, §8.3, §8.4, §8.5, §8.10 are prose; the remaining 5 sections walk cleanly)
- Files created: 2 — `src/mygpt/__init__.py` (rewritten twice — Ch.7 ending state, then Ch.8), `experiments/19_mha_one_head_equivalence.py`
- Shell commands run: 11 (core path + four §8.8 experiments)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly; the §8.7 byte-for-byte equivalence with the Chapter 7 `SingleHeadAttention(C, head_dim=C)` was confirmed (`torch.equal` is `True`, max abs diff `0.000e+00`); every §8.8 experiment prediction held empirically.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch08-rev1-20260502-054715/mygpt`

## Walkthrough

### Section: §8.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.7 ending state — VOCAB, to_ids, set_seed, TokenEmbedding, SingleHeadAttention with register_buffer + dropout)
- Commands run: `uv init mygpt --package`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes** — recapitulating the Ch.7 setup runs cleanly. The chapter points to `docs/_state_after_ch07.md` as the source of truth for the Ch.7 ending state, which is the right pattern.
- Issues raised here: none

### Section: §8.6 The MultiHeadAttention module
- Files written: `src/mygpt/__init__.py` (Ch.8 version with both `SingleHeadAttention` and the new `MultiHeadAttention`)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4

  Token ids shape:                (1, 4)
  Embedded shape (B, T, C):       (1, 4, 4)
  MultiHeadAttention output shape: (1, 4, 4)

  num_heads = 2, head_dim = 2, embed_dim = 4

  TokenEmbedding parameters:        16
  MultiHeadAttention parameters:    64
  Total parameters:                 80
  ```
- Expected output match: **yes** — exact match. Total parameter count 80 matches the §7.6 ending state, confirming the chapter's claim that the factorisation `(num_heads, head_dim)` does not change the total.
- Issues raised here: none

### Section: §8.7 num_heads=1 equivalence
- Files written: `experiments/19_mha_one_head_equivalence.py`
- Commands run: `uv run python experiments/19_mha_one_head_equivalence.py`
- Output:
  ```text
  SingleHeadAttention(4, 4):
  tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
           [-0.1713,  0.5772,  0.2182, -0.4687],
           [-0.3211,  0.5328,  0.1321, -0.3144],
           [-0.1588,  0.2404,  0.0839, -0.0570]]])

  MultiHeadAttention(4, num_heads=1):
  tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
           [-0.1713,  0.5772,  0.2182, -0.4687],
           [-0.3211,  0.5328,  0.1321, -0.3144],
           [-0.1588,  0.2404,  0.0839, -0.0570]]])

  identical:    True
  max abs diff: 0.000e+00
  ```
- Expected output match: **yes** — exact byte-for-byte match. The chapter's claim "MultiHeadAttention(C, 1) and SingleHeadAttention(C, C) produce byte-for-byte identical output" is confirmed at `torch.equal()` strictness (no tolerance needed).
- Issues raised here: none

### Section: §8.8 Experiments
- Files written: temporary edits / inline `python -c` invocations.
- Commands run:
  ```bash
  uv run python -c "import math; ..."     # exp 1: weights (1, 2, 4, 4)
  uv run python -c "from mygpt import MultiHeadAttention; ..."   # exp 2: divisibility
  uv run python -c "from mygpt import MultiHeadAttention; ..."   # exp 3: param count
  uv run python -c "import torch; ..."    # exp 4: max_seq_len enforcement
  ```
- Output (exp 1, four-axis weights): `weights shape: (1, 2, 4, 4)`. Both `weights[0, 0]` (head 0) and `weights[0, 1]` (head 1) are lower-triangular (`weights[i].triu(1) == 0` everywhere) and differ from each other (`torch.equal(weights[0, 0], weights[0, 1])` is `False`). Matches the chapter's prediction "both lower-triangular but with *different* internal numbers".
- Output (exp 2, divisibility): `ValueError: embed_dim (4) must be divisible by num_heads (3)` — exact substring match. `MultiHeadAttention(6, 3)` builds with `head_dim = 2`. Both match.
- Output (exp 3, parameter independence):
  ```text
  MHA(8, h=1, head_dim=8): 256 params
  MHA(8, h=2, head_dim=4): 256 params
  MHA(8, h=4, head_dim=2): 256 params
  MHA(8, h=8, head_dim=1): 256 params
  ```
  `4 * 8 * 8 = 256` exactly, regardless of `num_heads`. Matches the chapter's prediction.
- Output (exp 4, max_seq_len): `ValueError: input length T=4 exceeds max_seq_len=2` — exact substring match.
- Expected output match: **yes** for all four experiments.
- Issues raised here: none

### Section: §8.9 Exercises
- Files written: none (these are all reflective / proof-style exercises).
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§8.2–8.8 inside a fresh `/tmp/code-along-runs/ch08-rev1-20260502-054715/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every Expected-Output block matched the actual machine output, including:

- the §8.6 `mygpt` entry-point output (parameter count 80, head_dim 2, num_heads 2);
- the §8.7 byte-for-byte equivalence with `SingleHeadAttention(C, C)` from Chapter 7 — the most important promise of the chapter, confirmed at `torch.equal()` strictness with max abs diff `0.000e+00`;
- the four §8.8 experiment predictions: 4-axis weight shape, divisibility error, parameter-count independence from `num_heads`, and `max_seq_len` enforcement.

Three things worth noting on persona check, none of them findings:

1. **The shape transformation `(B, T, C) → (B, h, T, d_h)` is introduced before being used.** ✓ §8.3 → §8.4.
2. **The `.contiguous()` requirement is explained where it appears.** ✓ §8.4 + ex 2 in §8.9.
3. **Fourth consecutive chapter to land at zero findings on review #1** (Ch.5, Ch.6, Ch.7, Ch.8). The pre-review checklist + lessons register continue to pay off; this chapter's pre-review pass empirically verified all four §8.8 experiment predictions before the reviewer ran (visible in the chapter's commit message), exactly the AP-3 discipline the lessons register prescribes.

The chapter completes the attention half of the transformer; the parameter count chain `Ch.6 (64) == Ch.7 (64) == Ch.8 (64 for h=2)` provides a clean continuity narrative.
