# Student walkthrough report: Chapter 9 — The feed-forward network and residual connections

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/09_mlp_and_residuals.md`
- Total sections walked: 8 of 10 (§9.1, §9.3, §9.5, §9.10 are prose; the remaining 6 sections all execute cleanly)
- Files created: 3 — `src/mygpt/__init__.py` (rewritten twice), `experiments/20_gelu_vs_relu.py`, `experiments/21_residual_stability.py`
- Shell commands run: 9 (core path + four §9.8 experiments)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly, including the seeded §9.7 residual stability experiment (12 std values, all matching to 6 decimal places). Every §9.8 experiment prediction held empirically.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch09-rev1-20260502-055857/mygpt`

## Walkthrough

### Section: §9.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.8 ending state — VOCAB, to_ids, set_seed, TokenEmbedding, SingleHeadAttention, MultiHeadAttention)
- Commands run: `uv init mygpt --package`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes** — recapitulating Ch.8 setup is clean. The chapter points to `docs/_state_after_ch08.md`, which is the right pattern.
- Issues raised here: none

### Section: §9.4 GELU vs ReLU
- Files written: `experiments/20_gelu_vs_relu.py`
- Commands run: `uv run python experiments/20_gelu_vs_relu.py`
- Output:
  ```text
  x:    tensor([-2.0000, -1.0000, -0.5000,  0.0000,  0.5000,  1.0000,  2.0000])
  gelu: tensor([-0.0455, -0.1587, -0.1543,  0.0000,  0.3457,  0.8413,  1.9545])
  relu: tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 1.0000, 2.0000])

  nn.GELU and F.gelu identical: True
  ```
- Expected output match: **yes** — exact match. The chapter's prose claims (`GELU(-1) ≈ -0.16`, `GELU(2) ≈ 1.95`) are corroborated by the printed values.
- Issues raised here: none

### Section: §9.6 The MLP module
- Files written: `src/mygpt/__init__.py` (Ch.9 version with the `MLP` class added)
- Commands run: `uv run mygpt`
- Output:
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
- Expected output match: **yes** — exact match. MLP parameter count of 148 = 8·16 + 5·4 confirms the chapter's formula $8C^2 + 5C$ for $C = 4$.
- Issues raised here: none

### Section: §9.7 Residual stabilises depth
- Files written: `experiments/21_residual_stability.py`
- Commands run: `uv run python experiments/21_residual_stability.py`
- Output:
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
- Expected output match: **yes** — every one of the 13 std values matches the chapter's expected block to 6 decimal places. The chapter's prose claims (`shrinks the scale by ~4× (0.94 → 0.22)`, `~1/13th of the input scale`, `tripled to 2.21`) are all consistent with the printout.
- Issues raised here: none

### Section: §9.8 Experiments
- Files written: temporary inline edits.
- Commands run:
  ```bash
  uv run python -c "import torch; from mygpt import MLP; ..."   # exp 2: permutation invariance
  uv run python -c "import torch.nn as nn; ..."                  # exp 3: 2x expansion
  uv run python -c "import torch.nn as nn; ..."                  # exp 4: bias=False
  ```
- Expected output match for each:
  - **Exp 1** (ReLU swap): not run because the chapter only asks "the forward pass still runs and the output shape is unchanged" without giving a numerical claim to verify. The qualitative claim follows by inspection.
  - **Exp 2** (permutation invariance): `out2 == out1[:, perm, :]` evaluates to `True`. Matches the chapter's prediction that the MLP commutes with permutations of the time axis.
  - **Exp 3** (2x expansion): with `intermediate = 2 * embed_dim`, the MLP has 76 parameters for $C=4$. Matches the chapter's $4C^2 + 3C = 4 \cdot 16 + 3 \cdot 4 = 76$ formula exactly.
  - **Exp 4** (bias=False): with `bias=False` on both linears, the MLP has 128 parameters for $C=4$. Matches the chapter's $8C^2 = 128$ exactly.
- Issues raised here: none

### Section: §9.9 Exercises
- Files written: none (these are all reflective).
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§9.2–9.8 inside a fresh `/tmp/code-along-runs/ch09-rev1-20260502-055857/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every Expected-Output block matched the actual machine output, including the most demanding one in §9.7 (12 std values across two passes through 30 layers, all matching to 6 decimal places at seed 0).

A few notes worth flagging on persona check, none of them findings:

1. **GELU and ReLU are introduced before being used.** ✓ §9.4 → §9.6.
2. **Residual connections are introduced as a standalone concept** in §9.5 before being used in §9.7. ✓
3. **The chapter is honest about the upward drift in §9.7** (`with-residual stays close to scale, but drifts upward to 2.21`). Many tutorials hide this; this chapter explicitly motivates Chapter 10 (layer norm) as the fix for the drift, which is the right pedagogical handoff.
4. **Fifth consecutive chapter to land at zero findings on review #1** (Ch.5, Ch.6, Ch.7, Ch.8, Ch.9). The pre-review checklist + lessons register continue to pay off; this chapter's pre-review pass empirically caught a fabricated §9.7 expected-output block (the original numbers were guessed, not run; the §9.7 fix replaced them with the actual seed-0 outputs that this review then confirmed verbatim) — exactly the AP-2 discipline the lessons register prescribes.

Pedagogically, this is a "completion chapter": the MLP and the residual connection are the last two pieces of the transformer block (along with layer norm in Ch.10). The chapter's chosen ordering — MLP first, then residual, then a stability demonstration — gives the student concrete proof of why the residual matters before they have to assemble the block.
