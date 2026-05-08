# Student walkthrough report: Chapter 10 — Layer normalization

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/10_layer_norm.md`
- Total sections walked: 7 of 9 (§10.1, §10.5, §10.9 are prose; the remaining 6 sections all execute cleanly)
- Files created: 3 — `src/mygpt/__init__.py` (rewritten twice), `experiments/22_layernorm_by_hand.py`, `experiments/23_layernorm_drift.py`
- Shell commands run: 9 (core path + four §10.7 experiments)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly, including the seeded §10.6 drift numbers (six std values across 6 decimal places). Every §10.7 experiment prediction held empirically.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch10-rev1-20260502-061058/mygpt`

## Walkthrough

### Section: §10.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.9 ending state — VOCAB, to_ids, set_seed, TokenEmbedding, SingleHeadAttention, MultiHeadAttention, MLP — reconstructed by combining the prior chapters' state per `_state_after_ch09.md`)
- Commands run: `uv init mygpt --package`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes** — the setup runs cleanly. The chapter points to `docs/_state_after_ch09.md` as the source of truth for Ch.9 ending state, which is the right pattern; the snapshot file does require the student to reconstruct the full file from the prior state files (it only shows the new `MLP` class), but a careful student following the snapshot trail back to Ch. 8's snapshot can rebuild the whole file.
- Issues raised here: none

### Section: §10.3 By hand
- Files written: `experiments/22_layernorm_by_hand.py`
- Commands run: `uv run python experiments/22_layernorm_by_hand.py`
- Output:
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
- Expected output match: **yes** — exact match, including the float32-precision `0.999996` instead of an exact `1.000000`. The chapter explicitly notes this and explains the `eps` cause.
- Issues raised here: none

### Section: §10.4 Building mygpt.LayerNorm
- Files written: `src/mygpt/__init__.py` (Ch.10 version with `LayerNorm` added after `MLP`, plus the new `main`)
- Commands run: `uv run mygpt`
- Output:
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
- Expected output match: **yes** — exact match. Means are float32 noise around zero (chapter notes this); stds are `0.99996…0.99997` (within 5 × 10⁻⁵ of 1.0; chapter notes the eps cause). Parameter count of 172 = 16 + 8 + 148.
- Issues raised here: none

### Section: §10.6 LayerNorm controls drift
- Files written: `experiments/23_layernorm_drift.py`
- Commands run: `uv run python experiments/23_layernorm_drift.py`
- Output:
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
- Expected output match: **yes** — every std value matches the chapter's expected block to 6 decimal places. The comparison with §9.7 (2.21 → 1.69 at layer 30) is empirically supported.
- Issues raised here: none

### Section: §10.7 Experiments
- Files written: temporary inline edits.
- Commands run:
  ```bash
  uv run python -c "import torch; from mygpt import LayerNorm; ln = LayerNorm(4); ln.eval(); print(ln(torch.zeros(4)))"
  uv run python -c "..."   # gamma=2, beta=1
  uv run python -c "..."   # permutation invariance
  uv run python -c "..."   # nn.LayerNorm agreement
  ```
- Output:
  - **Exp 1** (zeros input): `tensor([0., 0., 0., 0.], grad_fn=<AddBackward0>)`. Matches the chapter's prediction.
  - **Exp 2** (gamma=2, beta=1 on x=(1,2,3,4)): `tensor([-1.6833, 0.1056, 1.8944, 3.6833])`. Matches the chapter's prediction exactly.
  - **Exp 3** (permutation invariance): `out2 == out1[:, perm, :]` evaluates to `True`. Matches.
  - **Exp 4** (nn.LayerNorm agreement): `torch.equal` → `False`, `torch.allclose(..., atol=1e-5)` → `True`. Matches the chapter's prediction exactly (it explicitly notes `equal` will fail and `allclose` will succeed).
- Expected output match: **yes** for all four experiments.
- Issues raised here: none

### Section: §10.8 Exercises
- Files written: none (these are all reflective).
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§10.2–10.7 inside a fresh `/tmp/code-along-runs/ch10-rev1-20260502-061058/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every Expected-Output block matched the actual machine output, including:

- the §10.3 by-hand calculation (mean = 2.5, var = 1.25, normed std = 0.999996 matching the chapter's expected block);
- the §10.4 main() output, including the four float32-noise per-token means (`1.34e-7`, `0.0`, `1.49e-8`, `5.96e-8`) and the four near-1 per-token stds (`0.9999693…0.9999963`);
- the §10.6 drift experiment (six std values across two columns of seed-0 outputs, all matching to 6 decimal places);
- the four §10.7 experiments — including the nn.LayerNorm-vs-ours agreement test, where the chapter correctly predicts `torch.equal == False` and `torch.allclose == True`.

Three things worth noting on persona check, none of them findings:

1. **The chapter is honest about float32 precision.** Three places where mathematically-exact identities appear as float32-noise-perturbed values (the `0.999996` std, the per-token means hovering at `~1e-7`, the `nn.LayerNorm` allclose-but-not-equal) are all flagged in the chapter prose. A literal student reading the printed output would see "this looks slightly off from what was promised" and find the explanation right there.
2. **`pre-norm` and `post-norm` are introduced before being used.** ✓ §10.5 → §10.6, with the §10.6 experiment using pre-norm, matching GPT-2's choice.
3. **Sixth consecutive chapter to land at zero findings on review #1.** The pre-review checklist + lessons register continue to pay off; this chapter's pre-review pass empirically caught a fabricated §10.4 expected block (the original draft predicted `[0.0, 0.0, 0.0, 0.0]` for the per-token means and `[1.1547, 1.1547, 1.1547, 1.1547]` for the stds, which are wrong because the script uses `unbiased=False` not `unbiased=True`; the `0.99996` numbers above are correct). The pre-review captured the actual values before the reviewer ran.

Pedagogically, this chapter completes the transformer block: attention (Ch.6–8), MLP (Ch.9), layer norm (this chapter). Chapter 11 will compose them into a single `TransformerBlock` class — the central unit of GPT-2 — using the pre-norm + residual pattern §10.5 motivates and §10.6 demonstrates.
