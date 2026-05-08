# Student walkthrough report: Chapter 13 — The forward pass with loss

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/13_forward_pass_with_loss.md`
- Total sections walked: 5 of 8 (§13.1, §13.5, §13.7, §13.8 are prose; the remaining 5 sections all execute)
- Files created: 2 — `src/mygpt/__init__.py` (Ch.12 ending state plus the new `forward(ids, targets=None)` signature and updated `main`), `experiments/26_cross_entropy_by_hand.py`
- Shell commands run: 7 (core path + four §13.6 experiments)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly; every §13.6 experiment prediction held empirically (1.3863 / ~0 / 100.0 / `loss=None`).

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch13-rev1-20260502-064010/mygpt`

## Walkthrough

### Section: §13.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.12 ending state)
- Commands run: `uv init`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes**
- Issues raised here: none

### Section: §13.3 Cross-entropy by hand
- Files written: `experiments/26_cross_entropy_by_hand.py`
- Commands run: `uv run python experiments/26_cross_entropy_by_hand.py`
- Output:
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
- Expected output match: **yes** — exact match. The by-hand calculation `e^1 ≈ 2.7183`, `e^2 ≈ 7.3891`, sum ≈ 12.124, softmax ≈ 0.6095, loss ≈ 0.4952 reproduces verbatim.
- Issues raised here: none

### Section: §13.4 GPT.forward returning loss
- Files written: `src/mygpt/__init__.py` (Ch.13 version with the new `forward(ids, targets=None)` signature and updated `main`)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4

  Input ids (B, T):    (1, 3)  [[0, 1, 2]]
  Targets   (B, T):    (1, 3)  [[1, 2, 3]]
  Logits    (B, T, V): (1, 3, 4)
  Loss:                4.2588

  Reference: log(V) = log(4) = 1.3863
  (Random-init loss is typically a small multiple of log(V); training drives it down.)
  ```
- Expected output match: **yes** — exact match including the loss value `4.2588`.
- Issues raised here: none

### Section: §13.6 Experiments
- Files written: temporary inline edits.
- Commands run:
  ```bash
  # exp 1: uniform logits → log(V)
  uv run python -c "import torch; ..."
  # exp 2: confident correct → ~0
  uv run python -c "..."
  # exp 3: confident wrong → ~100
  uv run python -c "..."
  # exp 4: targets=None → (logits, None)
  uv run python -c "..."
  ```
- Output:
  - **Exp 1** (uniform): `loss: 1.3863` — exactly `log(4)`. Matches.
  - **Exp 2** (confident correct): `loss: 0.000000e+00`. Matches "essentially 0".
  - **Exp 3** (confident wrong): `loss: 100.0000` — softmax of +100 at the wrong target gives probability ≈ `e^-100`, so `-log(e^-100) = 100`. Matches.
  - **Exp 4** (`targets=None`): `logits.shape = (1, 3, 4)`, `loss = None`. Matches.
- Expected output match: **yes** for all four.
- Issues raised here: none

### Section: §13.7 Exercises
- Files written: none (these are reflective).
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§13.2–13.6 inside a fresh `/tmp/code-along-runs/ch13-rev1-20260502-064010/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every numerical claim in the chapter held empirically:

- the by-hand cross-entropy calculation (`exp(z)` values, sum 12.124, softmax 0.6095, loss 0.4952) matched `F.cross_entropy` to float-32 precision;
- the §13.4 random-init loss of `4.2588` matched verbatim;
- the §13.6 experiments produced the predicted boundary cases (uniform → log(V), confident-correct → 0, confident-wrong → 100, targets=None → loss=None) exactly.

Persona check on every section passed: cross-entropy is defined in §13.3 before being used in §13.4; the next-token shift (input `[:-1]`, targets `[1:]`) is motivated in §13.1 before being applied in §13.4; the `(logits, None)` return convention for inference is explained in §13.5 before the student is asked to use it in §13.6 exp 4.

The chapter's pedagogical narrative — "we have a model and a loss, gradient descent (Ch.4) does the rest" — is now ready to cash in. Chapter 14 will write the training loop.
