# Student walkthrough report: Chapter 21 — Training-loop hardening

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/21_training_loop_hardening.md`
- Total sections walked: 6 of 6 executable subsections (§21.1 setup, §21.3 helpers, §21.5 _train_command + flags, §21.6 backward-compat, §21.7 hardened recipe, §21.9 spot-check of exp 1 and exp 3)
- Files modified: 1 — `src/mygpt/__init__.py` (append `cosine_warmup_lr`, append `estimate_val_loss`, replace `_train_command`, add five flags to `p_train`)
- Files produced by the CLI: 2 in the review env — `sh-default.ckpt`, `sh-hardened.ckpt`
- Shell commands run: 8
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.**

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0 (mps available; cuda not)
- Working directory: `/tmp/code-along-runs/ch21-review-125502/mygpt`

## Walkthrough

### Sections §21.1 – §21.5 (setup + code edits)
- §21.1: project initialised; tinyshakespeare.txt = 1,115,394 bytes.
- §21.3: `cosine_warmup_lr` and `estimate_val_loss` appended after `get_batch`, before `TokenEmbedding` (chapter line 108: "after `get_batch`, before the `TokenEmbedding` class"). Both functions land cleanly.
- §21.5: `_train_command` replaced wholesale; five new `p_train.add_argument(...)` blocks added between the existing `--precision` block and `set_defaults(...)`. Smoke test: `uv run mygpt train --help` shows all five new flags (`--val-split`, `--val-every`, `--schedule`, `--warmup`, `--max-grad-norm`).
- Issues: none.

### Section: §21.6 Backward-compat (defaults)
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --output sh-default.ckpt`
- Captured output (key lines):
  ```
  device:       mps
  precision:    fp32
  corpus chars: 1,115,394
  train chars:  1,115,394
  vocab_size:   65
  params:       207,296
  steps:        2000
  schedule:     constant (warmup=0)
  max_grad_norm:0.0
  step     1: loss = 41.0367
  step   500: loss = 2.5944
  step  1000: loss = 2.3529
  step  1500: loss = 2.1795
  step  2000: loss = 2.0785
  ```
- Expected output match: yes — exact, including the new header lines (`train chars:`, `schedule:`, `max_grad_norm:`). Loss values bit-identical to Ch.20 / Ch.19 / Ch.17 §17.5. Backward-compat preserved as the chapter promises.
- Issues: none.

### Section: §21.7 Hardened recipe
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --val-split 0.1 --val-every 500 --schedule cosine --warmup 100 --max-grad-norm 1.0 --output sh-hardened.ckpt`
- Captured output (key lines):
  ```
  device:       mps
  precision:    fp32
  corpus chars: 1,115,394
  train chars:  1,003,854
  val chars:    111,540
  vocab_size:   65
  params:       207,296
  steps:        2000
  schedule:     cosine (warmup=100)
  max_grad_norm:1.0
  step     1: loss = 41.5789  lr = 1.00e-05
  step   500: loss = 2.4393  val = 2.4744  lr = 8.95e-04
  step  1000: loss = 2.2950  val = 2.2975  lr = 5.41e-04
  step  1500: loss = 2.1387  val = 2.2136  lr = 1.61e-04
  step  2000: loss = 2.1927  val = 2.2152  lr = 0.00e+00
  ```
- Expected output match: yes — exact, character-for-character, including the LR schedule values (1.00e-05 / 8.95e-04 / 5.41e-04 / 1.61e-04 / 0.00e+00) and the train/val pairs at each step.
- The chapter's four observations §21.7 ("LR schedule is doing what it says", "Train and val travel together", "Final loss is higher than constant-LR", "Step-1 loss is different from default because of val_split") are all empirically supported by the captured output.
- Issues: none.

### Section: §21.9 — experiment spot-checks
- **Exp 1 (warmup=0)**: rerun with `--warmup 0`. Captured: `step 500: loss = 2.4133  val = 2.4435  lr = 8.54e-04`. Chapter claims "step 500 you get ~2.41 vs warmup-100's 2.44" — 2.4133 rounds to 2.41 ✓. The chapter's parenthetical "Step-1 loss is unchanged because the LR doesn't affect the *forward* pass" is also verified — step 1 in both warmup-100 and warmup-0 runs printed `loss = 41.5789`.
- **Exp 3 (constant + val)**: rerun with `--schedule constant --warmup 0 --max-grad-norm 0.0 --val-split 0.1 --val-every 500`. Captured: `step 2000: loss = 2.1323  val = 2.1602`. Chapter claims "train loss ~2.13 and val loss ~2.16" — 2.1323 rounds to 2.13, 2.1602 rounds to 2.16 ✓.
- Issues: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end including the two long training runs (~30 s + ~30 s on this M1) and two follow-up experiments from §21.9. Every Expected Output block matches character-for-character, including the new train/val/lr triples, the cosine LR schedule values, and the backward-compat default loss curve. The chapter's four §21.7 observations are all empirically supported. The two specific approximate-numeric claims in §21.9 (warmup-0 step-500 ~2.41, constant step-2000 train ~2.13 / val ~2.16) round correctly from the captured outputs.

The mathematical derivation in §21.3 ("cosine branch starts at $1+\cos(0)=2$, ends at $1+\cos(\pi)=0$, giving lr=max_lr at t=warmup, lr=min_lr at t=total") is internally consistent and matches the captured `lr` printouts (lr at step 500 was empirically 8.95e-04, which the formula predicts as $0.5 \cdot 10^{-3} \cdot (1 + \cos(\pi \cdot 400/1900)) \approx 8.94 \cdot 10^{-4}$ ✓).

§21.8 (the long overfitting experiment, ~75 s, no expected output block to compare against) and §21.10 (exercises) are not empirically tested in this review pass.

The chapter is ready to commit.
