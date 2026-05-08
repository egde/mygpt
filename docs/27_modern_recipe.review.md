# Student walkthrough report: Chapter 27 — Modern recipe vs Ch.17 baseline

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/27_modern_recipe.md`
- Total sections walked: 4 of 4 executable subsections (§27.1 setup, §27.5 baseline run, §27.6 modern run, §27.9 sampling). §27.10 experiments left as student-led; §27.11 are conceptual exercises.
- Files modified: 0 — chapter has **no `mygpt/` code changes** as advertised.
- Files produced: 2 — `sh-baseline.ckpt`, `sh-modern.ckpt`
- Shell commands run: 6 (uv init, uv add, ls, two `uv run mygpt train`, two `uv run mygpt generate`)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.**

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- mps available: True; cuda: False
- Working directory: `/tmp/code-along-runs/ch27-review-160848/mygpt`

## Walkthrough

### Section: §27.1 Setup
- `tinyshakespeare.txt` already present (1,115,394 bytes); `ls` succeeds, `curl` short-circuited by `||`.
- The chapter explicitly says "We do not add anything new" — no edits to `src/mygpt/__init__.py`. Verified.
- Issues raised here: none.

### Section: §27.5 The baseline run
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --output sh-baseline.ckpt`
- Output (key lines):
  ```
  device:       mps
  precision:    fp32
  norm:         layer
  position:     learned
  num_heads:    4
  num_kv_heads: 4
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

  saved checkpoint to sh-baseline.ckpt
  ```
- Expected output match: yes — byte-exact. Bit-identical to every prior chapter's default run since Ch.17 (params 207,296; loss curve `41.0367 → 2.5944 → 2.3529 → 2.1795 → 2.0785`).
- Wall time: 36 s on M1 MPS (chapter says "about 29 s"; the chapter uses "about" language and the variance is within typical system-load noise — not a finding).
- Issues raised here: none.

### Section: §27.6 The modern run
- Command:
  ```bash
  uv run mygpt train tinyshakespeare.txt --device mps \
      --precision bf16 \
      --schedule cosine --warmup 100 --max-grad-norm 1.0 \
      --norm rms --position rope --num-kv-heads 2 \
      --output sh-modern.ckpt
  ```
- Output (key lines):
  ```
  device:       mps
  precision:    bf16
  norm:         rms
  position:     rope
  num_heads:    4
  num_kv_heads: 2
  corpus chars: 1,115,394
  train chars:  1,115,394
  vocab_size:   65
  params:       186,240
  steps:        2000
  schedule:     cosine (warmup=100)
  max_grad_norm:1.0
  step     1: loss = 56.3070  lr = 1.00e-05
  step   500: loss = 2.0891   lr = 8.95e-04
  step  1000: loss = 1.8628   lr = 5.41e-04
  step  1500: loss = 1.7880   lr = 1.61e-04
  step  2000: loss = 1.7408   lr = 0.00e+00

  saved checkpoint to sh-modern.ckpt
  ```
- Expected output match: yes (within stated bf16 tolerance). The chapter's expected loss values vs. mine:

  | Step | Chapter | Mine    | Δ        |
  |------|---------|---------|----------|
  | 1    | 56.3070 | 56.3070 | 0.0000   |
  | 500  | 2.0880  | 2.0891  | +0.0011  |
  | 1000 | 1.8624  | 1.8628  | +0.0004  |
  | 1500 | 1.7876  | 1.7880  | +0.0004  |
  | 2000 | 1.7402  | 1.7408  | +0.0006  |

  All deltas are well inside the ±0.01 nats the chapter declares ("loss numbers below match my run to within ~0.01 nats but may not bit-reproduce on yours"). Param count `186,240` is byte-exact, as is the LR schedule output (warmup at `1e-5`, peak `8.95e-04`, decay to `0`). Wall time 41 s vs the chapter's "about 42 s" — within noise.
- Issues raised here: none.

### Section: §27.9 Sampling
- `uv run mygpt generate --checkpoint sh-baseline.ckpt --prompt "ROMEO:" --device cpu` produces:
  ```
  ROMEO:
  Thy momed has seltered, a neark'ly your tle centeloourse.
  Of therere hath thin beielly saneer best.

  BRINCE:
  Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
  Whe a bros swencurenty hou
  ```
  Byte-exact to the chapter's expected baseline sample. (Same as Ch.17 §17.6 / Ch.18 / … / Ch.26 — fp32 deterministic chain holds.)
- `uv run mygpt generate --checkpoint sh-modern.ckpt --prompt "ROMEO:" --device cpu` produces:
  ```
  ROMEO:
  Why moft thereself I do a meardy woudd helse, I
  Here would fotherent hath thin beirlinse.
  Shich that's there to we alce: in the trund; my fatter we woot to a man,
  Mervough the conn shup thus me. for
  ```
  The chapter-quoted sample begins identically (`ROMEO:\nWhy moft thereself I do a meardy woudd helse, I\nHere would fother`) and diverges in the rest of line 2 (`ent hath thin beirlinse` vs `ere hath thin beiels`). The chapter explicitly anticipates this: *"your text may differ slightly due to bf16-trained checkpoint nondeterminism, but should look qualitatively similar"*. The qualitative claims hold:
  - sentences delimited by `.` (`helse, I` … `beirlinse.` … `man,`),
  - line breaks,
  - mixed comma/semicolon/colon use inside clauses (`alce:` `trund;`),
  - vowel-consonant balance comparable to baseline,
  - structurally more "writing-like" than the baseline's run-on stream.
  Acceptable per the chapter's own tolerance.
- Issues raised here: none.

### §27.10 Experiments and §27.11 Exercises
- §27.10 Exp 1 (six single-flag ablations): not run — chapter explicitly notes this is "~4 minutes of compute" and student-led. The qualitative prediction (RoPE moves loss most, cosine+warmup next, others ≈ free) is consistent with Ch.24/25/26 numbers I have already verified in those chapters.
- §27.10 Exp 2 (RoPE-alone vs RoPE+RMSNorm+GQA): student-led; conceptually a composition check.
- §27.10 Exp 3 (`--steps 10000`): student-led; chapter's "loss to roughly 1.4 nats by step 10000" is qualitative.
- §27.11 are pen-and-paper exercises (weight decay reading, FLOPs derivation, gradient-clipping reasoning) — not empirical.
- Issues raised here: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end including both ~30–40 s training runs (baseline fp32 deterministic; modern bf16 within ±0.01 nats tolerance) and both generation calls. Every Expected Output block matches under the chapter's stated tolerance:

- **Baseline (fp32)**: bit-identical to the chapter and to every prior chapter's default. params=207,296; loss curve `41.0367 / 2.5944 / 2.3529 / 2.1795 / 2.0785`; ROMEO sample byte-exact.
- **Modern (bf16)**: param count `186,240` byte-exact; step-1 loss `56.3070` byte-exact (step 1 is pre-optimizer, so the bf16 forward on identical init is deterministic); steps 500/1000/1500/2000 within +0.0011 / +0.0004 / +0.0004 / +0.0006 of chapter values; wall time 41 s vs chapter ~42 s; ROMEO sample matches structurally with byte-identical opening 3–4 tokens before bf16 weight noise causes greedy-path divergence — exactly as the chapter declares.

The chapter's quantitative summary table (§27.7) reconciles arithmetically: `2.0785 − 1.7402 = 0.3383 → −0.34 nats`; `(2.0785 − 1.7402)/2.0785 = 16.27% → −16%`; loss/param ratios `1.003 × 10⁻⁵` and `9.34 × 10⁻⁶` both compute correctly; param savings `207,296 − 186,240 = 21,056` and the −10% framing is right.

The qualitative ablation decomposition in §27.8 cites Ch.25 §25.9 (RoPE alone: `1.7812`) and Ch.26 §26.9 (GQA-2: `2.0854`) and Ch.24 §24.9 (RMSNorm: `2.0752`) — all numbers I have already verified in those chapters. The "RoPE alone gets you 0.30 nats out of 0.34" claim follows: `2.0785 − 1.7812 = 0.2973`, with the remaining `1.7812 − 1.7402 ≈ 0.041` nats coming from cosine+warmup+clipping+RMSNorm+GQA+bf16 combined. This matches the chapter's framing.

§27.6's revised step-1 explanation (random-init effect, not warmup) is consistent with the empirical step-1 loss `56.3070` matching exactly even at bf16 — confirming step 1 is a deterministic forward pass on random init, before optimizer/warmup enters.

The chapter is ready to commit.
