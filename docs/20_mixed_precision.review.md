# Student walkthrough report: Chapter 20 — Mixed precision training (bf16)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/20_mixed_precision.md`
- Total sections walked: 4 of 4 executable subsections (§20.1 setup, §20.4 code edits, §20.5 fp32 run, §20.6 bf16 run, §20.8 backward-compat smoke test)
- Files modified: 1 — `src/mygpt/__init__.py` (replace `_train_command`, replace `_generate_command`, add `--precision` to both subparsers)
- Files produced by the CLI: 2 — `sh-fp32.ckpt`, `sh-bf16.ckpt`
- Shell commands run: 8 (uv init, uv add, curl, two `uv run mygpt train`, one `uv run mygpt generate`, one `mygpt train --help` to verify the flag)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.**

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0
- torch: 2.11.0
- mps available: True; cuda available: False
- Working directory: `/tmp/code-along-runs/ch20-review-122450/mygpt`

## Walkthrough

### Section: §20.1 Setup
- Files written: `src/mygpt/__init__.py` (Ch.19 ending state copied from `LLM Fundamentals/mygpt/`)
- Commands run: `uv init mygpt --package`, `cd mygpt && mkdir -p experiments`, `uv add torch numpy`, `curl -s -o tinyshakespeare.txt …`
- Output: project initialised; tinyshakespeare.txt = 1,115,394 bytes.
- Issues raised here: none.

### Sections §20.2 – §20.4 (concept + code edits)
- §20.2 (bit-format primer) and §20.3 (autocast pattern) are prose. Math is correct: 1 + 8 + 7 = 16 (bf16), 1 + 8 + 23 = 32 (fp32), 1 + 5 + 10 = 16 (fp16). Range claims (3.4×10³⁸ for bf16, 6.5×10⁴ for fp16) are correct.
- §20.4 code edits applied cleanly:
  - `_train_command`: added `print(f"precision: …")` line and the `if args.precision == "bf16"` autocast wrap around the forward.
  - `_generate_command`: added the autocast wrap around `generate(...)` for bf16.
  - `p_train.add_argument("--precision", …)` and matching `p_gen.add_argument(...)` added between the existing `--device` block and `set_defaults(...)`.
- Smoke test: `uv run mygpt train --help` shows both `[--device {auto,cuda,mps,cpu}]` and `[--precision {fp32,bf16}]`.
- Issues raised here: none.

### Section: §20.5 fp32 still bit-reproduces Ch.19
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --precision fp32 --output sh-fp32.ckpt`
- Captured output:
  ```
  device:       mps
  precision:    fp32
  corpus chars: 1,115,394
  vocab_size:   65
  params:       207,296
  steps:        2000
  step     1: loss = 41.0367
  step   500: loss = 2.5944
  step  1000: loss = 2.3529
  step  1500: loss = 2.1795
  step  2000: loss = 2.0785

  saved checkpoint to sh-fp32.ckpt
  ```
- Expected output match: yes — exact, including the new `precision: fp32` line. Loss values bit-identical to Ch.19 §19.6 and Ch.17 §17.5. Backward-compat preserved.
- Issues raised here: none.

### Section: §20.6 bf16 (within tolerance)
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --precision bf16 --output sh-bf16.ckpt`
- Captured output:
  ```
  device:       mps
  precision:    bf16
  corpus chars: 1,115,394
  vocab_size:   65
  params:       207,296
  steps:        2000
  step     1: loss = 41.0393
  step   500: loss = 2.5938
  step  1000: loss = 2.3540
  step  1500: loss = 2.1799
  step  2000: loss = 2.0790

  saved checkpoint to sh-bf16.ckpt
  ```
- Expected output match: **yes, within the chapter's declared ±0.01 tolerance.** Diffs against the chapter's quoted bf16 numbers:
  - step 1: 41.0393 vs 41.0393 (exact)
  - step 500: 2.5938 vs 2.5926 (Δ 0.0012)
  - step 1000: 2.3540 vs 2.3532 (Δ 0.0008)
  - step 1500: 2.1799 vs 2.1793 (Δ 0.0006)
  - step 2000: 2.0790 vs 2.0797 (Δ 0.0007)
  All within ±0.0015, well inside the chapter's declared ±0.01 tolerance. The chapter's prose explicitly says "treat the bf16 expected outputs as ±0.01 tolerance, not bit-exact" — this run honours that contract.
- The chapter's §20.7 honest finding ("bf16 is slower than fp32 here") is corroborated: this fp32 run took ~29 s and the bf16 run took ~36 s wall-clock — same ~1.25× slowdown the chapter quotes.
- Issues raised here: none.

### Section: §20.8 Backward-compat smoke test
- Command: `uv run mygpt generate --checkpoint sh-fp32.ckpt --prompt "ROMEO:" --device cpu`
- Output:
  ```
  device: cpu

  ROMEO:
  Thy momed has seltered, a neark'ly your tle centeloourse.
  Of therere hath thin beielly saneer best.

  BRINCE:
  Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
  Whe a bros swencurenty hou
  ```
- Expected output match: yes — byte-for-byte exact. The fp32 checkpoint loads cleanly and produces the canonical Ch.17 §17.6 / Ch.19 §19.7 sample. No regression from the autocast addition.
- Issues raised here: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end including the two long training runs (~29 s + ~36 s on the same M1 the chapter was authored on). All Expected Output blocks pass: fp32 is bit-identical to Ch.19, bf16 is within the chapter's own declared ±0.01 tolerance, the backward-compat generation is byte-exact. The chapter's headline pedagogical finding — that bf16 is *slower* than fp32 at this model size on M1 MPS — is empirically reproduced (29 s vs 36 s, ~1.25× slowdown).

The chapter's prose is honest about non-determinism: it asks the reader to treat bf16 numbers as approximate, and the actual run's deviations from the chapter's quoted bf16 values are well within the declared tolerance band.

§20.7's forward-looking claims about CUDA tensor-core hardware ("the matmul itself is 2× faster in bf16") and §20.10 ex 1's fp16-overflow numerical example were not empirically tested (no CUDA in this environment, and the bf16-vs-fp16 contrast is conceptual). Both are pedagogically standard claims.

The chapter is ready to commit.
