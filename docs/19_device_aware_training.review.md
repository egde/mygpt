# Student walkthrough report: Chapter 19 — Device-aware training

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/19_device_aware_training.md`
- Total sections walked: 5 of 5 executable subsections (§19.1, §19.3-§19.5 code edits, §19.6 two training runs, §19.7 three generation runs, §19.8 four experiments spot-checked, §19.9 exercise 4 verified)
- Files modified: 1 — `src/mygpt/__init__.py` (4 edits: append `pick_device`, replace `set_seed`, modify `load_checkpoint`'s first body line, replace `_train_command` + `_generate_command`, add `--device` to both subparsers)
- Files produced by the CLI: 2 — `shakespeare-cpu.ckpt`, `shakespeare-mps.ckpt`
- Shell commands run: 11 (uv init, uv add, curl, two `uv run mygpt train`, three `uv run mygpt generate`, two ad-hoc verifications, one error-mode probe)
- Issues found: **4** (0 blockers, 3 mismatches, 1 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Will gently snag a careful student in §19.8** — the experiment-block prose contains three small inaccuracies. None block the build. The main empirical content (§19.6 loss curves, §19.7 generation outputs) is exact.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0
- torch: 2.11.0 (macOS arm64 build — NOT compiled with CUDA support)
- mps available: True
- cuda available: False
- Working directory: `/tmp/code-along-runs/ch19-review-115150/mygpt`

## Walkthrough

### Section: §19.1 Setup
- Files written: `src/mygpt/__init__.py` (Ch.18 ending state copied from `LLM Fundamentals/mygpt/`)
- Commands run: `uv init mygpt --package`, `cd mygpt && mkdir -p experiments`, `uv add torch numpy`, `curl -s -o tinyshakespeare.txt …`
- Output: project initialised; torch+numpy installed; tinyshakespeare.txt = 1,115,394 bytes (matches Ch.17).
- Issues raised here: none.

### Sections §19.3 – §19.5 (code edits)
- All four edits land cleanly:
  1. `pick_device` appended after `set_seed`.
  2. `set_seed` replaced with the multi-device version.
  3. `load_checkpoint`'s `torch.load(path)` → `torch.load(path, map_location="cpu")`.
  4. `_train_command` and `_generate_command` replaced with device-aware versions.
  5. `--device` added to both `p_train` and `p_gen`.
- Smoke test: `uv run mygpt --help` and `uv run mygpt train --help` both show the new `--device {auto,cuda,mps,cpu}` flag.
- Issues raised here: none.

### Section: §19.6 Run 1 — CPU baseline
- Command: `uv run mygpt train tinyshakespeare.txt --device cpu --output shakespeare-cpu.ckpt`
- Output:
  ```
  device:       cpu
  corpus chars: 1,115,394
  vocab_size:   65
  params:       207,296
  steps:        2000
  step     1: loss = 41.0367
  step   500: loss = 2.5944
  step  1000: loss = 2.3529
  step  1500: loss = 2.1795
  step  2000: loss = 2.0785

  saved checkpoint to shakespeare-cpu.ckpt
  ```
- Expected output match: yes — exact, including the new `device: cpu` line.
- Issues raised here: none.

### Section: §19.6 Run 2 — MPS
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --output shakespeare-mps.ckpt`
- Output (key lines): identical loss values to the CPU run (`step 1: 41.0367`, `step 2000: 2.0785`); first line reads `device: mps`.
- Expected output match: yes — exact. Confirms the chapter's claim that fp32 deterministic matmul makes the loss curve agree to 4 decimals across CPU and MPS at this model size.
- Issues raised here: none.

### Section: §19.7 — three generation runs
- All three commands produce exactly the text quoted in the chapter:
  - `--checkpoint shakespeare-cpu.ckpt --device cpu` → `Thy momed has seltered, …` (Ch.17 §17.6 byte-for-byte).
  - `--checkpoint shakespeare-mps.ckpt --device mps` → `That hal nos themst your hallon murd: …`.
  - `--checkpoint shakespeare-cpu.ckpt --device mps` → identical to the second one (confirms checkpoint device-portability).
- Expected output match: yes — exact across all three.
- Issues raised here: none.

### Section: §19.8 — experiments (spot-checked)
- **Exp 1** (`--device` omitted on this Mac): runs cleanly and produces the MPS sample. **However**, the chapter prose says *"The CLI prints which device was chosen"* — this is true for `mygpt train` (which has a `print(f"device: {device}")` line) but **not** for `mygpt generate` (which does not print the device anywhere). The student running `uv run mygpt generate --checkpoint … --prompt …` without `--device` will see only the generated text, no device confirmation. Issue #1.
- **Exp 2** (`--device cuda` on a Mac): the actual error on a macOS arm64 torch install is `AssertionError: Torch not compiled with CUDA enabled`, **not** `RuntimeError: Found no NVIDIA driver on your system` (which is the error on Linux/Windows torch builds that have CUDA support but no driver). For the chapter's primary audience (M-series Mac users), the chapter's quoted error is wrong. Issue #2.
- **Exp 3** (timing): not formally re-run; the §19.6 timings already confirm MPS ~1.5× CPU on this M1.
- **Exp 4** (samples diverge at the first generated token): not actually true. With prompt `"ROMEO:"`, both CPU and MPS samples start with `"\nT"` and diverge at the 4th generated character (`Thy` vs `That`). Tokens 1, 2, 3 (newline, "T", "h") are identical across devices. Issue #3.
- Issues raised here: #1, #2, #3.

### Section: §19.9 Exercise 4 — verified
- Captured: `torch.manual_seed(0); torch.rand(3, device='cpu')` → `tensor([0.4963, 0.7682, 0.0885])`; `torch.mps.manual_seed(0); torch.rand(3, device='mps')` → `tensor([0.3664, 0.0755, 0.7784], device='mps:0')`. Different. Exercise's claim holds.
- Issues raised here: none.

## Issues

### Issue #1 — Mismatch — §19.8 exp 1: CLI does not print the device on `generate`
- Section: §19.8 — experiment 1 prose at chapter line 365.
- Where: *"Run `uv run mygpt generate --checkpoint shakespeare-cpu.ckpt --prompt "ROMEO:"` without `--device`. The CLI prints which device was chosen; on an M1 with no CUDA, it picks `mps`, so the sample matches the MPS run above."*
- What happened: the actual `_generate_command` does **not** print the device. Only `_train_command` has a `print(f"device: {device}")` line. The student running the suggested command sees only the generated text, no `device: mps` confirmation.
- Why a student would be stuck: not stuck, but the experiment loses its punchline — the student is told to look for evidence that auto picked MPS and then doesn't see any.
- Suggested fix: either add `print(f"device: {device}")` to `_generate_command` (matching `_train_command`'s convention) and re-record §19.7's expected outputs to include that line, or change the §19.8 exp 1 prose to *"the sample will match the MPS run above (confirming auto picked MPS); to see the device print, use `mygpt train` instead, which logs it at the top."* The first fix is cleaner pedagogically; it would also make exp 1 a proper "look at this confirmation" experiment.

### Issue #2 — Mismatch / AP-8 — §19.8 exp 2: wrong error message for the Mac audience
- Section: §19.8 — experiment 2 prose at chapter line 366.
- Where: *"`uv run mygpt generate --checkpoint shakespeare-cpu.ckpt --prompt "ROMEO:" --device cuda` raises a PyTorch error: `RuntimeError: Found no NVIDIA driver on your system`."*
- What happened: on the macOS arm64 torch build (which most Part-II readers will have, since Apple M1 is the canonical hardware), the actual error is `AssertionError: Torch not compiled with CUDA enabled`. This is a different exception type AND a different message. The `RuntimeError: Found no NVIDIA driver…` only appears on Linux/Windows torch installs that *were* compiled with CUDA but find no driver at runtime.
- Why a student would be stuck: not stuck — they get *an* error — but the error doesn't match what the chapter says, prompting confusion about whether their setup is broken in a different way.
- Suggested fix: change the quote to: *"raises `AssertionError: Torch not compiled with CUDA enabled` (on a macOS torch install) or `RuntimeError: Found no NVIDIA driver on your system` (on a Linux/Windows torch install with CUDA support but no driver). The exact message depends on whether your torch wheel was built with CUDA support."* This honest dual-quote fixes the AP-8 issue and educates the reader about the difference.

### Issue #3 — Mismatch — §19.8 exp 4: samples don't diverge at the *first* token
- Section: §19.8 — experiment 4 prose at chapter line 368.
- Where: *"Run generation twice with `--seed 0 --device cpu` then `--seed 0 --device mps`. The samples diverge from the very first generated token."*
- What happened: the two captured samples both begin with `"\nT"` after the prompt `"ROMEO:"`. They share the first 2-3 generated tokens (newline, "T", "h") and diverge at the 4th token (`Thy` vs `That`). The "very first generated token" claim is too strong.
- Why a student would be stuck: not stuck — the samples *do* diverge, just slightly later — but a careful student running the comparison and seeing the shared prefix may wonder if they did the experiment wrong.
- Suggested fix: change "diverge from the very first generated token" to *"diverge within the first few generated tokens (the model is so confident at the start that even with different RNGs the most-likely token wins; once the distribution is less peaked, the device's RNG decides)."* This both honest about the empirical observation and pedagogically richer.

### Issue #4 — Clarity — §19.4 mentions PRNG types without defining them
- Section: §19.4 prose at chapter line 130.
- Where: *"The CPU's Mersenne Twister and MPS's Philox-style PRNG generate different sequences from the same seed."*
- What happened: the math-literate ML-naive reader has met `torch.manual_seed` but not pseudo-random number generator algorithms. "Mersenne Twister" and "Philox-style PRNG" drop in without definition; they're throwaway justifications, but a careful student might pause to look them up.
- Why a student would be stuck: not stuck — the next sentence ("we will see this directly in §19.6") makes the names safely ignorable — but the cold proper nouns interrupt the flow.
- Suggested fix: replace with *"The CPU and MPS use different pseudo-random algorithms internally, so the same seed produces different sequences."* Drop the algorithm names. (Or, if you want to keep them as enrichment, footnote them: *"The CPU uses Mersenne Twister; MPS uses a Philox-family PRNG. Both are well-studied; the names don't matter for our purposes."*)

## Confidence and caveats

I walked the full chapter end-to-end, including the long training runs (§19.6, ~42 s + ~28 s on this M1) and all three of §19.7's generation runs. Every value in the chapter's Expected-Output blocks (loss curves, params=207,296, the three sample texts) matches character-for-character. The three Mismatches are all in the §19.8 experiment block, which gives suggested student activities rather than chapter-canonical outputs — they're real findings, but the chapter's empirical core is sound.

The Clarity finding (Issue #4) is borderline; some authors would ignore it. I include it because the chapter consistently introduces terms before using them in Part I, and this slightly breaks that pattern for two algorithm names that aren't doing pedagogical work.

The cumulative-snapshot setup pattern noted in earlier reviews remains unchanged for Part II — the §19.1 setup tells the student to consume `_state_after_ch18.md`, which itself documents only the chapter's *delta* and chains back through earlier snapshots. Same convention as Ch.10–18; not a new finding.

Once Issues #1–#4 are addressed, the chapter is ready to commit.
