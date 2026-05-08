# Student walkthrough report: Chapter 24 — RMSNorm replaces LayerNorm

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/24_rmsnorm.md`
- Total sections walked: 7 of 7 executable subsections (§24.1 setup, §24.3 + §24.4 + §24.5 code edits, §24.6 checkpoint format, §24.7 CLI flag, §24.8 backward-compat run, §24.9 RMS run, §24.10 generate-from-each, §24.11 spot-checks)
- Files modified: 1 — `src/mygpt/__init__.py` (append `RMSNorm` + `_make_norm`, replace `TransformerBlock` + `GPT.__init__` + `save_checkpoint` + `load_checkpoint` to honor `norm_type`, add `--norm` CLI flag, pass `args.norm` through `_train_command`, add `norm:` print line)
- Files produced: 2 — `sh-layer.ckpt`, `sh-rms.ckpt`
- Shell commands run: 7 (uv init, uv add, curl, two `uv run mygpt train`, two `uv run mygpt generate`, plus three ad-hoc `uv run python -c …` for §24.11 exp 1 and exp 3)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.**

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- mps available: True; cuda: False
- Working directory: `/tmp/code-along-runs/ch24-review-142717/mygpt`

## Walkthrough

### Sections §24.1 – §24.7 (setup + code edits)
- §24.3: `RMSNorm` class appended after `LayerNorm`, before `TransformerBlock`. Two-line forward pass (`x.pow(2).mean(...)` + `x / rms * weight`).
- §24.4: `_make_norm` factory function appended after `RMSNorm`.
- §24.5: `TransformerBlock.__init__` and `GPT.__init__` both accept `norm_type="layer"` and dispatch via `_make_norm`. `GPT.ln_f` now uses `_make_norm(...)` instead of `LayerNorm(...)` directly.
- §24.6: `save_checkpoint` writes a new `norm_type` field (with `getattr(model, "norm_type", "layer")` defensive default for pre-Ch.24 models). `load_checkpoint` reads it via `config.get("norm_type", "layer")` to keep pre-Ch.24 `.ckpt` files loading.
- §24.7: `_train_command` passes `norm_type=args.norm` to `GPT(...)` and prints `norm: ...`. New `--norm {layer, rms}` flag on `p_train`.
- All edits land cleanly. `uv run mygpt train --help` shows `[--norm {layer,rms}]`.
- Issues raised here: none.

### Section: §24.8 Backward-compat default (LayerNorm)
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --output sh-layer.ckpt`
- Output (key lines):
  ```
  precision:    fp32
  norm:         layer
  corpus chars: 1,115,394
  …
  params:       207,296
  …
  step     1: loss = 41.0367
  step   500: loss = 2.5944
  step  1000: loss = 2.3529
  step  1500: loss = 2.1795
  step  2000: loss = 2.0785
  ```
- Expected output match: yes — exact, including the new `norm: layer` line. Loss values bit-identical to Ch.21 / Ch.20 / Ch.19 / Ch.17 default runs. Backward-compat preserved.
- Issues: none.

### Section: §24.9 RMSNorm run
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --norm rms --output sh-rms.ckpt`
- Output (key lines):
  ```
  precision:    fp32
  norm:         rms
  …
  params:       206,720
  …
  step     1: loss = 41.2164
  step   500: loss = 2.5935
  step  1000: loss = 2.3517
  step  1500: loss = 2.1689
  step  2000: loss = 2.0752
  ```
- Expected output match: yes — exact. Param count exactly 206,720 (= 207,296 - 576 = 9 × 64 saved across 9 norm instances). Final loss 2.0752 matches the chapter to 4 decimals.
- Issues: none.

### Section: §24.10 Generate from each checkpoint
- `uv run mygpt generate --checkpoint sh-layer.ckpt --prompt "ROMEO:" --device cpu` produces the byte-exact Ch.17 §17.6 sample (`Thy momed has seltered…`). Backward-compat with the *current* CharTokenizer-style checkpoint format works. (A pre-Ch.24 `.ckpt` with no `norm_type` field would also load — the `config.get("norm_type", "layer")` default ensures that — but is not tested here since this iteration produced the LayerNorm ckpt with the new code path.)
- `uv run mygpt generate --checkpoint sh-rms.ckpt --prompt "ROMEO:" --device cpu` produces exactly the chapter-quoted RMSNorm sample (`Thy momed haveseltered ad neards way…`). The first 14 characters `ROMEO:\nThy momed` match the LayerNorm sample (top-k 10 keeps the most-likely token in low-entropy positions); the two diverge at character 15 onward.
- Issues: none.

### Section: §24.11 Experiments — spot checks
- **Exp 1 (saved config):** `torch.load("sh-rms.ckpt")["config"]` returns `{'vocab_size': 65, 'embed_dim': 64, 'num_heads': 4, 'num_layers': 4, 'max_seq_len': 64, 'norm_type': 'rms'}`; the `sh-layer.ckpt` version has `'norm_type': 'layer'`. The chapter's claim "the output includes `'norm_type': 'rms'`" is verified verbatim.
- **Exp 3 (manual RMSNorm):** `RMSNorm(embed_dim=2)(torch.tensor([3.0, 4.0]))` returns `tensor([0.8485, 1.1314])`. The chapter quotes "≈ [0.849, 1.131]" — match to three decimals. The math `sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536` and `[3, 4] / 3.536 ≈ [0.849, 1.131]` is correct.
- Exp 2 (forced wrong norm_type) and Exp 4 (param-count formula derivation) are framed as student investigations rather than precise empirical claims; both are conceptually sound.
- Issues: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end including the two ~30 s training runs (LayerNorm + RMSNorm) and the two generation calls. Every Expected-Output block matches character-for-character, including the new `norm:` print line, the param counts (207,296 vs 206,720), the LayerNorm loss curve (bit-identical to Ch.21), the RMSNorm loss curve, the LayerNorm ROMEO sample (bit-identical to Ch.17 §17.6), and the RMSNorm ROMEO sample. The §24.11 exp 1 config dump and exp 3 numerical formula are both reproduced exactly. The math in §24.2 (LayerNorm vs RMSNorm formulas) is internally consistent and matches the implementation.

§24.12 exercises are conceptual; they were not exhaustively re-derived but the hints (variance identity, Linear-bias-absorption argument, eps-placement gradient analysis) are textbook and correct.

The chapter is ready to commit.
