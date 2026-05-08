# Student walkthrough report: Chapter 25 — RoPE: rotary position embeddings

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/25_rope.md`
- Total sections walked: 7 of 7 executable subsections (§25.1 setup, §25.4 helpers, §25.5 class threading, §25.6 checkpoint format, §25.7 CLI, §25.8 backward-compat, §25.9 RoPE run, §25.11 sampling, §25.12 spot-checks)
- Files modified: 1 — `src/mygpt/__init__.py` (append `precompute_rope_cache` + `apply_rope`, replace `MultiHeadAttention` + `TransformerBlock` + `GPT.__init__` and `GPT.forward` + `save_checkpoint` + `load_checkpoint`, add `--position` CLI flag, pass `args.position` through `_train_command`)
- Files produced: 2 — `sh-learned.ckpt`, `sh-rope.ckpt`
- Shell commands run: 8 (uv init, uv add, curl, two `uv run mygpt train`, two `uv run mygpt generate`, three `uv run python` ad-hoc verifications)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.**

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- mps available: True; cuda: False
- Working directory: `/tmp/code-along-runs/ch25-review-145701/mygpt`

## Walkthrough

### Sections §25.1 – §25.7 (setup + code edits)
- §25.4: `precompute_rope_cache` and `apply_rope` appended after the existing module classes (before `MultiHeadAttention`).
- §25.5: `MultiHeadAttention.__init__` accepts `position_type="learned"`, conditionally registers `rope_cos`/`rope_sin` buffers; `forward` conditionally applies RoPE to Q and K after the head split.
- `TransformerBlock.__init__` and `GPT.__init__` both accept `position_type` and pass it through. `GPT` conditionally allocates `position_embedding` only when `position_type == "learned"`; `GPT.forward` skips the `+ position_embedding(positions)` add when not learned.
- §25.6: `save_checkpoint` writes a new `position_type` field (with `getattr` defensive default for pre-Ch.25 models). `load_checkpoint` reads it via `config.get("position_type", "learned")` for backward compat.
- §25.7: `_train_command` passes `position_type=args.position` to `GPT(...)` and prints `position: ...`. New `--position {learned, rope}` flag on `p_train` (default `learned`).
- All edits land cleanly; `uv run mygpt train --help` shows `[--position {learned,rope}]`.
- Issues raised here: none.

### Section: §25.8 Backward-compat default run
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --output sh-learned.ckpt`
- Output (key lines):
  ```
  precision:    fp32
  norm:         layer
  position:     learned
  …
  params:       207,296
  …
  step     1: loss = 41.0367
  step   500: loss = 2.5944
  step  1000: loss = 2.3529
  step  1500: loss = 2.1795
  step  2000: loss = 2.0785
  ```
- Expected output match: yes — exact, including the new `position: learned` line. Loss values bit-identical to Ch.24/Ch.21/Ch.17 default runs. Backward-compat preserved.
- Issues raised here: none.

### Section: §25.9 RoPE run
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --position rope --output sh-rope.ckpt`
- Output (key lines):
  ```
  position:     rope
  …
  params:       203,200
  …
  step     1: loss = 55.6207
  step   500: loss = 2.3110
  step  1000: loss = 1.9569
  step  1500: loss = 1.8594
  step  2000: loss = 1.7812
  ```
- Expected output match: yes — exact. Param count drops by 4,096 (= 64 × 64 = `max_seq_len * embed_dim` of the dropped position-embedding table). Final loss 1.7812 matches the chapter's claim of "substantially better" than learned (2.0785).
- Issues raised here: none.

### Section: §25.11 Sampling
- `mygpt generate --checkpoint sh-learned.ckpt --prompt "ROMEO:" --device cpu` produces the byte-exact Ch.17 §17.6 sample (`Thy momed has seltered…`). Backward-compat preserved with the current code path.
- `mygpt generate --checkpoint sh-rope.ckpt --prompt "ROMEO:" --device cpu` produces exactly the chapter-quoted RoPE sample (`Thy whis whaths be dedood nevery words welevet and furwight nawaren?…`).
- Issues raised here: none.

### Section: §25.12 — spot checks
- **Exp 2 (cos cache):** `precompute_rope_cache(head_dim=16, max_seq_len=8)` returns `cos[0] = [1, 1, 1, 1, 1, 1, 1, 1]` (all 1s, position 0) and `cos[1] = [0.5403, 0.9504, 0.9950, 0.9995, 0.9999, 1.0000, 1.0000, 1.0000]`. The first column at position 1 is `cos(1) ≈ 0.5403` and the last column trails off to 1.0 — exactly the spectrum the chapter describes.
- **Exp 3 (identity at position 0):** `apply_rope(torch.randn(2, 4, 1, 16), cos, sin)` equals the input exactly (`torch.allclose(out, x)` returns `True`). Verifies the rotation is identity at position 0.
- Issues raised here: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end including the two ~30 s training runs (learned + RoPE) and verified every Expected Output block character-for-character. The new `position:` print line, both param counts (207,296 / 203,200), the bit-identical Ch.21 LayerNorm-learned default loss curve, the RoPE loss curve (41/55/2.31/1.96/1.86/1.78), and both ROMEO samples reproduce exactly. The §25.12 exp 2 (cos lookup spectrum) and exp 3 (identity at position 0) both pass.

The math in §25.3 (rotation matrix formulation, geometric ladder, relative-position dot-product property) is internally consistent and matches the implementation in §25.4. The §25.10 "killer property" claim about generation past trained seq_len is presented in prose without an empirical demonstration, but the chapter explicitly notes this and includes the recipe in §25.12 exp 1 — that's a reasonable trade since the demo would require another retraining round.

The chapter is ready to commit.
