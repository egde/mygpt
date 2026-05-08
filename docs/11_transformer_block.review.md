# Student walkthrough report: Chapter 11 — The transformer block

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/11_transformer_block.md`
- Total sections walked: 6 of 8 (§11.1, §11.8 are prose; the remaining 6 sections all execute cleanly)
- Files created: 2 — `src/mygpt/__init__.py` (rewritten with full Ch.10 state plus `TransformerBlock`), `experiments/24_stack_two_blocks.py`
- Shell commands run: 8 (core path + four §11.6 experiments + §11.7 ex 4 verification)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly, every §11.6 experiment prediction held empirically, every §11.7 quantitative claim held.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch11-rev1-20260502-061949/mygpt`

## Walkthrough

### Section: §11.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.10 ending state — VOCAB, to_ids, set_seed, TokenEmbedding, SingleHeadAttention, MultiHeadAttention, MLP, LayerNorm)
- Commands run: `uv init mygpt --package`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes**
- Issues raised here: none

### Section: §11.4 Verifying the block
- Files written: `src/mygpt/__init__.py` (with `TransformerBlock` appended after `LayerNorm` and `main` updated)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4

  Token ids shape:           (1, 4)
  Embedded shape (B, T, C):  (1, 4, 4)
  Block output shape:        (1, 4, 4)

  TransformerBlock(
    (ln1): LayerNorm()
    (mha): MultiHeadAttention(
      (W_Q): Linear(in_features=4, out_features=4, bias=False)
      (W_K): Linear(in_features=4, out_features=4, bias=False)
      (W_V): Linear(in_features=4, out_features=4, bias=False)
      (W_O): Linear(in_features=4, out_features=4, bias=False)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (out_drop): Dropout(p=0.0, inplace=False)
    )
    (ln2): LayerNorm()
    (mlp): MLP(
      (fc1): Linear(in_features=4, out_features=16, bias=True)
      (act): GELU(approximate='none')
      (fc2): Linear(in_features=16, out_features=4, bias=True)
      (drop): Dropout(p=0.0, inplace=False)
    )
  )

  TokenEmbedding parameters:   16
  TransformerBlock parameters: 228
  Total parameters:            244
  ```
- Expected output match: **yes** — exact match. The `228` parameter count matches the §11.1 formula $12 C^2 + 9 C = 192 + 36 = 228$ for $C=4$.
- Issues raised here: none

### Section: §11.5 Stacking blocks
- Files written: `experiments/24_stack_two_blocks.py`
- Commands run: `uv run python experiments/24_stack_two_blocks.py`
- Output:
  ```text
  input shape:    (1, 4, 4)
  after 2 blocks: (1, 4, 4)

  TokenEmbedding params: 16
  2 TransformerBlocks:   456  (= 2 * 228)
  Total:                 472
  ```
- Expected output match: **yes** — exact match. `2 * 228 = 456` confirmed; shape preserved across the stack.
- Issues raised here: none

### Section: §11.6 Experiments
- Files written: temporary inline edits.
- Commands run:
  ```bash
  uv run python -c "from mygpt import TransformerBlock; b = TransformerBlock(4, 1); print(sum(p.numel() for p in b.parameters()))"
  uv run python -c "from mygpt import TransformerBlock; b = TransformerBlock(8, 2); print(sum(p.numel() for p in b.parameters()))"
  uv run python -c "..."   # max abs diff at random init
  uv run python -c "..."   # max abs diff after zeroing W_O.weight, fc2.weight, fc2.bias
  ```
- Output:
  - **Exp 1** (1-head block): `params: 228`. Matches the chapter's prediction (param count is independent of `num_heads`).
  - **Exp 2** (`TransformerBlock(8, 2)`): `params: 840`. Matches the chapter's prediction $12 \cdot 64 + 9 \cdot 8 = 840$ exactly.
  - **Exp 3** (random init, `block(x) - x`): `max abs diff: 0.9104`. Chapter says "of order 1"; 0.9104 fits.
  - **Exp 4** (zero W_O.weight + fc2.weight + fc2.bias → identity): `max abs diff: 0.0000e+00`. Matches "should be ~0 (within float32 noise)".
- Expected output match: **yes** for all four.
- Issues raised here: none

### Section: §11.7 Exercises
- Files written: none (these are reflective).
- Commands run: spot-check of ex 4 — `T=65` with `max_seq_len=64` raises `ValueError: input length T=65 exceeds max_seq_len=64`; `T ∈ {1, 2, 4, 16, 64}` all preserve shape `(1, T, 4)`. Both matches the chapter.
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§11.2–11.7 inside a fresh `/tmp/code-along-runs/ch11-rev1-20260502-061949/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every Expected-Output block matched exactly, including the full `nn.Module.__repr__` tree of `TransformerBlock` (which lists `ln1`, `mha`, `ln2`, `mlp` in that order, with their internal layers), and the four §11.6 numerical predictions (228, 840, ~0.91, ~0).

A few observations on persona check, none of them findings:

1. **The chapter is composition, not new mathematics.** Every concept (LayerNorm, MultiHeadAttention, MLP, residual connections, pre-norm) was introduced in earlier chapters. The block is `four lines of forward, four sub-modules, two residuals` — and the chapter is honest about this.
2. **Pre-norm vs post-norm reasoning is referenced back to §10.5** rather than re-derived. ✓ Cleaner than re-explaining; the student has it just one chapter back.
3. **Seventh consecutive chapter to land at zero findings on review #1** (Ch.5–11). The pre-review checklist + lessons register continue to pay off; this chapter's pre-review pass empirically verified all four §11.6 experiments and the §11.7 ex 4 ValueError before the reviewer ran (visible in the chapter's commit message).
4. **The chapter's parameter formula `12 C^2 + 9 C` is reproducible from prior chapters' formulas.** A student tracking parameters chapter-by-chapter has every term: $4 C^2$ from MHA (Ch.8), $8 C^2 + 5 C$ from MLP (Ch.9), $2 \cdot 2 C$ from two LayerNorms (Ch.10). Sum: $12 C^2 + 9 C$. Verified empirically: $228$ for $C=4$, $840$ for $C=8$.

Pedagogically this is the keystone chapter for Part III: after Chapter 11 the body of GPT-2 is complete. The last two pieces (position embeddings and the language-modelling head) come in Chapter 12, and the chapter's "What's next" preview points to that.
