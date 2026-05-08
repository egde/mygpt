# Student walkthrough report: Chapter 5 — From text to numbers: tokens and embeddings

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/05_tokens_and_embeddings.md`
- Total sections walked: 9 of 11 (§5.1, §5.5, §5.8, §5.11 are prose; the remaining 7 sections all execute cleanly)
- Files created: 4 — `src/mygpt/__init__.py` (rewritten twice), `experiments/10_lookup_vs_onehot.py`, `experiments/11_nn_embedding.py`, `experiments/12_token_embedding.py`
- Shell commands run: 13 (core path + four §5.9 experiments + two §5.10 numeric checks)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** on the first review pass — every Expected-Output block matched exactly, every prose prediction in §5.9 was empirically verified, every quantitative claim in §5.10 holds.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch05-rev1-20260502-051150/mygpt`

## Walkthrough

### Section: §5.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.4 ending state — VOCAB, to_ids, set_seed)
- Commands run:
  ```bash
  uv init mygpt --package
  cd mygpt
  mkdir -p experiments
  uv add torch numpy
  ```
- Expected output match: **yes** — recapitulating the Ch.4 setup runs cleanly with the same dependency list and torch/numpy versions seen in earlier chapters.
- Issues raised here: none

### Section: §5.3 The problem with one-hot
- Files written: `experiments/10_lookup_vs_onehot.py`
- Commands run: `uv run python experiments/10_lookup_vs_onehot.py`
- Output:
  ```text
  Embedding matrix W (V=4, C=4):
  tensor([[-1.1258, -1.1524, -0.2506, -0.4339],
          [ 0.8487,  0.6920, -0.3160, -2.1152],
          [ 0.3223, -1.2633,  0.3500,  0.3081],
          [ 0.1198,  1.2377,  1.1168, -0.2473]])

  one-hot for id=1: tensor([0., 1., 0., 0.])

  one-hot @ W   = tensor([ 0.8487,  0.6920, -0.3160, -2.1152])
  W[1]            = tensor([ 0.8487,  0.6920, -0.3160, -2.1152])
  identical:    True
  ```
- Expected output match: **yes** — exact byte-for-byte match. The one-hot multiplication and the row indexing return identical tensors.
- Issues raised here: none

### Section: §5.4 nn.Embedding
- Files written: `experiments/11_nn_embedding.py`
- Commands run: `uv run python experiments/11_nn_embedding.py`
- Expected output match: **yes** — exact match. Parameter count (16), shapes ((4,4), (4,) → (4,4)), `matches emb.weight[ids]: True`. The seeded `nn.Embedding(4, 4)` produces the exact same matrix as `torch.randn(4, 4)` from §5.3 (both consume the same seed-0 stream).
- Issues raised here: none

### Section: §5.6 Extending mygpt: TokenEmbedding
- Files written: `src/mygpt/__init__.py` (Ch.5 version with `TokenEmbedding` class)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4
  to_ids(['I', 'love', 'AI', '!']) = tensor([0, 1, 2, 3])

  TokenEmbedding(V=4, C=4):
  TokenEmbedding(
    (embedding): Embedding(4, 4)
  )
  params = 16

  te(ids) shape = (4, 4)
  ```
- Expected output match: **yes** — exact match, including the multi-line `nn.Module.__repr__` output for `TokenEmbedding`.
- Issues raised here: none

### Section: §5.7 End-to-end example
- Files written: `experiments/12_token_embedding.py`
- Commands run: `uv run python experiments/12_token_embedding.py`
- Expected output match: **yes** — exact match. Single-sentence rows are exactly `te.embedding.weight` rows; batched output shape is `(2, 4, 4)`.
- Issues raised here: none

### Section: §5.9 Experiments
- Files written: temporary edits to `experiments/12_token_embedding.py` (C=4 → C=32 → restored), `experiments/11_nn_embedding.py` (added mean/std print → restored).
- Commands run:
  ```bash
  uv run python experiments/12_token_embedding.py    # exp 1: C=32
  uv run python experiments/11_nn_embedding.py       # exp 2: mean/std for V=4, C=4
  uv run python -c "import torch; ..."                # exp 2: V=100, C=10
  uv run python -c "import torch; ..."                # exp 3: emb([1,1,1])
  uv run python -c "from mygpt import ..."            # exp 4: zero_grad on uninit grad
  ```
- Expected output match for each:
  - **Exp 1** (C=32): `params = 128`, single-sentence shape `(4, 32)`, batched shape `(2, 4, 32)`. Chapter predicts "shapes printed should become (4, 32) and (2, 4, 32). Parameter count grows from 16 to V·C = 128." ✓
  - **Exp 2** (V=4, C=4): mean = -0.1193, std = 0.9368. Chapter predicts "roughly $-0.12$ and $0.94$ at seed 0". ✓
  - **Exp 2** (V=100, C=10): mean = 0.0289, std = 1.0287. Chapter predicts "around $0.03$ and around $1.03$". ✓
  - **Exp 3** (`emb([1,1,1])`): shape = `(3, 4)`, all three rows identical to `emb.weight[1]`. Chapter predicts "$(3, C)$ and the three rows are identical". ✓
  - **Exp 4** (`zero_grad` on uninitialised grad): `te.embedding.weight.grad is None`. Chapter explains "`.zero_grad()` only resets `.grad` if it has been allocated, which only happens after the first `.backward()`." ✓
- Issues raised here: none

### Section: §5.10 Exercises
- Files written: none (these are reflection / one-off arithmetic).
- Commands run: spot-checks of the GPT-2 numbers used in the exercises.
- Output:
  - Ex 1 verifies that `50257 * 768 = 38597376 ≈ 38.60M`, matching the chapter's prompt of "$V = 50{,}257$, $C = 768$" and consistent with the §5.4 statement "about 38.6 million parameters".
  - Ex 2 prompt says total ~124M, embedding ~38.6M; tied means count once. The implied answer is `38.60 / 124 ≈ 31.1%` of total parameters, which matches widely-published GPT-2-small breakdowns.
- Expected output match: **yes** for both numerical setups.
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§5.2–5.10 inside a fresh `/tmp/code-along-runs/ch05-rev1-20260502-051150/` directory. The `set_seed(0)` discipline holds throughout: with the same seed, `torch.randn(4, 4)` from §5.3 and `nn.Embedding(4, 4).weight` from §5.4 produce exactly the same tensor (because both consume the same first 16 values of the seeded stream), and that same matrix appears for a third time as `te.embedding.weight` in §5.7 — letting a careful student visually confirm the correspondence across all three sections.

Two things worth noting as judgement calls:

1. The §5.4 expected-output block contains the line `Parameter containing:` before the tensor printout, and ends with `requires_grad=True` inside the tensor's repr. PyTorch produces both lines exactly as shown. I did not flag the leading "`Parameter containing:`" preamble as a formatting issue because it appears verbatim in `nn.Embedding`'s default repr.
2. The §5.7 batched output shape is `(2, 4, 4)` — chapter wrote it correctly throughout. (An earlier draft of this chapter, per the commit history, had a meta-pedagogical block with a deliberate `(2, 4, 8)` typo plus commentary; that block has been removed and is not present in the version reviewed here.)

The chapter's pre-review pass — visible in the running ledger at `docs/_review_lessons.md` — caught one issue before this review (an exp 2 mean/std mismatch, fixed at -0.06 → -0.12). That discipline appears to be paying off: this chapter is the first to land at zero findings on review #1.
