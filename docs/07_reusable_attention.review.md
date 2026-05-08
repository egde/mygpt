# Student walkthrough report: Chapter 7 — A reusable attention module

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/07_reusable_attention.md`
- Total sections walked: 8 of 10 (§7.1, §7.10 are prose; the remaining 8 sections all execute cleanly)
- Files created: 4 — `src/mygpt/__init__.py` (rewritten twice), `experiments/16_buffer_vs_param.py`, `experiments/17_dropout_modes.py`, `experiments/18_refactor_equivalence.py`
- Shell commands run: 11 (core path + four §7.8 experiments)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — every Expected-Output block matched exactly; the §7.7 byte-for-byte equivalence with the Chapter 6 version was confirmed; every §7.8 experiment prediction held empirically.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch07-rev1-20260502-053608/mygpt`

## Walkthrough

### Section: §7.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.6 ending state — VOCAB, to_ids, set_seed, TokenEmbedding, SingleHeadAttention v1)
- Commands run: `uv init mygpt --package`; `cd mygpt`; `mkdir -p experiments`; `uv add torch numpy`
- Expected output match: **yes** — recapitulating the Ch.6 setup runs cleanly. The chapter points to `docs/_state_after_ch06.md` as the source of truth for the Ch.6 ending state, which is the right pattern.
- Issues raised here: none

### Section: §7.4 register_buffer in detail
- Files written: `experiments/16_buffer_vs_param.py`
- Commands run: `uv run python experiments/16_buffer_vs_param.py`
- Output:
  ```text
  named_parameters:
    weight: shape=(2,), requires_grad=True

  named_buffers:
    offset: shape=(2,), requires_grad=False

  len(list(m.parameters())) = 1  # only the learnable bit
  len(m.state_dict())       = 2  # parameters AND buffers — both saved
  ```
- Expected output match: **yes** — exact match. The contrast (1 parameter, 2 state-dict entries) demonstrates the buffer/parameter distinction precisely.
- Issues raised here: none

### Section: §7.5 Dropout
- Files written: `experiments/17_dropout_modes.py`
- Commands run: `uv run python experiments/17_dropout_modes.py`
- Output:
  ```text
  x = tensor([[1., 1., 1., 1.],
          [1., 1., 1., 1.]])

  dropout(x) in train mode (random zeros, others scaled by 1/(1-0.5) = 2):
  tensor([[2., 2., 2., 2.],
          [0., 2., 0., 0.]])

  dropout(x) in eval mode (identity — same as input):
  tensor([[1., 1., 1., 1.],
          [1., 1., 1., 1.]])
  ```
- Expected output match: **yes** — exact match (the seed-42 dropout pattern is reproducible).
- Issues raised here: none

### Section: §7.6 Refactoring SingleHeadAttention
- Files written: `src/mygpt/__init__.py` (Ch.7 version with register_buffer + dropout)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4

  Token ids shape:           (1, 4)
  Embedded shape (B, T, C):  (1, 4, 4)
  Attention output (B, T, C): (1, 4, 4)

  TokenEmbedding parameters:        16
  SingleHeadAttention parameters:   64
  SingleHeadAttention buffers:      4096  (causal_mask, not trained)
  Total parameters:                 80
  ```
- Expected output match: **yes** — exact match. Parameter count (64) is unchanged from Ch.6; buffer count is exactly 64 × 64 = 4096 as predicted.
- Issues raised here: none

### Section: §7.7 Equivalence with Chapter 6
- Files written: `experiments/18_refactor_equivalence.py`
- Commands run: `uv run python experiments/18_refactor_equivalence.py`
- Output:
  ```text
  OLD (Chapter 6):
  tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
           [-0.1713,  0.5772,  0.2182, -0.4687],
           [-0.3211,  0.5328,  0.1321, -0.3144],
           [-0.1588,  0.2404,  0.0839, -0.0570]]])

  NEW (Chapter 7):
  tensor([[[-0.3995,  0.5858,  0.1750, -0.5428],
           [-0.1713,  0.5772,  0.2182, -0.4687],
           [-0.3211,  0.5328,  0.1321, -0.3144],
           [-0.1588,  0.2404,  0.0839, -0.0570]]])

  identical:    True
  max abs diff: 0.000e+00
  ```
- Expected output match: **yes** — exact byte-for-byte match. The refactor is mathematically a no-op on the dropout=0 path, as the chapter promises.
- Issues raised here: none

### Section: §7.8 Experiments
- Files written: temporary edits to `experiments/18_refactor_equivalence.py` (max_seq_len=2 → restored, dropout=0.5 → restored).
- Commands run:
  ```bash
  uv run python experiments/18_refactor_equivalence.py    # exp 1: max_seq_len=2 with T=4 input
  uv run python experiments/18_refactor_equivalence.py    # exp 2: dropout=0.5, no .eval()
  uv run python -c "from mygpt import SingleHeadAttention; ..."   # exp 3: buffer device
  uv run python -c "import torch; ..."                             # exp 4: state_dict round-trip
  ```
- Output (exp 1, max_seq_len=2 with T=4): `ValueError: input length T=4 exceeds max_seq_len=2`. The exact substring matches the chapter's prediction.
- Output (exp 2, dropout=0.5 train mode): the new module's output now contains many zeros (e.g. row 1 is `[0, 0, 0, 0]`) and is far from the old output (max abs diff 1.757). Successive runs at the same seed produce different outputs (verified separately). Matches the chapter's prediction "successive runs at the same seed produce different outputs". Confirmed that row sums of attention weights are no longer 1 after dropout (some rows go to 2, others to 0; one example saw `[2.0, 0.0, 0.0, 0.4955]`).
- Output (exp 3, buffer device): `device: cpu` — trivially true on a CPU-only run; chapter notes the CUDA case as conditional on having a GPU machine. ✓
- Output (exp 4, state_dict round-trip): `mask preserved: True` — `torch.save`/`load_state_dict` correctly restores the buffer. ✓
- Expected output match: **yes** for all four experiments.
- Issues raised here: none

### Section: §7.9 Exercises
- Files written: none (these are all reflective / proof-style exercises).
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§7.2–7.8 inside a fresh `/tmp/code-along-runs/ch07-rev1-20260502-053608/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every Expected-Output block and every §7.8 prediction matched the actual machine output. The byte-for-byte equivalence claim in §7.7 — the most important promise of the chapter — held exactly: `torch.equal(out_old, out_new) is True`, with `max abs diff = 0.000e+00`.

A few things worth noting on persona check, none of them findings:

1. **`register_buffer`, `nn.Dropout`, and `module.train()`/`.eval()` are introduced before being used.** ✓
2. **The `max_seq_len = 64` choice is justified** ("comfortably more than the four tokens of `I love AI !`"). ✓
3. **The chapter's main pedagogical move — "no parameter change, byte-for-byte identical output" — is empirically demonstrated** in §7.7, which is exactly the right way to teach a refactor.
4. **Third consecutive chapter to land at zero findings on review #1** (Ch.5, Ch.6, Ch.7). The pre-review checklist + lessons register are paying off; this chapter's pre-review pass empirically verified §7.8 exp 1 and exp 2 before the reviewer ran (visible in the chapter's commit message), exactly the AP-3 discipline the lessons register prescribes.

Pedagogically, this is a "polish chapter" — it does not introduce new mathematics. The chapter's job is to teach a piece of `nn.Module` engineering (`register_buffer`) and a regularisation operation (`nn.Dropout`) without changing any of the math. Both are taught with their own dedicated experiment (16 and 17 respectively) before the refactor itself in §7.6, which is the right ordering.
