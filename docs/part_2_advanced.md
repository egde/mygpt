---
title: Part II — Advanced Topics
parent: myGPT
nav_order: 3
has_children: true
---

# Part II — Advanced Topics

Ten chapters that upgrade `mygpt` from a hand-rolled toy to a code base whose architecture and training recipe match real modern open-weight LLMs (Llama-style: BPE + RoPE + RMSNorm + GQA + cosine LR + bf16). Same model size as Part I; modern recipe.

---

## Prerequisites

[Part I — LLM Fundamentals](part_1_fundamentals.html) finished. An Apple M1 / M2 / M3 / M4 Mac (any RAM tier; 8 GB works), or a CUDA GPU, or willingness to wait on CPU. ~10 GB free disk for the Chapter 28 Wikipedia subset.

---

## What changes

- **Ch.19–21** — training infrastructure: device-aware (MPS/CUDA/CPU), mixed precision (bf16), validation loss + cosine LR schedule + gradient clipping.
- **Ch.22–23** — BPE tokenization: build the algorithm from scratch, then wire it into `mygpt` alongside the existing `CharTokenizer`.
- **Ch.24–26** — modern architecture: replace `LayerNorm` with `RMSNorm`, learned position embeddings with **RoPE**, multi-head attention with **GQA**.
- **Ch.27–28** — payoff: same training run as Part I but with the modern stack (Ch.27); then a real ~500 MB Wikipedia training run on M1 in 1–3 hours (Ch.28).

**Backward compatibility:** every Part-I checkpoint continues to load. Architecture flags (`--norm`, `--position`, `--num-kv-heads`, `--tokenizer`) coexist with Part-I defaults; Part-II checkpoints record which combination was used.

---

## Chapters

19. [Device-aware training](19_device_aware_training.html)
20. [Mixed precision training (bf16)](20_mixed_precision.html)
21. [Training-loop hardening](21_training_loop_hardening.html)
22. [BPE from scratch (algorithm)](22_bpe_from_scratch.html)
23. [`BPETokenizer` in `mygpt`](23_bpe_tokenizer_in_mygpt.html)
24. [RMSNorm replaces LayerNorm](24_rmsnorm.html)
25. [RoPE: rotary position embeddings](25_rope.html)
26. [GQA: grouped-query attention](26_gqa.html)
27. Modern recipe vs Ch.17 baseline *(coming soon)*
28. Modern recipe at scale (Wikipedia) *(coming soon)*
