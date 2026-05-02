---
title: myGPT
nav_order: 1
has_children: true
---

# myGPT

A code-along book that builds a tiny GPT-2-level language model from scratch in PyTorch. Two parts:

- **[Part I — LLM Fundamentals](part_1_fundamentals.html)** (Chapters 1–18). Build the package end-to-end on CPU. By the end you have a `mygpt` CLI that trains and generates from any text file.
- **[Part II — Advanced Topics](part_2_advanced.html)** (Chapters 19–28). Modernize the architecture and training recipe (BPE + RoPE + RMSNorm + GQA + cosine LR + bf16) on Apple M1 / CUDA / CPU. By the end your `mygpt` matches the recipe of real open-weight LLMs at toy scale.

The package itself lives at [github.com/egde/mygpt](https://github.com/egde/mygpt). Read the parts in order; Part II assumes Part I.

> **Stuck on a chapter?** Each of the 28 chapters has a corresponding `chapter_states/chNN/` snapshot — a complete, runnable `uv` package matching the end-state of that chapter. If your code stops working partway through, copy the snapshot for the *previous* chapter over your working tree and continue. See [`chapter_states/README.md`](https://github.com/egde/mygpt/tree/main/chapter_states) on GitHub for the full usage guide.

---

The running example throughout the book is the four tokens

```text
I  love  AI  !
```

We use these tokens over and over again, because they are short enough to inspect by hand and rich enough to demonstrate every idea — from probability distributions to attention.
