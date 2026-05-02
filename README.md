# myGPT — LLM Fundamentals & Advanced Topics

A 28-chapter code-along tutorial that builds **`mygpt`**, a tiny GPT-2-level language model in PyTorch, from scratch. Part I (Chapters 1–18) covers the fundamentals: tokens, attention, transformer blocks, training, and a CLI. Part II (Chapters 19–28) covers the modern recipe: bf16 mixed precision, cosine LR + warmup + clipping, BPE tokenisation, RMSNorm, RoPE, GQA, and a real ~500 MB Wikipedia training run.

## Two things in this repo

```
.
├── docs/   →  the published tutorial (https://egde.github.io/mygpt/)
└── mygpt/  →  the Python package the tutorial constructs, ready to install
```

### [Read the tutorial on the published site →](https://egde.github.io/mygpt/)

28 chapters, math-literate-but-ML-naive audience, every line of code introduced and explained. The published site has LaTeX math, syntax highlighting, code-block copy buttons, and a chapter-by-chapter sidebar.

### Use the package

```bash
cd mygpt
uv sync

# train a 207k-parameter character-level GPT on Tiny Shakespeare (~45 s on CPU)
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
uv run mygpt train tinyshakespeare.txt --output shakespeare.ckpt

# generate
uv run mygpt generate --checkpoint shakespeare.ckpt --prompt "ROMEO:"
```

See `mygpt/README.md` for full CLI reference and the public API.

## Acknowledgements

Architecture and training are patterned after Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). `mygpt` is intentionally simpler — every file fits in one head.

## Stuck on a chapter? Reset to a known-good state

Each chapter has a corresponding `chapter_states/chNN/` snapshot — a complete, runnable `uv` package matching the end-state of that chapter. If your code stops working partway through, copy the snapshot for the *previous* chapter over your working tree and continue. See [`chapter_states/README.md`](chapter_states/README.md) for the full usage guide.
