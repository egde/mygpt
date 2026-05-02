# mygpt — LLM Fundamentals

A 18-chapter code-along tutorial that builds **`mygpt`**, a tiny GPT-2-level character-level language model in PyTorch, from scratch.

## Two things in this repo

```
.
├── docs/   →  the published tutorial (https://egde.github.io/mygpt/)
└── mygpt/  →  the Python package the tutorial constructs, ready to install
```

### [Read the tutorial →](https://egde.github.io/mygpt/)

18 chapters, math-literate-but-ML-naive audience. Every line of code in the package is introduced and explained.

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
