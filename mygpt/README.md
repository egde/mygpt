# mygpt

A tiny GPT-2-level language model built from scratch in PyTorch. This is the package that the **[LLM Fundamentals](https://egde.github.io/mygpt/)** tutorial constructs chapter by chapter. Every line of code in `src/mygpt/__init__.py` is introduced and explained in one of the 18 tutorial chapters.

## Quick start

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
```

### Train

```bash
# 1. download a plain-text corpus (Tiny Shakespeare in this example)
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2. train a 207k-parameter character-level GPT (~45 s on CPU)
uv run mygpt train tinyshakespeare.txt --output shakespeare.ckpt
```

### Generate

```bash
uv run mygpt generate --checkpoint shakespeare.ckpt --prompt "ROMEO:"
```

### Tweak

Both subcommands have flags for every hyperparameter. List them:

```bash
uv run mygpt --help
uv run mygpt train --help
uv run mygpt generate --help
```

## What's in here

```
mygpt/
├── pyproject.toml              # uv-managed package metadata
├── src/mygpt/__init__.py       # the entire package: every class, function, and the CLI
└── experiments/                # standalone scripts referenced from the tutorial
    ├── 01_hello_mygpt.py       # Ch.2 — first uv-managed run
    ├── 02_tensors.py           # Ch.3 — PyTorch primer
    ├── ...                     # one or more per chapter
    └── 40_generate_shakespeare.py  # Ch.17 — sample from the trained Shakespeare model
```

Each experiment is a standalone `uv run python experiments/NN_name.py` invocation; the tutorial chapter that introduces it explains what it does, what to expect, and what it teaches.

## Public API

The `mygpt` package exports:

| Name                  | Type        | Introduced | Purpose                                                              |
|-----------------------|-------------|------------|----------------------------------------------------------------------|
| `set_seed`            | function    | Ch. 4      | Seed PyTorch's RNG                                                   |
| `get_batch`           | function    | Ch. 14     | Sample a `(B, T)` (input, target) batch from a 1-D corpus tensor     |
| `TokenEmbedding`      | nn.Module   | Ch. 5      | `(B, T)` → `(B, T, C)`                                               |
| `SingleHeadAttention` | nn.Module   | Ch. 6, 7   | `(B, T, C)` → `(B, T, C)` causal self-attention (1 head)             |
| `MultiHeadAttention`  | nn.Module   | Ch. 8      | `(B, T, C)` → `(B, T, C)` causal self-attention (h heads)            |
| `MLP`                 | nn.Module   | Ch. 9      | `(B, T, C)` → `(B, T, C)` position-wise feed-forward                 |
| `LayerNorm`           | nn.Module   | Ch. 10     | `(..., C)` → `(..., C)` per-token mean/std normalisation             |
| `TransformerBlock`    | nn.Module   | Ch. 11     | `(B, T, C)` → `(B, T, C)` pre-norm + residual block (mha + mlp)      |
| `GPT`                 | nn.Module   | Ch. 12, 13 | `(B, T)` → `(B, T, V)` complete decoder-only transformer             |
| `generate`            | function    | Ch. 15     | Autoregressive sampler with greedy / temperature / top-k modes       |
| `CharTokenizer`       | class       | Ch. 16     | Character-level encode / decode / save / load                        |
| `save_checkpoint`     | function    | Ch. 18     | Bundle (model, tokenizer, config) into one `.ckpt` file              |
| `load_checkpoint`     | function    | Ch. 18     | Reload `(model, tokenizer)` from a `.ckpt` file                      |
| `main`                | function    | Ch. 18     | argparse dispatcher with `train` and `generate` subcommands          |

## Read the tutorial

The package is best understood alongside the chapters that build it. The full tutorial is published at:

**https://egde.github.io/mygpt/**

## Acknowledgements

The model architecture, training loop, and the Tiny Shakespeare example are patterned after Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). `mygpt` is a *teaching* package — it is intentionally simpler than nanoGPT (no GPU support, no BPE, no learning-rate schedule) so every line can be read in one sitting.

## License

This is a tutorial artifact. Copy, modify, and learn from it freely.
