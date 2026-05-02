# mygpt

A tiny GPT-2-level language model built from scratch in PyTorch. This is the package that the **[myGPT](https://egde.github.io/mygpt/)** tutorial constructs chapter by chapter — 18 chapters of LLM Fundamentals (Part I) plus 10 chapters of Advanced Topics (Part II) covering RoPE, GQA, RMSNorm, BPE, bf16 training, and a real Wikipedia run. Every line of code in this package is introduced and explained in one of the 28 tutorial chapters.

## Quick start

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
```

### Train

```bash
# 1. download a plain-text corpus (Tiny Shakespeare in this example)
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2. train a 207k-parameter character-level GPT (~30 s on M1 MPS, ~45 s on CPU)
uv run mygpt train tinyshakespeare.txt --output shakespeare.ckpt
```

The default recipe is the Ch.17 baseline (LayerNorm + learned position embeddings + full multi-head attention + fp32 + constant LR + char tokenizer). For the Part-II modern recipe (RMSNorm + RoPE + GQA + bf16 + cosine LR + warmup + clipping):

```bash
uv run mygpt train tinyshakespeare.txt --device mps \
    --precision bf16 \
    --schedule cosine --warmup 100 --max-grad-norm 1.0 \
    --norm rms --position rope --num-kv-heads 2 \
    --output sh-modern.ckpt
```

For Wikipedia-scale BPE training (Ch.28 finale):

```bash
uv run mygpt train wikipedia.txt --device mps \
    --tokenizer bpe --num-merges 1024 --bpe-train-bytes 50000000 \
    --embed-dim 192 --num-layers 4 --num-heads 6 --num-kv-heads 2 --max-seq-len 256 \
    --batch-size 16 --seq-len 256 \
    --norm rms --position rope --precision bf16 \
    --schedule cosine --warmup 500 --max-grad-norm 1.0 \
    --steps 10000 --print-every 200 --checkpoint-every 2500 \
    --output wiki.ckpt
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

## Package layout

```
mygpt/
├── pyproject.toml              # uv-managed package metadata (version 0.28.0)
├── README.md
└── src/mygpt/                  # multi-file layout (introduced from Ch.5)
    ├── __init__.py             # re-export shim
    ├── attention.py            # SingleHeadAttention, MultiHeadAttention,
    │                           # precompute_rope_cache, apply_rope
    ├── block.py                # TransformerBlock
    ├── checkpoint.py           # save_checkpoint, load_checkpoint
    ├── cli.py                  # main, _train_command, _generate_command
    ├── embedding.py            # TokenEmbedding
    ├── generate.py             # generate
    ├── mlp.py                  # MLP
    ├── model.py                # GPT
    ├── norm.py                 # LayerNorm, RMSNorm, _make_norm
    ├── tokenizer.py            # CharTokenizer, BPETokenizer
    └── utils.py                # set_seed, pick_device, get_batch,
                                # cosine_warmup_lr, estimate_val_loss, VOCAB
```

The chapter snapshots in `../chapter_states/chNN/` mirror this layout for every chapter from Ch.5 onwards (Ch.1–4 are too small to be worth splitting and stay monolithic). Reader-facing experiments live under `../chapter_states/chNN/experiments/` — each snapshot bundles the cumulative experiment set up to that chapter.

## Public API

The `mygpt` package exports the following from its submodules. Import either via the top-level shim (`from mygpt import GPT`) or the submodule directly (`from mygpt.model import GPT`).

| Name                         | Module          | Introduced       | Purpose                                                                  |
|------------------------------|-----------------|------------------|--------------------------------------------------------------------------|
| `set_seed`                   | `utils`         | Ch. 4            | Seed PyTorch's RNG                                                       |
| `pick_device`                | `utils`         | Ch. 19           | Resolve `auto` / `cuda` / `mps` / `cpu`                                  |
| `get_batch`                  | `utils`         | Ch. 14           | Sample a `(B, T)` (input, target) batch from a 1-D corpus tensor         |
| `cosine_warmup_lr`           | `utils`         | Ch. 21           | Cosine LR with linear warmup                                             |
| `estimate_val_loss`          | `utils`         | Ch. 21           | Mean cross-entropy over `n_eval` random batches of `val_data`            |
| `TokenEmbedding`             | `embedding`     | Ch. 5            | `(B, T)` → `(B, T, C)`                                                   |
| `SingleHeadAttention`        | `attention`     | Ch. 6, 7         | `(B, T, C)` → `(B, T, C)` causal self-attention (1 head)                 |
| `MultiHeadAttention`         | `attention`     | Ch. 8 + Ch. 25 (RoPE) + Ch. 26 (GQA) | `(B, T, C)` → `(B, T, C)` causal self-attention (h heads, optional GQA + RoPE) |
| `precompute_rope_cache`      | `attention`     | Ch. 25           | Build cos/sin tables for rotary position embeddings                      |
| `apply_rope`                 | `attention`     | Ch. 25           | Apply position-dependent rotation to Q or K                              |
| `MLP`                        | `mlp`           | Ch. 9            | `(B, T, C)` → `(B, T, C)` position-wise feed-forward                     |
| `LayerNorm`                  | `norm`          | Ch. 10           | `(..., C)` → `(..., C)` per-token mean/std normalisation                 |
| `RMSNorm`                    | `norm`          | Ch. 24           | RMS normalisation (no mean subtraction, no bias)                         |
| `TransformerBlock`           | `block`         | Ch. 11           | `(B, T, C)` → `(B, T, C)` pre-norm + residual (mha + mlp)                |
| `GPT`                        | `model`         | Ch. 12, 13       | `(B, T)` → `(B, T, V)` complete decoder-only transformer                 |
| `generate`                   | `generate`      | Ch. 15           | Autoregressive sampler with greedy / temperature / top-k modes           |
| `CharTokenizer`              | `tokenizer`     | Ch. 16           | Character-level encode / decode / save / load                            |
| `BPETokenizer`               | `tokenizer`     | Ch. 23 + Ch. 28 (fast path) | Word-level BPE; `from_corpus` + `encode_corpus` for real-text scale       |
| `save_checkpoint`            | `checkpoint`    | Ch. 18 + Ch. 28 (BPE state, atomic save) | Bundle (model, tokenizer, config) into one `.ckpt`            |
| `load_checkpoint`            | `checkpoint`    | Ch. 18 + Ch. 28 | Reload `(model, tokenizer)` from a `.ckpt`; backward-compat for every Ch.18+ checkpoint |
| `main`                       | `cli`           | Ch. 18           | argparse dispatcher with `train` and `generate` subcommands              |

## CLI flags introduced per chapter

| Flag                    | Default     | Introduced |
|-------------------------|-------------|------------|
| `--device {auto,cuda,mps,cpu}` | `auto`  | Ch. 19     |
| `--precision {fp32,bf16}` | `fp32`     | Ch. 20     |
| `--val-split FRAC`      | `0.0`       | Ch. 21     |
| `--val-every N`         | `0`         | Ch. 21     |
| `--schedule {constant,cosine}` | `constant` | Ch. 21 |
| `--warmup STEPS`        | `0`         | Ch. 21     |
| `--max-grad-norm FLOAT` | `0.0` (off) | Ch. 21     |
| `--norm {layer,rms}`    | `layer`     | Ch. 24     |
| `--position {learned,rope}` | `learned` | Ch. 25    |
| `--num-kv-heads N`      | `= num_heads` | Ch. 26   |
| `--tokenizer {char,bpe}` | `char`     | Ch. 28     |
| `--num-merges N`        | `1024`      | Ch. 28     |
| `--bpe-train-bytes N`   | `0` (full)  | Ch. 28     |
| `--checkpoint-every N`  | `0` (off)   | Ch. 28     |

Defaults preserve the Ch.17 baseline so every flag is opt-in; the modern recipe is `--norm rms --position rope --num-kv-heads 2 --precision bf16 --schedule cosine --warmup 500 --max-grad-norm 1.0` (`--tokenizer bpe` for the Wikipedia run).

## Read the tutorial

The package is best understood alongside the chapters that build it:

**https://egde.github.io/mygpt/**

## Acknowledgements

The model architecture, training loop, and the Tiny Shakespeare example are patterned after Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). `mygpt` is a *teaching* package — every concept appears as the simplest possible code that demonstrates it (no FlashAttention, no KV cache, no distributed training).

## License

[MIT](../LICENSE).
