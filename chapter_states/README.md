# `chapter_states/`

Per-chapter snapshots of the `mygpt` package.

## What this directory is

Each `chNN/` subdirectory is a self-contained `uv` package matching the
end-state of Chapter `NN` in the tutorial. It contains exactly what your own
`mygpt/` directory should look like once you finish that chapter:

- `pyproject.toml` — `version = "0.NN.0"`, `description = "Chapter NN: <title>"`,
  with the dependencies installed by that point.
- `src/mygpt/` — the package source. From Chapter 5 onwards this is a multi-file
  layout (`attention.py`, `block.py`, `embedding.py`, `mlp.py`, `model.py`,
  `norm.py`, `tokenizer.py`, `generate.py`, `checkpoint.py`, `utils.py`,
  `cli.py`), with `__init__.py` reduced to a re-export shim. The chapter prose
  guides you through editing each file directly. Ch.27's snapshot reproduces
  the canonical `step 2000: loss = 2.0785` on Tiny Shakespeare bit-for-bit.
  Chapters 1–4 stay monolithic because the code is too small to be worth splitting.
- `experiments/` — the cumulative set of experiment scripts the tutorial has
  asked you to save by that point (copied byte-for-byte from
  `experiments/`).

## When to use it

If you are stuck on Chapter N, your `__init__.py` looks wrong, or you can't
figure out which edit broke things, **`chapter_states/chNN/` is the canonical
end-state**. Copy it over your working tree and pick up from there.

## How to use it

From the repository root:

```bash
# 1. (one time) make sure your working dir is empty / archived elsewhere
rm -rf my_working_mygpt

# 2. copy the snapshot you need
cp -r chapter_states/ch10 my_working_mygpt
cd my_working_mygpt

# 3. install
uv sync
```

From here on, `uv run mygpt` and `uv run python experiments/<n>_<name>.py`
behave exactly as the chapter ends.

If a chapter requires a downloaded data file (e.g. `tinyshakespeare.txt` from
Chapter 17 onwards), the snapshot does **not** include it — re-run the `curl`
command from that chapter to fetch it.

## Snapshot index

| Snapshot | Chapter | Title |
|----------|---------|-------|
| `ch01` | Chapter 1 | What is a language model? |
| `ch02` | Chapter 2 | Project setup with `uv` |
| `ch03` | Chapter 3 | PyTorch in 20 minutes: tensors, autograd, modules |
| `ch04` | Chapter 4 | How machines learn: loss, gradients, gradient descent |
| `ch05` | Chapter 5 | From text to numbers: tokens and embeddings |
| `ch06` | Chapter 6 | Single-head self-attention from scratch |
| `ch07` | Chapter 7 | A reusable attention module |
| `ch08` | Chapter 8 | Multi-head attention |
| `ch09` | Chapter 9 | The feed-forward network and residual connections |
| `ch10` | Chapter 10 | Layer normalization |
| `ch11` | Chapter 11 | Putting it together: the transformer block |
| `ch12` | Chapter 12 | Position embeddings and the language modeling head |
| `ch13` | Chapter 13 | The forward pass with loss |
| `ch14` | Chapter 14 | Training loop: gradient descent in practice |
| `ch15` | Chapter 15 | Generation: sampling text from a trained model |
| `ch16` | Chapter 16 | A reusable character tokenizer |
| `ch17` | Chapter 17 | Training on a real text file |
| `ch18` | Chapter 18 | Checkpoints, inference, and a CLI |
| `ch19` | Chapter 19 | Device-aware training (MPS / CUDA / CPU) |
| `ch20` | Chapter 20 | Mixed precision training (bf16) |
| `ch21` | Chapter 21 | Training-loop hardening |
| `ch22` | Chapter 22 | Byte-pair encoding from scratch |
| `ch23` | Chapter 23 | `BPETokenizer` in `mygpt` |
| `ch24` | Chapter 24 | RMSNorm replaces LayerNorm |
| `ch25` | Chapter 25 | RoPE: rotary position embeddings |
| `ch26` | Chapter 26 | GQA: grouped-query attention |
| `ch27` | Chapter 27 | Modern recipe vs Ch.17 baseline |
| `ch28` | Chapter 28 | Modern recipe at scale (Wikipedia) |

## How this directory is built

`scripts/build_chapter_states.py` rebuilds the entire tree from scratch. It is
deterministic: re-running it on the same repo produces identical output.
