---
title: 18. Checkpoints, inference, and a CLI
nav_order: 19
parent: LLM Fundamentals
---

# Chapter 18 — Checkpoints, inference, and a CLI

This is the last chapter. Up to now every experiment has been a hand-rolled Python file with hard-coded paths, prompts, and hyperparameters. That is fine for learning — the explicit code is the point — but it isn't how you'd *use* the package. The §1.10 promise was:

> by the end of this tutorial you will have a Python package called `mygpt` that you can train on a text file and use to generate text from the command line.

This chapter delivers exactly that. By the end you will be able to type, from any directory:

```bash
uv run mygpt train tinyshakespeare.txt --output shakespeare.ckpt
uv run mygpt generate --checkpoint shakespeare.ckpt --prompt "ROMEO:"
```

…and watch a 207k-parameter character-level GPT train and then generate, with the *same* loss curve and the *same* sample text we observed in Ch.17 — but now driven entirely from the command line, with a single self-contained checkpoint file that bundles the model and its tokenizer together.

What changes in `mygpt`:

- **`save_checkpoint(model, tokenizer, path)`** — bundle weights + tokenizer + architecture config into one `.ckpt` file.
- **`load_checkpoint(path)` → `(model, tokenizer)`** — reload everything from one file.
- **`main()`** is *replaced* with an `argparse`-based dispatcher that exposes `mygpt train` and `mygpt generate` subcommands.

What does *not* change: `GPT`, `CharTokenizer`, `generate`, `get_batch`, `set_seed`. They are already the right shape; we just glue them together.

---

## 18.1 Setup

If you finished Chapter 17, you have the trained Shakespeare model and the corpus already. Nothing to download.

If you skipped, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Overwrite **`src/mygpt/__init__.py`** with the Chapter 17 ending state from `docs/_state_after_ch17.md`. Then re-run §17.2's download:

```bash
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 18.2 Self-contained checkpoints

Until now we have saved the model and the tokenizer to *separate* files: `shakespeare_gpt.pt` (the `state_dict`) and `shakespeare_tokenizer.json` (the alphabet). To reload, the user has to remember:

1. Which two files belong together.
2. What architecture the model was trained with (`vocab_size`, `embed_dim`, `num_heads`, `num_layers`, `max_seq_len`).

Both are easy to get wrong. A real CLI ships *one* file that contains everything needed to reload the model.

We bundle three things into a single Python dict and let `torch.save` serialise it:

```python
{
    "model_state_dict": model.state_dict(),
    "tokenizer_chars":  tokenizer.chars,           # the alphabet, as a list[str]
    "config": {                                    # architecture
        "vocab_size":  ...,
        "embed_dim":   ...,
        "num_heads":   ...,
        "num_layers":  ...,
        "max_seq_len": ...,
    },
}
```

`torch.save` happily serialises a dict whose values are tensors, lists, and ints. On reload, `torch.load` returns the same dict, from which we re-build the tokenizer (`CharTokenizer(chars)`) and the model (`GPT(**config)` plus `load_state_dict`).

**Append the following two functions to** 📄 `src/mygpt/__init__.py` (after `CharTokenizer`, before `main`):

```python
def save_checkpoint(model: "GPT", tokenizer: "CharTokenizer", path: str) -> None:
    """Bundle model weights, tokenizer, and architecture into one .ckpt file."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_chars": tokenizer.chars,
            "config": {
                "vocab_size":  model.vocab_size,
                "embed_dim":   model.embed_dim,
                "num_heads":   model.num_heads,
                "num_layers":  model.num_layers,
                "max_seq_len": model.max_seq_len,
            },
        },
        path,
    )


def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`."""
    ckpt = torch.load(path)
    config = ckpt["config"]
    tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
    model = GPT(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer
```

Two things worth flagging:

- **`dropout=0.0` on reload, always.** Dropout matters during training (it injects noise into activations); during inference it is always disabled. The original training-time dropout is therefore not part of the checkpoint — `load_checkpoint` always reconstructs the model with `dropout=0.0`.
- **The class is reconstructed, not pickled.** We store the *config* (five ints), not the `GPT` object itself. This is the correct way: a pickled object would be brittle to code changes (rename a class, the pickle breaks); a config dict survives any refactor that preserves the constructor signature.

---

## 18.3 The `mygpt train` subcommand

The training loop is the same one we used in §17.5: build a tokenizer from the text, encode the corpus, build a model, run AdamW for `args.steps` steps, save a checkpoint. The only change is that *every* knob comes from `args` (a parsed `argparse.Namespace`) instead of being hard-coded.

The function signature is `_train_command(args) -> None` — it takes the parsed CLI arguments and returns nothing. Naming it with a leading `_` signals "internal helper, not for direct use" — students should call `mygpt train ...` from the shell, not `mygpt._train_command(...)` from Python.

**Append the following function to** 📄 `src/mygpt/__init__.py` (after `load_checkpoint`, before `main`):

```python
def _train_command(args) -> None:
    with open(args.text_file) as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode(text)

    set_seed(0)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"corpus chars: {len(text):,}")
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")

    set_seed(42)
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.seq_len)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.print_every == 0 or step == args.steps:
            print(f"step {step:>5}: loss = {loss.item():.4f}")

    save_checkpoint(model, tokenizer, args.output)
    print(f"\nsaved checkpoint to {args.output}")
```

The two `set_seed(...)` calls match what §17.5's experiment did: seed `0` for model initialisation, seed `42` for the get_batch RNG. Running `mygpt train` with default flags will therefore produce *exactly the same loss curve* as `experiments/39_train_shakespeare.py` did in Ch.17. We will verify this in §18.6.

---

## 18.4 The `mygpt generate` subcommand

Symmetric, simpler: load a checkpoint, encode the prompt, call `generate`, decode, print.

**Append the following function to** 📄 `src/mygpt/__init__.py` (after `_train_command`, before `main`):

```python
def _generate_command(args) -> None:
    model, tokenizer = load_checkpoint(args.checkpoint)
    set_seed(args.seed)
    prompt = tokenizer.encode(args.prompt).unsqueeze(0)
    out = generate(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0]))
```

`set_seed(args.seed)` immediately before generation is what makes the output deterministic — different `--seed` values give different samples from the same model, the same `--seed` always gives the same sample.

---

## 18.5 The argparse dispatcher

`main()` until now has been a one-line hello-world that printed the four-token vocabulary. We *replace* it with an argparse-based dispatcher that recognises two subcommands and routes them to the helpers we just wrote.

**Replace the existing `main()` in** 📄 `src/mygpt/__init__.py` **with**:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mygpt",
        description="Tiny GPT trainer and text generator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a GPT on a plain-text file.")
    p_train.add_argument("text_file", help="Path to a UTF-8 text file.")
    p_train.add_argument("--output", default="model.ckpt", help="Checkpoint output path.")
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--seq-len", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--embed-dim", type=int, default=64)
    p_train.add_argument("--num-heads", type=int, default=4)
    p_train.add_argument("--num-layers", type=int, default=4)
    p_train.add_argument("--max-seq-len", type=int, default=64)
    p_train.add_argument("--dropout", type=float, default=0.0)
    p_train.add_argument("--print-every", type=int, default=500)
    p_train.set_defaults(func=_train_command)

    p_gen = sub.add_parser("generate", help="Generate text from a checkpoint.")
    p_gen.add_argument("--checkpoint", required=True)
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max-new-tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=10)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.set_defaults(func=_generate_command)

    args = parser.parse_args()
    args.func(args)
```

Three things to read off:

- **`sub.add_parser("train", ...)` and `sub.add_parser("generate", ...)`** create *subcommands*. `mygpt train ...` hits `p_train`'s arguments; `mygpt generate ...` hits `p_gen`'s. Anything else gets a usage message.
- **`set_defaults(func=...)`** is the standard argparse trick for dispatch: each subcommand attaches its handler function to `args`, and `main` just calls `args.func(args)` at the end. No `if/elif` chain to maintain.
- **`required=True` on the subparsers** means `mygpt` with no subcommand prints help and exits. (This is the modern argparse default for dispatch; without it, omitting the subcommand silently does nothing.)

The `[project.scripts]` table that `uv init mygpt --package` set up in `pyproject.toml` already contains `mygpt = "mygpt:main"`. So `uv run mygpt ...` calls the new `main()` we just wrote.

`uv` will reinstall the package automatically the next time you run it (`uv` syncs the editable install on demand). No `uv sync` step needed.

---

## 18.6 Use it: train, then generate

Confirm the CLI is wired up:

```bash
uv run mygpt --help
```

**Expected output:**

```text
usage: mygpt [-h] {train,generate} ...

Tiny GPT trainer and text generator.

positional arguments:
  {train,generate}
    train           Train a GPT on a plain-text file.
    generate        Generate text from a checkpoint.

options:
  -h, --help        show this help message and exit
```

Now train. Default flags reproduce the Ch.17 hyperparameters; `--output shakespeare.ckpt` writes the bundled checkpoint:

```bash
uv run mygpt train tinyshakespeare.txt --output shakespeare.ckpt
```

**Expected output:**

```text
corpus chars: 1,115,394
vocab_size:   65
params:       207,296
steps:        2000
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to shakespeare.ckpt
```

(Wall-clock seconds will differ; the loss values will not.)

Compare the loss curve to Ch.17 §17.5: at every shared step (`1`, `500`, `1000`, `2000`) the values match exactly. The CLI is not a different training loop — it is the *same* training loop, with the same seeds, behind a more convenient interface. (The only difference is that the CLI prints at `--print-every 500` by default, so you also see step `1500: 2.1795` here, which Ch.17's hard-coded list of step indices skipped.)

Now generate:

```bash
uv run mygpt generate --checkpoint shakespeare.ckpt --prompt "ROMEO:"
```

**Expected output:**

```text
ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

Compare to Ch.17 §17.6: byte-for-byte identical. We trained from the CLI, saved a single bundled checkpoint, reloaded it from a *different* invocation of the CLI, and recovered the exact same sample. The pipeline is end-to-end reproducible.

---

## 18.7 Experiments

1. **Try a different prompt.** `uv run mygpt generate --checkpoint shakespeare.ckpt --prompt "JULIET:"`. The model produces speech-like text with a new (probably gibberish) speaker label after the first speech.
2. **Cooler sampling.** Add `--temperature 0.5`. The output stays closer to the most-likely token at each step.
3. **A fresh seed.** Add `--seed 1` (vs the default `0`). Different sample, same model.
4. **Train on something else.** Take a small text file you have lying around (a README, a poem, a dump of your shell history) and run `uv run mygpt train your_file.txt --output your_model.ckpt --steps 1000`. With a small corpus and `--steps 1000` the run finishes well under a minute on CPU. Sample with `uv run mygpt generate --checkpoint your_model.ckpt --prompt "<some prefix in your file>"` and watch the model imitate the *style* of your text.
5. **A smaller, faster model.** Train with `--embed-dim 32 --num-heads 2 --num-layers 2 --steps 1000`. Loss plateaus higher (around 2.6 after 1000 steps on Tiny Shakespeare); generation is qualitatively worse but the run finishes in well under a minute.
6. **`mygpt --help` and `mygpt train --help`.** argparse generates `--help` for free at every level. Read both — every flag is documented automatically from the `add_argument` calls in §18.5.

After each experiment, restore any file you changed before moving on.

---

## 18.8 Exercises

1. **What's *in* a checkpoint?** Inspect `shakespeare.ckpt`:
   ```python
   import torch
   ckpt = torch.load("shakespeare.ckpt")
   print(list(ckpt.keys()))
   print(ckpt["config"])
   print(ckpt["tokenizer_chars"][:10])
   print(list(ckpt["model_state_dict"].keys())[:5])
   ```
   You should see the three top-level keys (`model_state_dict`, `tokenizer_chars`, `config`), the architecture dict, the first ten characters of the alphabet, and a few weight-tensor names from the model. This is the entire reload contract.
2. **Why no `--seed` on `train`?** `_train_command` hard-codes `set_seed(0)` for model init and `set_seed(42)` for batch sampling. Argue that exposing those as flags would be a *footgun* for a reproducibility-first tutorial: any change to either seed silently changes the loss curve and the captured Expected Output blocks. (Real CLIs typically expose a single `--seed` and document that the loss curve depends on it; for our pedagogical CLI, fixing the seeds keeps Ch.17/Ch.18 in lock-step.)
3. **Round-trip a 100-character prompt.** `mygpt generate --prompt "..."` encodes the prompt with the *checkpoint's* tokenizer. What happens if you give a prompt that contains a character the tokenizer was never trained on (say, an em-dash)? Trace through `_generate_command` and predict the failure mode. (Hint: it's a `KeyError` from §16.4 — same one we saw in §16.8 exp 2.)
4. **A second subcommand pattern.** Suppose you want to add `mygpt evaluate --checkpoint ckpt.pt --text-file file.txt` that loads a checkpoint and prints the average cross-entropy on a held-out text file. Sketch the `_evaluate_command(args)` function and the `add_parser("evaluate", ...)` block. Don't implement it; the design exercise is the point.

---

## 18.9 Looking back, looking forward

You have built a Python package called `mygpt` that:

- tokenizes arbitrary text at the character level (Ch.16),
- runs a complete decoder-only transformer with weight-tied LM head (Ch.5–13),
- trains via AdamW on a real text corpus (Ch.14, Ch.17),
- samples autoregressively with greedy / temperature / top-k modes (Ch.15),
- packages all of that behind a `mygpt train` / `mygpt generate` CLI with self-contained checkpoints (this chapter).

The §1.10 promise is delivered. You can train `mygpt` on *any* text file and generate from the result.

What this *isn't*: a competitive language model. The 207k-parameter character-level model produces words and rhythm but not meaning; modern LLMs are a million times larger, trained on a thousand times more text, with a more efficient tokenizer (BPE) and a more sophisticated training pipeline (data shuffling, learning-rate schedules, gradient clipping, mixed-precision arithmetic, distributed across many GPUs).

If you want to keep going, three reasonable next steps:

- **Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)** is the natural next package up. It is exactly the same architecture as `mygpt` (in fact, nanoGPT is what `mygpt` is patterned after) but with the production niceties: BPE tokenization via `tiktoken`, GPU support, gradient accumulation, learning-rate decay, validation loss tracking, distributed training.
- **A real tokenizer.** Replace `CharTokenizer` with a BPE tokenizer such as `tiktoken` or `tokenizers`. The model architecture doesn't change at all; only `vocab_size` (now ~50k) and the encode/decode paths.
- **A real corpus.** OpenWebText, FineWeb, The Pile — open datasets in the 10 GB to 10 TB range that real LLMs train on. Tiny Shakespeare is 1 MB; serious training is six to seven orders of magnitude larger.

> **Looking ahead — what to remember from this chapter**
>
> 1. A trained model is meaningless without its tokenizer and architecture config. Always save them together; `save_checkpoint` does this in one file.
> 2. argparse subcommands with `set_defaults(func=...)` is the standard pattern for a multi-mode CLI. No `if/elif` chain.
> 3. Reproducibility comes from seeds at fixed points (`set_seed(0)` for init, `set_seed(42)` for batches, `set_seed(args.seed)` for sampling). Move those seeds and the loss curve and sample text move with them.
> 4. The CLI does not change the math — it is the same training loop and the same generator behind `argparse`. Convenience layers should never *change* what they package.

That's the tutorial. Thanks for reading.
