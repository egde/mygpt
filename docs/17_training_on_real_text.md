---
title: 17. Training on a real text file
nav_order: 18
parent: Part I — LLM Fundamentals
---

# Chapter 17 — Training on a real text file

Up to now everything has run on toy data — the four-token cycle from Ch.14, the same four words tokenized at character level in Ch.16. This chapter throws a real corpus at the pipeline:

- **Tiny Shakespeare**: ~1.1 MB of plain-text Shakespeare collected by Andrej Karpathy. 65 distinct characters, ~1.1 million tokens at character level.
- **A 207k-parameter `GPT`** sized to fit on a laptop CPU.
- **A real training run**: 2,000 AdamW steps in under a minute on CPU.
- **A real sample**: 200 generated tokens that *look like* Shakespeare — capitalised speakers, line breaks, vaguely English vocabulary — even though every word is invented.

Nothing in `mygpt` changes for this chapter. Everything we built in Ch.5–16 is now used together against a non-toy corpus: `CharTokenizer` (Ch.16), `GPT` (Ch.12–13), `get_batch` (Ch.14), AdamW training loop (Ch.14), `generate` (Ch.15).

---

## 17.1 Setup

If you finished Chapter 16, you already have everything: `mygpt/` with `CharTokenizer`, `GPT`, `get_batch`, `generate`. No package source changes in this chapter.

If you skipped, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Overwrite **`src/mygpt/__init__.py`** with the Chapter 16 ending state from `docs/_state_after_ch16.md`.

You are ready.

---

## 17.2 The corpus

Tiny Shakespeare is a 1.1 MB plain-text concatenation of Shakespeare's plays formatted as

```text
SPEAKER:
Lines of dialogue.

OTHER SPEAKER:
More lines.
```

Download it. From the project root:

```bash
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

`curl -s` (silent) prints nothing on success; you should be returned to the prompt with `tinyshakespeare.txt` of about 1.1 MB sitting in the directory. To confirm:

```bash
wc -c tinyshakespeare.txt
head -3 tinyshakespeare.txt
```

**Expected output:**

```text
 1115394 tinyshakespeare.txt
First Citizen:
Before we proceed any further, hear me speak.
```

That's 1,115,394 bytes (≈ 1.1 MB) and the first three lines are a speaker label `First Citizen:` followed by the start of the play. If the byte count or first three lines differ, re-download — the file shouldn't be changing.

---

## 17.3 Tokenising the corpus

We use `CharTokenizer.from_text(text)` to scan the corpus and build the alphabet, then `tok.encode(text)` to convert to a 1-D tensor of ids.

**Save the following to** 📄 `experiments/38_tokenize_shakespeare.py`:

```python
"""Experiment 38 — Tokenize Tiny Shakespeare."""

from mygpt import CharTokenizer


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)

    print(f"corpus chars:   {len(text):,}")
    print(f"vocab_size:     {tok.vocab_size}")
    print(f"chars:          {tok.chars}")
    print()
    print(f"data shape:     {tuple(data.shape)}")
    print(f"data.dtype:     {data.dtype}")
    print(f"first 60 ids:   {data[:60].tolist()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/38_tokenize_shakespeare.py
```

**Expected output:**

```text
corpus chars:   1,115,394
vocab_size:     65
chars:          ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

data shape:     (1115394,)
data.dtype:     torch.int64
first 60 ids:   [18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 0, 14, 43, 44, 53, 56, 43, 1, 61, 43, 1, 54, 56, 53, 41, 43, 43, 42, 1, 39, 52, 63, 1, 44, 59, 56, 58, 46, 43, 56, 6, 1, 46, 43, 39, 56, 1, 51, 43, 1, 57, 54, 43, 39, 49, 8]
```

Read the alphabet: 65 characters — newline, space, the punctuation Shakespeare actually uses (`! $ & ' , - . : ; ?`), the digit `3` (yes, there is exactly one digit in the entire corpus, in a stage direction), all 26 uppercase letters and all 26 lowercase letters. No "y" without "Y", no Unicode quirks: this is plain ASCII Shakespeare.

The first 60 ids decode to `"First Citizen:\nBefore we proceed any further, hear me speak."` — the speaker label, the line break (id `0` is `'\n'`), and the first line up to and including the period. (You can verify this with `tok.decode(data[:60])` if you like.)

---

## 17.4 Sizing the model

We pick a model that is *just* big enough to do something interesting on CPU:

```python
GPT(vocab_size=65, embed_dim=64, num_heads=4, num_layers=4, max_seq_len=64, dropout=0.0)
```

That is **207,296 parameters** — bigger than the 2,240-parameter toy in Ch.14 by two orders of magnitude, but still tiny compared to GPT-2's 124 M. For reference:

| Source                  | Params       |
|-------------------------|--------------|
| Ch.14 toy GPT (V=4)     | 2,240        |
| **This chapter (V=65)** | **207,296**  |
| GPT-2 small             | 124,412,160  |

The four hyperparameters (`embed_dim`, `num_heads`, `num_layers`, `max_seq_len`) all matter:

- **`embed_dim=64`**: each character becomes a 64-dimensional vector. GPT-2 uses 768; we use 64 to keep training fast.
- **`num_heads=4`**: attention is split into 4 heads of `head_dim = 64/4 = 16`. GPT-2 uses 12 heads.
- **`num_layers=4`**: 4 stacked `TransformerBlock`s. GPT-2 small uses 12.
- **`max_seq_len=64`**: the model can see at most 64 characters of context. GPT-2 uses 1024. Sixty-four characters is roughly one short line of Shakespeare — enough to learn local structure, not enough to learn long-range dependencies.

`dropout=0.0` because our model is small and we won't overtrain badly enough to need regularisation. (Real training runs use 0.1.)

---

## 17.5 The training run

The same four-line training loop from §14.5, with three differences:

1. **`get_batch` samples from the Tiny Shakespeare data tensor** (length 1,115,394) instead of the 256-token toy corpus.
2. **More steps** (2,000 instead of 200) because the task is harder.
3. **We save the trained checkpoint and the tokenizer** at the end so §17.6 can load them and generate.

**Save the following to** 📄 `experiments/39_train_shakespeare.py`:

```python
"""Experiment 39 — Train a 207k-parameter GPT on Tiny Shakespeare.

2,000 AdamW steps with B=16, T=64. Loss drops from ~41 (random init,
confidently wrong) to ~2.08 (a meaningful improvement over log(65) ≈ 4.17,
the random-baseline cross-entropy).
"""

import math
import time

import torch

from mygpt import GPT, CharTokenizer, get_batch, set_seed


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)

    set_seed(0)
    model = GPT(
        vocab_size=tok.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, T = 16, 64
    print(f"params:         {n_params:,}")
    print(f"vocab_size:     {tok.vocab_size}")
    print(f"corpus chars:   {len(data):,}")
    print(f"batch_size B:   {B}")
    print(f"seq_len T:      {T}")
    print(f"reference log(V) = {math.log(tok.vocab_size):.4f}\n")

    set_seed(42)
    t0 = time.time()
    for step in range(1, 2001):
        x, y = get_batch(data, B, T)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step in (1, 10, 100, 500, 1000, 2000):
            print(f"step {step:>4}: loss = {loss.item():.4f}  ({time.time()-t0:.1f}s)")

    torch.save(model.state_dict(), "shakespeare_gpt.pt")
    tok.save("shakespeare_tokenizer.json")
    print("\nSaved shakespeare_gpt.pt and shakespeare_tokenizer.json")


if __name__ == "__main__":
    main()
```

Run it (this is the longest-running command in the tutorial — about a minute on a typical laptop CPU):

```bash
uv run python experiments/39_train_shakespeare.py
```

**Expected output:**

```text
params:         207,296
vocab_size:     65
corpus chars:   1,115,394
batch_size B:   16
seq_len T:      64
reference log(V) = 4.1744

step    1: loss = 41.0367  (0.0s)
step   10: loss = 15.0190  (0.2s)
step  100: loss = 3.2371  (2.0s)
step  500: loss = 2.5944  (9.9s)
step 1000: loss = 2.3529  (19.5s)
step 2000: loss = 2.0785  (45.1s)

Saved shakespeare_gpt.pt and shakespeare_tokenizer.json
```

(Wall-clock seconds will differ on your machine; the loss values will not.)

Two shorthands to keep in mind while reading the curve below: `loss` is the negative log-probability the model assigns the correct next character; `exp(loss)` is the **effective number of characters** the model is still uncertain between. (We will use that second number to read the final loss of `2.08` at the end of this section.)

Read the curve:

- **Step 1: loss ≈ 41.** The randomly-initialised model is *confidently wrong* — its logits happen to point hard at random tokens, and cross-entropy of a confident-wrong prediction is unbounded. (See Ch.13 §13.4 for the full derivation; the principle is identical to Ch.14's V=4 toy where step-1 loss was 5.27 — only the numbers scale with V.)
- **Step 10: loss ≈ 15.** Already collapsing toward `log(V) ≈ 4.17`; AdamW has knocked the wildly-confident logits down toward a flat distribution.
- **Step 100: loss ≈ 3.24.** Below the random-baseline `log(V)`. The model now predicts character-level structure better than uniform guessing.
- **Step 500: loss ≈ 2.59.** It has learned letter co-occurrence ("q" tends to be followed by "u", capital letters start words, line ends are followed by newlines).
- **Step 2000: loss ≈ 2.08.** It has learned the *shape* of Shakespeare — speaker labels, line breaks, the rough rhythm of words. Further training keeps lowering the loss but with diminishing returns per step; we stop here because the next chapter wraps everything in a CLI.

That `~2.08` does not mean the model writes Shakespeare. It means the model's average uncertainty about the next character is `e^2.08 ≈ 8` characters out of 65 — so it has narrowed the field by a factor of 8 over uniform guessing. We will see what 2.08-loss text looks like in the next section.

---

## 17.6 Sampling from the trained model

The trained checkpoint is saved as `shakespeare_gpt.pt` and the tokenizer as `shakespeare_tokenizer.json`. We reload both, prompt with `"ROMEO:"`, and generate 200 new characters with `temperature=1.0` and `top_k=10`.

**Save the following to** 📄 `experiments/40_generate_shakespeare.py`:

```python
"""Experiment 40 — Sample from the trained Shakespeare GPT."""

import torch

from mygpt import GPT, CharTokenizer, generate, set_seed


def main() -> None:
    tok = CharTokenizer.load("shakespeare_tokenizer.json")
    set_seed(0)
    model = GPT(
        vocab_size=tok.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout=0.0,
    )
    model.load_state_dict(torch.load("shakespeare_gpt.pt"))

    set_seed(0)
    prompt = tok.encode("ROMEO:").unsqueeze(0)
    out = generate(model, prompt, max_new_tokens=200, temperature=1.0, top_k=10)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/40_generate_shakespeare.py
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

Read what worked:

- **Speaker label structure.** `ROMEO:` is followed by a newline and dialogue. The model produces a *new* speaker label `BRINCE:` (not a real Shakespeare character, but the right *shape*: capital letters, colon, line break) after the first speech.
- **Word-shaped tokens.** `Thy`, `your`, `Of`, `hath`, `to`, `my`, `for`, `you` are real English words. Most of the rest are pronounceable nonsense.
- **Line breaks and punctuation.** Commas mid-line, periods at the end. The rhythm of two-to-three short clauses per line is recognisably Shakespearean.

Read what didn't:

- Most "words" are gibberish (`momed`, `seltered`, `centeloourse`).
- No grammar to speak of.
- No semantic content — `BRINCE` is not a character, the text is not making any argument.

This is exactly what a 207k-parameter character-level model trained on 1 MB for 45 seconds *should* produce: it has learned the *shape* of the data (speakers, line breaks, capitalisation, the most common words) but not the deeper structure. Bigger models trained for longer on larger corpora — say, GPT-2 small at ~124M parameters — produce qualitatively much more coherent pastiche.

---

## 17.7 Experiments

1. **Cooler temperature.** Edit `experiments/40_generate_shakespeare.py` to call `generate(model, prompt, max_new_tokens=200, temperature=0.5, top_k=10)`. The output stays closer to the most-likely token at every step — fewer gibberish words, more repetition.
2. **Hotter temperature.** Try `temperature=1.5`. The output gets noticeably more random — more gibberish, more odd character sequences (`xq`, `kv`).
3. **Bigger top-k.** Try `top_k=40`. The output sees more of the long tail; some samples are more creative, others are more nonsensical.
4. **Different prompt.** Try `prompt = tok.encode("KING HENRY:").unsqueeze(0)`. The model carries over the speaker-label structure: it produces something speaker-like after the colon.
5. **Train longer.** Edit `experiments/39_train_shakespeare.py` to `range(1, 5001)` instead of `range(1, 2001)`. Watch the loss continue to drop — past 2000 steps the gain per step is small but nonzero. Re-generate and notice the words are slightly less broken.
6. **Train smaller.** Edit `experiments/39_train_shakespeare.py` to use `embed_dim=32, num_heads=2, num_layers=2`. Re-train. The loss plateaus higher (around 2.5) and the generated text is qualitatively worse — more repetition, fewer real words. Capacity matters.

After each experiment, restore any file you changed before moving on.

---

## 17.8 Exercises

1. **What is "loss = 2.08" in plain English?** Cross-entropy loss is the *negative log-likelihood per token*, in nats. The model's average probability assigned to the *correct* next character is therefore $e^{-2.08} \approx 0.125$, or about **12.5%**. Compare that to the uniform baseline $1/V = 1/65 \approx 1.5\%$. The model is roughly **8×** better than random guessing per character. Argue from this that loss reductions of `0.1` matter far more at low loss than at high loss (the relative likelihood gain is exponential in the loss reduction).
2. **Why does step-1 loss exceed `log(V)`?** A perfectly *uniform* random predictor has loss exactly `log(V) = 4.17` for V=65. But our random initialisation has wide logits, so it predicts with high *confidence* on random tokens — and confident wrong predictions are penalised quadratically harder than diffuse wrong ones. Sketch a 3-token softmax with logits `(10, 0, 0)` and one with logits `(0, 0, 0)`, and compute the cross-entropy if the correct class is `2` in each case.
3. **Parameter budget.** From the formulas in earlier chapters, derive that a `GPT(V, C, h, N, L)` model has approximately `V*C + L*C + N*(12*C^2 + 9*C) + 2*C` parameters (ignoring the LM head, which is tied to the token embedding). Plug in V=65, C=64, N=4, L=64; you should get 207,296. (Hint: `4*(12*64^2 + 9*64) = 4*49728 = 198,912` for the blocks.)
4. **Why character-level is hard.** A typical Shakespeare line is ~50 characters. At T=64 we see at most one full line plus a bit. Argue that this means the model effectively cannot learn dependencies that span a stanza or a speech, only ones that span a single line — which is one reason real models use much larger context windows (T=1024 for GPT-2, T=128k+ for newer models).

---

## 17.9 What's next

We can now train and generate on real text. The last piece is *ergonomics*: replacing the bespoke `experiments/39_train_shakespeare.py` and `experiments/40_generate_shakespeare.py` scripts with a proper command-line interface. Chapter 18 builds:

- `mygpt train file.txt` — read any text file, build a tokenizer, train a model, save both as a checkpoint.
- `mygpt generate --checkpoint ckpt.pt --prompt "..."` — load a checkpoint, sample from it.
- Checkpoint files that bundle the tokenizer with the model so a checkpoint is a self-contained unit.

After Chapter 18, the package is **done** — the §1.10 promise ("a Python package called `mygpt` that you can train on a text file and use to generate text from the command line") is fully delivered.

> **Looking ahead — what to remember from this chapter**
>
> 1. Nothing in `mygpt` changed. We composed `CharTokenizer` (Ch.16), `GPT` (Ch.12–13), `get_batch` (Ch.14), the AdamW loop (Ch.14), and `generate` (Ch.15) into one end-to-end pipeline.
> 2. A 207k-parameter character-level GPT trained on 1 MB of Shakespeare for 45 seconds reaches loss ≈ 2.08, about 8× better than random per character. It produces text with speaker labels, line breaks, and recognisable English words — the *shape* of Shakespeare, not the content.
> 3. Loss reductions are *exponential* in their effect on per-token probability — `0.1` loss at step 100 buys far less than `0.1` loss at step 2000.
> 4. A trained checkpoint is meaningless without its tokenizer. Save them together; load them together.

On to [Chapter 18 — Checkpoints, inference, and a CLI](18_cli_and_checkpoints.md).
