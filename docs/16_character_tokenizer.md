---
title: 16. A reusable character tokenizer
nav_order: 17
parent: LLM Fundamentals
---

# Chapter 16 — A reusable character tokenizer

So far the entire tutorial has run on the four-token vocabulary

```python
VOCAB = ("I", "love", "AI", "!")
```

That worked because we hand-picked it. Real text is not so kind: a paragraph of English contains thousands of distinct words, punctuation patterns, capitalisations and whitespace conventions. We need a *tokenizer* — a small, separate component whose only job is to map text → ids and ids → text. It is independent of the model: training the model on *different* text just means swapping out the tokenizer.

This chapter builds the simplest possible tokenizer, **character-level**: every distinct character becomes its own token. By the end you will have:

- understood why the tokenizer is a separable component (and why GPT-2 actually uses something fancier, called BPE),
- added a `CharTokenizer` class to `mygpt` with `encode`, `decode`, `save`, and `load` methods,
- watched a round-trip encode → decode reproduce the input text exactly,
- saved a tokenizer to disk and reloaded it, ready for Chapter 17 to train a real model on real text.

---

## 16.1 What a tokenizer is, and why it is separate from the model

Recall what the GPT model expects as input: a `(B, T)` long tensor of integer ids in $[0, V)$, where $V$ is the vocabulary size. The model neither knows nor cares whether token id 5 stands for the word `"the"`, the character `'t'`, or a sub-word fragment like `"##ing"`. That mapping — text ↔ ids — lives entirely in the **tokenizer**.

This separation has two consequences:

1. **Different tokenizers, same model architecture.** A 124M-parameter GPT-2 with `vocab_size=50257` is the *same* architecture as our `vocab_size=4` toy. Only the embedding table, the LM head, and (because they are tied) the readout dimension differ.
2. **The tokenizer must be saved alongside the model.** A trained model whose token id 5 means `'t'` is gibberish if loaded with a tokenizer where id 5 means `'q'`. Persistence matters.

Real GPT-2 uses a **byte-pair encoding (BPE)** tokenizer with 50,257 sub-word tokens — a compromise that gives common words like `"the"` their own id while breaking rare words into pieces. We won't implement BPE; it is its own subject. Character-level is the simplest tokenizer that works on arbitrary text, and it is the natural starting point for a tutorial.

A *character-level* tokenizer:

- vocabulary = the sorted list of distinct characters in the training text,
- `encode(text)` = look up every character's id,
- `decode(ids)` = look up every id's character and concatenate.

That's the whole idea.

---

## 16.2 Setup

If you finished Chapter 15, you already have everything: `mygpt/` with `get_batch`, `GPT`, `generate`, and the `trained_gpt.pt` checkpoint.

If you skipped, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Overwrite **`src/mygpt/__init__.py`** with the Chapter 15 ending state from `docs/_state_after_ch15.md`. (`trained_gpt.pt` from Ch.14 is *not* required for this chapter; we don't load any checkpoint here.)

You are ready.

---

## 16.3 Building the alphabet from text

The very first step in character-level tokenization is to scan a text and collect every distinct character. Python makes this a one-liner:

```python
text = "I love AI !"
chars = sorted(set(text))
```

`set(text)` deduplicates; `sorted(...)` puts the result in a deterministic order so the same text always produces the same vocabulary on different machines.

For our familiar string the alphabet is:

```python
[' ', '!', 'A', 'I', 'e', 'l', 'o', 'v']   # 8 distinct characters
```

Eight, not four — the four-token `VOCAB` of earlier chapters treated `"love"` as one symbol; the character tokenizer treats it as four (`'l'`, `'o'`, `'v'`, `'e'`).

The id-of-character lookup is just a dict over `enumerate`:

```python
stoi = {c: i for i, c in enumerate(chars)}   # "string to int"
itos = {i: c for i, c in enumerate(chars)}   # "int to string"
```

`stoi[' ']` is `0`; `stoi['I']` is `3`; `itos[3]` is `'I'`. This is exactly the same structure as the earlier `VOCAB`/`VOCAB.index(...)` pair — we have just generalised it to "whatever characters appear in the training text".

A small experiment to see this for our running example.

**Save the following to** 📄 `experiments/34_alphabet.py`:

```python
"""Experiment 34 — Build a character alphabet from a string."""


def main() -> None:
    text = "I love AI !"
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    print(f"text:        {text!r}")
    print(f"vocab_size:  {len(chars)}")
    print(f"chars:       {chars}")
    print(f"stoi:        {stoi}")
    print(f"itos:        {itos}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/34_alphabet.py
```

**Expected output:**

```text
text:        'I love AI !'
vocab_size:  8
chars:       [' ', '!', 'A', 'I', 'e', 'l', 'o', 'v']
stoi:        {' ': 0, '!': 1, 'A': 2, 'I': 3, 'e': 4, 'l': 5, 'o': 6, 'v': 7}
itos:        {0: ' ', 1: '!', 2: 'A', 3: 'I', 4: 'e', 5: 'l', 6: 'o', 7: 'v'}
```

---

## 16.4 The `CharTokenizer` class

We package this into a class that exposes a clean four-method API: `encode`, `decode`, `save`, `load`. Plus a class-method constructor `from_text(text)` that builds the alphabet for you.

**Append the following to** 📄 `src/mygpt/__init__.py` (after the `generate` function, before `main`):

```python
import json


class CharTokenizer:
    """Character-level tokenizer.

    Vocabulary = the sorted list of distinct characters seen in the training
    text. Token id = position in that list.
    """

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer whose vocabulary is the alphabet of `text`."""
        return cls(sorted(set(text)))

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` to a 1-D long tensor of ids."""
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode a 1-D tensor of ids back to a string."""
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        """Persist the tokenizer to a JSON file at `path`."""
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Reload a tokenizer from a JSON file produced by `save`."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"])
```

A few small choices worth flagging:

- **`json`, not `torch.save`.** A tokenizer is a list of characters and nothing else; using JSON keeps the file small, human-readable, and editable in a text editor. The model checkpoint will continue to use `torch.save`/`torch.load`.
- **`encode` returns a `torch.Tensor`, not a list.** Every downstream consumer (`GPT.forward`, `generate`, `get_batch`) expects tensors. Returning the right type avoids a `torch.tensor(...)` cast at every call site.
- **`decode` accepts a 1-D tensor.** `int(i)` works for both Python ints and zero-dim tensors, so iterating over a 1-D tensor yields scalars that decode cleanly.
- **The vocabulary is fixed at construction.** `encode` will raise a `KeyError` if you pass a character that wasn't in the training text. We will *not* gracefully handle out-of-vocabulary characters in this tutorial; the next chapter just takes the union over the full training file.

---

## 16.5 Encoding and decoding: the round-trip

Now the satisfying experiment: encode a string, then decode the ids back, and check that we recover the original text.

**Save the following to** 📄 `experiments/35_encode_decode.py`:

```python
"""Experiment 35 — Build a CharTokenizer and round-trip encode/decode."""

from mygpt import CharTokenizer


def main() -> None:
    text = "I love AI ! I love AI !"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    back = tok.decode(ids)

    print(f"text:       {text!r}")
    print(f"vocab_size: {tok.vocab_size}")
    print(f"ids shape:  {tuple(ids.shape)}")
    print(f"first 12 ids: {ids[:12].tolist()}")
    print(f"decoded:    {back!r}")
    print(f"round-trip ok: {back == text}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/35_encode_decode.py
```

**Expected output:**

```text
text:       'I love AI ! I love AI !'
vocab_size: 8
ids shape:  (23,)
first 12 ids: [3, 0, 5, 6, 7, 4, 0, 2, 3, 0, 1, 0]
decoded:    'I love AI ! I love AI !'
round-trip ok: True
```

Read the 12 ids: `3, 0, 5, 6, 7, 4, 0, 2, 3, 0, 1, 0`. Decoded character-by-character against `itos`:

| id | char    |
|----|---------|
| 3  | `'I'`   |
| 0  | `' '`   |
| 5  | `'l'`   |
| 6  | `'o'`   |
| 7  | `'v'`   |
| 4  | `'e'`   |
| 0  | `' '`   |
| 2  | `'A'`   |
| 3  | `'I'`   |
| 0  | `' '`   |
| 1  | `'!'`   |
| 0  | `' '`   |

That spells `"I love AI ! "` — and the next 11 ids (`3, 0, 5, 6, 7, 4, 0, 2, 3, 0, 1`) repeat for `"I love AI !"`. The round-trip is exact.

---

## 16.6 Saving and loading

A tokenizer is meaningless unless you can re-load the same vocabulary that the model was trained against. `CharTokenizer.save(path)` writes a small JSON file; `CharTokenizer.load(path)` re-creates the tokenizer from it.

**Save the following to** 📄 `experiments/36_save_load.py`:

```python
"""Experiment 36 — Save a tokenizer to disk and reload it.

The reloaded tokenizer encodes to the same ids as the original.
"""

from mygpt import CharTokenizer


def main() -> None:
    text = "I love AI !"
    tok1 = CharTokenizer.from_text(text)
    tok1.save("tokenizer.json")
    print(f"saved tokenizer.json (vocab_size={tok1.vocab_size})")

    tok2 = CharTokenizer.load("tokenizer.json")
    print(f"loaded tokenizer.json (vocab_size={tok2.vocab_size})")

    sample = "love AI"
    ids1 = tok1.encode(sample)
    ids2 = tok2.encode(sample)
    print(f"sample:        {sample!r}")
    print(f"original ids:  {ids1.tolist()}")
    print(f"reloaded ids:  {ids2.tolist()}")
    print(f"identical:     {ids1.tolist() == ids2.tolist()}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/36_save_load.py
```

**Expected output:**

```text
saved tokenizer.json (vocab_size=8)
loaded tokenizer.json (vocab_size=8)
sample:        'love AI'
original ids:  [5, 6, 7, 4, 0, 2, 3]
reloaded ids:  [5, 6, 7, 4, 0, 2, 3]
identical:     True
```

If you `cat tokenizer.json`, you will see exactly:

```json
{"chars": [" ", "!", "A", "I", "e", "l", "o", "v"]}
```

A tokenizer file for our running example is 51 bytes (the JSON above, no trailing newline). A real-text tokenizer for the Tiny Shakespeare corpus we will use in Chapter 17 has ~65 distinct characters and the JSON file is around 350 bytes. The size is negligible compared to the model, but the *correctness* of pairing the right tokenizer with the right model is essential.

---

## 16.7 Tokenizing the running example for the model

For the rest of the tutorial, the *full* training pipeline takes:

1. **A training text file.** (e.g. `tiny_shakespeare.txt`.)
2. **A tokenizer built from it.** (`CharTokenizer.from_text(text)`.)
3. **A 1-D long tensor of every character's id.** (`tok.encode(text)`.)
4. **A `GPT(vocab_size=tok.vocab_size, ...)` model** sized to that vocabulary.
5. **The `get_batch` sampler from Ch.14**, the **AdamW training loop from Ch.14**, the **`generate` function from Ch.15**.

For sanity, let's wire this up with the existing four-token running example: scan the string, build a tokenizer, encode it into the corpus tensor, and confirm the result is shaped exactly like the corpus we had in Chapter 14 — except the vocabulary is 8 (characters) instead of 4 (whole words).

**Save the following to** 📄 `experiments/37_corpus_with_tokenizer.py`:

```python
"""Experiment 37 — Build the training corpus tensor via CharTokenizer.

This is the same shape of input the Ch.14 training loop expects — just
with vocab_size=8 (characters) instead of 4 (whole words).
"""

import torch

from mygpt import CharTokenizer


def main() -> None:
    text = "I love AI ! " * 16   # 192 characters total
    tok = CharTokenizer.from_text(text)
    data = tok.encode(text)

    print(f"text length (chars): {len(text)}")
    print(f"vocab_size:          {tok.vocab_size}")
    print(f"data shape:          {tuple(data.shape)}")
    print(f"data.dtype:          {data.dtype}")
    print(f"first 24 ids:        {data[:24].tolist()}")
    print(f"decoded[:24]:        {tok.decode(data[:24])!r}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/37_corpus_with_tokenizer.py
```

**Expected output:**

```text
text length (chars): 192
vocab_size:          8
data shape:          (192,)
data.dtype:          torch.int64
first 24 ids:        [3, 0, 5, 6, 7, 4, 0, 2, 3, 0, 1, 0, 3, 0, 5, 6, 7, 4, 0, 2, 3, 0, 1, 0]
decoded[:24]:        'I love AI ! I love AI ! '
```

Notice the structure: every 12 characters we get the cycle `"I love AI ! "` (with the trailing space). The repeating-cycle property is preserved — Chapter 17's training loop will rediscover it on real Shakespeare with a much larger alphabet.

We have not retrained the model. The Ch.14 checkpoint was trained on a *4*-token vocabulary, so it cannot consume our character-level ids. Chapter 17 will train a fresh GPT against the character corpus produced here.

---

## 16.8 Experiments

1. **Try it on a longer sentence.** Edit `experiments/35_encode_decode.py`'s `text` to `"The quick brown fox jumps over the lazy dog."`. The vocab grows to 29 distinct characters (the 26 lowercase letters, plus space, period, and the uppercase `'T'`); the round-trip still works.
2. **Out-of-vocabulary fails loudly.** Try `tok.encode("Hello world")` after `tok = CharTokenizer.from_text("I love AI !")`. The character `'H'` was never in the training text, so `encode` raises `KeyError: 'H'`. This is intentional — we *want* it to fail loudly rather than silently produce a wrong-shaped token sequence.
3. **Sorting makes the vocabulary deterministic.** Build two tokenizers, one from `"abcd"` and one from `"dcba"`. Both produce the same `chars = ['a', 'b', 'c', 'd']` and the same `tok.encode("a") == tensor([0])`, because `sorted(set(text))` doesn't care what order the input characters appeared in. This is the whole reason `from_text` calls `sorted` — same training text → same vocabulary, regardless of OS, Python version, or hash randomisation.
4. **Save and reload across runs.** Run experiment 36 once, then in a fresh `uv run python` invocation reload `tokenizer.json` and confirm `tok2.encode("love")` is identical to what you got in the first run. The persistence is the whole reason the tokenizer needs to be a separate, saveable artifact.

After each experiment, restore any file you changed before moving on.

---

## 16.9 Exercises

1. **Vocabulary growth on real text.** A 1 MB plain-text file like the Tiny Shakespeare corpus contains roughly 65 distinct characters (uppercase + lowercase alphabet + digits + punctuation + whitespace). Argue that `vocab_size` for a character-level tokenizer is bounded by the size of the alphabet of the training text, not by its length. (Hint: `set(text)` cannot be larger than the number of distinct *characters*, regardless of how long `text` is.)
2. **Why sub-word tokenizers exist.** Argue that a *character-level* tokenizer requires the model to learn long-range dependencies (e.g. "what comes after `t-h-e- -c-a-`?") that a word-level or sub-word tokenizer would express in a single id. (Hint: every English word becomes between ~3 and ~12 tokens at character level; ~1 to ~3 at sub-word level.)
3. **Decoding in batches.** Our `decode` takes a 1-D tensor. The model produces `(B, T)` tensors. Sketch how you would write `decode_batch(ids: torch.Tensor) -> list[str]` for a `(B, T)` tensor; check what `for row in ids:` iterates over.
4. **Tokenizer round-trip identity.** Argue that for a character-level tokenizer trained on `text`, the round-trip `tok.decode(tok.encode(text))` always returns `text` exactly. Why is this *not* true for BPE tokenizers in general? (Hint: BPE tokenizes at the byte level and merges based on frequency; it can re-tokenize an already-tokenized string differently.)

---

## 16.10 What's next

We have a tokenizer that can chew on arbitrary text. Chapter 17 puts it to work:

- download the Tiny Shakespeare corpus (a ~1 MB plain-text file),
- build a `CharTokenizer` from it,
- train a `GPT(vocab_size=tokenizer.vocab_size, embed_dim=128, num_heads=4, num_layers=4, max_seq_len=128, dropout=0.1)` on it,
- watch the loss drop and the model start producing pseudo-Shakespearean text.

Chapter 18 wraps everything in a CLI and adds checkpointing, so you can `mygpt train file.txt` and `mygpt generate --prompt "..."` from any text file.

> **Looking ahead — what to remember from this chapter**
>
> 1. The tokenizer is a separate component from the model. Different text → different tokenizer → different vocabulary, but the same model architecture.
> 2. Character-level is the simplest tokenizer: alphabet = `sorted(set(text))`, encode = lookup, decode = lookup.
> 3. The tokenizer must be saved alongside the model. A model whose token id 5 means `'t'` is gibberish if loaded with a tokenizer where id 5 means `'q'`.
> 4. Real GPT-2 uses BPE (byte-pair encoding), a sub-word tokenizer with 50,257 ids — the same architecture, just a much bigger vocabulary.

On to [Chapter 17 — Training on a real text file](17_training_on_real_text.md) *(coming soon)*.
