---
title: 23. BPETokenizer in mygpt
nav_order: 5
parent: Part II — Advanced Topics
---

# Chapter 23 — `BPETokenizer` in `mygpt`

Chapter 22 taught the BPE algorithm; this chapter wires it into `mygpt` as a `BPETokenizer` class with the same shape as `CharTokenizer` (Ch.16): `from_text`, `encode`, `decode`, `save`, `load`. Once the class is in place, *every part of the model pipeline that used `CharTokenizer` works unchanged with `BPETokenizer`* — same `(B, T)` long-tensor inputs, same checkpoint format. We are swapping the *vocabulary*, not the *model*.

By the end of this chapter you will have:

- added `BPETokenizer` to `mygpt`,
- trained it on Tiny Shakespeare with 512 merges (~70 s on M1) and seen the resulting 577-token vocabulary,
- watched the round-trip `encode → decode` reproduce arbitrary text exactly,
- compared BPE's compression to `CharTokenizer`'s — about **2.3× shorter** sequences for the same text on Tiny Shakespeare,
- saved and reloaded the trained tokenizer to a small JSON file.

The chapter does **not** retrain the model on BPE-encoded data — that is left for §23.9 / Ch.27 / Ch.28. We focus on the tokenizer itself.

---

## 23.1 Setup

This chapter assumes Chapter 22 — `mygpt/` is the Ch.21 ending state (BPE was a pure-pedagogy chapter that didn't modify the package). You also have `tinyshakespeare.txt` from Part I:

```bash
ls tinyshakespeare.txt
```

If missing:

```bash
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 23.2 One simplification from Ch.22's algorithm

Chapter 22 used `</w>` end-of-word markers because the textbook BPE algorithm pre-splits the text on whitespace. For Tiny Shakespeare — a corpus dense with newlines, indentation, and punctuation — that pre-split is awkward and lossy. Production BPE (GPT-2, tiktoken, our class here) drops the marker and operates on the *whole character sequence at once*: pairs cross word boundaries freely, the algorithm just merges whatever's most frequent.

The merge rule for "low low low" (3 occurrences) is now a pair of three characters and a space: `('w', ' ')` — because that's what shows up in the raw text. There is nothing wrong with that; in fact it is *useful*: a single token now represents "low followed by a space", which is a real, frequent thing in this corpus.

Beyond that one change, the algorithm is identical to Ch.22's: count adjacent pair frequencies, merge most-frequent, repeat.

---

## 23.3 The `BPETokenizer` class

We need a top-level `import collections` (the algorithm uses `collections.Counter`). Add it next to the existing `import math`:

**At the top of** 📄 `src/mygpt/tokenizer.py`, **add**:

```python
import collections
```

(Place it next to the existing imports at the top of the file.)

Then the class itself. It follows the same five-method shape as `CharTokenizer` from Ch.16: `__init__`, `from_text`, `encode`, `decode`, `save`, `load`. The serialised state is two JSON-friendly objects: `chars` (the alphabet — sorted unique characters of the training text) and `merges` (an ordered list of `(a, b)` string-pair merge rules).

**Append the following class to** 📄 `src/mygpt/tokenizer.py` (after `CharTokenizer`, before `save_checkpoint`):

```python
class BPETokenizer:
    """Byte-pair-encoding tokenizer.

    Training: scan the text as a sequence of single-character symbols, count
    adjacent-pair frequencies, merge the most-frequent pair into a new symbol,
    repeat for `num_merges` iterations.

    The serialised state is two JSON-friendly objects:
      - `chars`: the initial alphabet (sorted unique characters of the training text)
      - `merges`: the ordered list of (a, b) string-pair merge rules
    The vocabulary is `chars + [a + b for (a, b) in merges]`, with id `i` given
    by position in that combined list.
    """

    def __init__(self, chars: list[str], merges: list[tuple[str, str]]) -> None:
        self.chars = list(chars)
        self.merges = [tuple(m) for m in merges]
        # Build the vocab in id order: alphabet first, then merges in creation order.
        vocab = list(self.chars)
        for a, b in self.merges:
            vocab.append(a + b)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = {s: i for i, s in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str, num_merges: int) -> "BPETokenizer":
        """Train a BPE tokenizer on `text` for `num_merges` merge iterations."""
        chars = sorted(set(text))
        symbols = list(text)
        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            pairs: collections.Counter[tuple[str, str]] = collections.Counter()
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            a, b = best
            merged = a + b
            merges.append((a, b))
            new_symbols: list[str] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return cls(chars, merges)

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` to a 1-D long tensor of ids by applying the merges in order."""
        symbols = list(text)
        for a, b in self.merges:
            new_symbols: list[str] = []
            i = 0
            n = len(symbols)
            merged = a + b
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return torch.tensor([self.stoi[s] for s in symbols], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode a 1-D tensor of ids back to a string."""
        return "".join(self.vocab[int(i)] for i in ids)

    def save(self, path: str) -> None:
        """Persist the tokenizer to a JSON file at `path`."""
        with open(path, "w") as f:
            json.dump({"chars": self.chars, "merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Reload a tokenizer from a JSON file produced by `save`."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"], [tuple(m) for m in data["merges"]])
```

Three things worth pointing out:

- **`from_text` and `encode` share their inner merge loop.** Training picks *which* pair to merge by frequency; encoding picks *which* pair to merge by the rule list. The actual replace-this-pair-with-this-symbol operation is the same.
- **`encode` runs once per merge rule.** For a tokenizer with 512 rules, encoding scans the input 512 times. That's slow on long inputs (we'll see ~20 s on the 1.1 MB Tiny Shakespeare). Real implementations use byte-pair priority queues to do this in one pass; we don't, because the algorithm is already complex enough.
- **`stoi` and `vocab` are mirrors.** `stoi[symbol]` returns the id; `vocab[id]` returns the symbol. Decode is just a list lookup.

---

## 23.4 Training on Tiny Shakespeare

Train the tokenizer with 512 merges. This is the slowest single command in the book — about 70 seconds on M1 — because the inner loop counts pairs across the entire 1.1 MB corpus once per merge. Real implementations are 100× faster; ours is the textbook version with no priority queue.

**Save the following to** 📄 `experiments/42_bpe_train.py`:

```python
"""Experiment 42 — Train a BPE tokenizer on Tiny Shakespeare with 512 merges.

Saves the trained tokenizer to shakespeare.bpe.json (small JSON file, < 5 KB).
~70 s on M1 MPS; reloading is instant.
"""

import time

from mygpt import BPETokenizer


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    t0 = time.time()
    tok = BPETokenizer.from_text(text, num_merges=512)
    elapsed = time.time() - t0

    print(f"training took:    {elapsed:.1f}s")
    print(f"vocab_size:       {tok.vocab_size}")
    print(f"chars (alphabet): {len(tok.chars)}")
    print(f"merges:           {len(tok.merges)}")
    print()
    print(f"first 10 alphabet chars: {tok.chars[:10]}")
    print(f"first 10 merges:")
    for i, (a, b) in enumerate(tok.merges[:10]):
        print(f"  {i + 1}. ({a!r}, {b!r}) -> {(a + b)!r}")

    tok.save("shakespeare.bpe.json")
    print()
    print("saved to shakespeare.bpe.json")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/42_bpe_train.py
```

**Expected output (the wall-clock will vary; the rest is deterministic):**

```text
training took:    69.0s
vocab_size:       577
chars (alphabet): 65
merges:           512

first 10 alphabet chars: ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3']
first 10 merges:
  1. ('e', ' ') -> 'e '
  2. ('t', 'h') -> 'th'
  3. ('t', ' ') -> 't '
  4. ('s', ' ') -> 's '
  5. ('d', ' ') -> 'd '
  6. (',', ' ') -> ', '
  7. ('o', 'u') -> 'ou'
  8. ('e', 'r') -> 'er'
  9. ('i', 'n') -> 'in'
  10. ('y', ' ') -> 'y '

saved to shakespeare.bpe.json
```

Read the merge ordering: `('e', ' ')` is first because 'e' followed by space (i.e. word ending in 'e') is the most-frequent pair in the corpus — Shakespeare uses words ending in 'e' constantly. Then 'th' (the start of `the`, `that`, `this`). Then 't ' (word ending in 't'). Then 's ' (word ending in 's'). The algorithm captures the *grammar of token-final consonants* before it starts capturing actual word-shaped fragments.

The 65-character alphabet is identical to Tiny Shakespeare's character set (we re-derived it from `sorted(set(text))`).

---

## 23.5 Encoding and decoding: the round-trip

Same shape as Chapter 16's `CharTokenizer` round-trip — encode a sample, decode the ids, confirm we get the input back.

**Save the following to** 📄 `experiments/43_bpe_round_trip.py`:

```python
"""Experiment 43 — Encode/decode a sample with the trained BPE tokenizer."""

from mygpt import BPETokenizer


def main() -> None:
    tok = BPETokenizer.load("shakespeare.bpe.json")

    sample = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    ids = tok.encode(sample)
    back = tok.decode(ids)

    print(f"sample chars:  {len(sample)}")
    print(f"encoded ids:   {len(ids)}")
    print(f"compression:   {len(sample) / len(ids):.2f}x")
    print(f"round-trip ok: {back == sample}")
    print()
    print("First 20 ids decoded as their token strings:")
    for i, tok_id in enumerate(ids[:20].tolist()):
        print(f"  {i:>2}: id={tok_id:>3}  symbol={tok.vocab[tok_id]!r}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/43_bpe_round_trip.py
```

**Expected output:**

```text
sample chars:  61
encoded ids:   25
compression:   2.44x
round-trip ok: True

First 20 ids decoded as their token strings:
   0: id=535  symbol='First '
   1: id= 15  symbol='C'
   2: id=125  symbol='it'
   3: id= 47  symbol='i'
   4: id= 64  symbol='z'
   5: id= 79  symbol='en'
   6: id= 76  symbol=':\n'
   7: id= 14  symbol='B'
   8: id= 43  symbol='e'
   9: id=464  symbol='fore '
  10: id=347  symbol='we '
  11: id=415  symbol='pro'
  12: id= 41  symbol='c'
  13: id=135  symbol='ee'
  14: id= 69  symbol='d '
  15: id=528  symbol='any '
  16: id= 44  symbol='f'
  17: id=142  symbol='ur'
  18: id=177  symbol='ther'
  19: id= 70  symbol=', '
```

Read the tokenization of `"First Citizen:\nBefore we proceed any further, hear me speak.\n"` (61 characters → 25 tokens):

- **`'First '`** (id 535) is *one* token, with the trailing space included. Because `First ` is a frequent prefix in Shakespeare's stage directions, BPE has merged the whole 6-character span.
- **`'C'`, `'it'`, `'i'`, `'z'`, `'en'`** — `Citizen` got broken into 5 pieces. BPE has not seen `Citizen` often enough to make it a single token at 512 merges.
- **`':\n'`** (id 76) is one token: a colon followed immediately by a newline. Common in Shakespeare's speaker-label convention.
- **`'fore '`, `'we '`, `'pro'`, …** — small word fragments and full short words get their own tokens (with trailing spaces where appropriate).

The trailing-space tokens are BPE's main compression win on prose: spaces are predictable enough that they get glued onto adjacent characters, and the resulting "word + space" tokens encode the most common short word forms.

---

## 23.6 Compression compared to `CharTokenizer`

How much shorter is BPE's output on the *full* corpus, compared to `CharTokenizer`?

**Save the following to** 📄 `experiments/44_bpe_vs_char_corpus.py`:

```python
"""Experiment 44 — Compare BPE vs Char encoding length on the full corpus.

Skipping over the slow BPE encode (~20 s); the result is deterministic.
"""

import time

from mygpt import BPETokenizer, CharTokenizer


def main() -> None:
    with open("tinyshakespeare.txt") as f:
        text = f.read()

    char_tok = CharTokenizer.from_text(text)
    bpe_tok = BPETokenizer.load("shakespeare.bpe.json")

    t0 = time.time()
    char_ids = char_tok.encode(text)
    t_char = time.time() - t0

    t0 = time.time()
    bpe_ids = bpe_tok.encode(text)
    t_bpe = time.time() - t0

    print(f"corpus chars:           {len(text):,}")
    print(f"CharTokenizer ids:      {len(char_ids):,}  ({t_char:.1f}s to encode)")
    print(f"BPETokenizer  ids:      {len(bpe_ids):,}  ({t_bpe:.1f}s to encode)")
    print(f"BPE compression ratio:  {len(char_ids) / len(bpe_ids):.2f}x")
    print(f"Char vocab_size:        {char_tok.vocab_size}")
    print(f"BPE  vocab_size:        {bpe_tok.vocab_size}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/44_bpe_vs_char_corpus.py
```

**Expected output (timings vary; counts are deterministic):**

```text
corpus chars:           1,115,394
CharTokenizer ids:      1,115,394  (0.1s to encode)
BPETokenizer  ids:      487,961  (19.6s to encode)
BPE compression ratio:  2.29x
Char vocab_size:        65
BPE  vocab_size:        577
```

**The numbers to remember:**

- BPE produces **2.29× fewer tokens** than character-level on the same text. With 1024 merges (twice as many), the ratio rises to ~2.7×; with the full 50,257-token GPT-2 vocabulary, it reaches ~4× on English text.
- **At inference time, fewer tokens means faster generation per character produced.** A 100-character generation that used to take 100 forward passes now takes ~44.
- **Tradeoff at training time**: bigger vocabulary means a bigger token-embedding matrix (`V × C` parameters) and bigger LM head. We will quantify this in Ch.27 when we run a full training pass with `--tokenizer bpe`.

---

## 23.7 Save / load round-trip

Saving and loading is identical in shape to `CharTokenizer.save`/`load` but writes a slightly bigger JSON file (the merge list adds bytes). For 512 merges:

```bash
ls -l shakespeare.bpe.json
wc -c shakespeare.bpe.json
```

The file is ~7 KB on disk — three orders of magnitude smaller than a model checkpoint. Reloading is instant: `BPETokenizer.load(path)` reads the JSON and rebuilds the `vocab` and `stoi` lookup. No training is repeated.

A trained tokenizer is *attached to its corpus*: the merges are tuned to Shakespeare's bigram statistics. Re-training on a different corpus produces a different tokenizer; the JSON file *captures the entire training run* in an artifact you can ship with the model checkpoint. (Ch.18's `save_checkpoint` already bundles a `CharTokenizer`'s `chars` list inside the `.ckpt` dict; in Ch.27/28 we will extend it to bundle a `BPETokenizer`'s `merges` list too.)

---

## 23.8 Experiments

1. **More merges, more compression.** Edit `experiments/42_bpe_train.py` to use `num_merges=1024` instead of 512. Training will take ~120 s on M1; vocab_size becomes 1089; full-corpus compression rises to ~2.7×. The first 512 merges are *identical* to the 512-merge run (BPE is deterministic given the same corpus).

2. **Tokenize a sentence with no trained-corpus vocabulary.** Try `tok.encode("Le ciel est bleu.")`. Most characters fall through (no merges apply because `ciel`, `bleu` are not in the training set), but lowercase letters that *are* in the alphabet still encode to single-character tokens. The space-character merges still help: `'l ', 'e '` (if either of those exist as merges) get applied. Output length will be close to the character length.

3. **A token's "characters" string.** Pick any merged-token id (say, 535 — the `'First '` token from §23.5). Print `tok.vocab[535]`. This is the literal string of characters that decoded into that single token: `'First '`. Every token's text is just `vocab[id]`; decode is `"".join(vocab[i] for i in ids)`.

4. **Check decode is a left-inverse of encode.** Run `tok.decode(tok.encode(text)) == text` for several test strings (the corpus, the first paragraph, a random typed-in sentence). Always True — BPE never loses information because every character in the input is a member of the alphabet (and no merge ever removes a character; merges only group existing characters).

After each experiment, restore any file you changed before moving on.

---

## 23.9 Exercises

1. **Why is `from_text` slow but `encode` fast on small inputs?** Both use the same merge loop. Argue that `from_text`'s cost scales as `O(num_merges × corpus_chars)`, while `encode`'s cost scales as `O(num_merges × input_chars)`. For a 60-character sample with 512 merges, `encode` is `60 × 512 ≈ 31 k` ops — milliseconds. For the 1.1 M-character corpus, it is `1.1 M × 512 ≈ 560 M` ops — 19 seconds.

2. **Could `encode` be faster?** Read the [minBPE](https://github.com/karpathy/minbpe) implementation. It uses a per-position priority queue keyed by merge rank, so each input character touches each merge at most once instead of once per merge. The asymptotic cost drops to `O(input_chars × log(num_merges))`. Argue why this would be a worthwhile optimisation for Ch.28's full Wikipedia training but not for our toy demo.

3. **Tokenizer extensibility.** The `chars` list is fixed at training time. What happens if you call `tok.encode(text)` on text containing a character not in `chars`? Walk through the code to predict the failure mode. (Hint: `self.stoi[s]` will raise `KeyError`.)

4. **Information-theoretic compression.** Argue from Shannon's source coding theorem that *any* tokenizer's compression ratio on a corpus is bounded by the corpus's entropy. Tiny Shakespeare's character-level entropy is roughly 4.5 bits per character (the alphabet has 65 symbols ≈ 6 bits, but English regularities reduce it). Argue why a 1024-merge BPE tokenizer's ~2.7× compression is *not* contradicting this bound. (Hint: BPE doesn't compress the *bits*, it compresses the *number of tokens*. Each BPE token still carries ~9 bits of identity in a 577-token vocab.)

---

## 23.10 What's next

The next chapter, **Chapter 24 — RMSNorm replaces LayerNorm**, leaves tokenization behind and starts modernising the *architecture*. RMSNorm is what Llama and Mistral use; Chapter 11's `LayerNorm` will get a sibling class, and a `--norm {rms, layer}` flag will let the student pick at training time.

> **Looking ahead — what to remember from this chapter**
>
> 1. `BPETokenizer` is shape-compatible with `CharTokenizer`: `from_text`, `encode`, `decode`, `save`, `load`. The model code doesn't care which one feeds it ids.
> 2. Training BPE is slow (~70 s on Tiny Shakespeare with 512 merges) but you only do it *once per corpus* and ship the JSON file alongside the checkpoint.
> 3. The vocabulary is `chars + merges-in-creation-order`; decoding is `"".join(vocab[i] for i in ids)`.
> 4. Compression on Tiny Shakespeare: **~2.3× shorter sequences** at 512 merges, ~2.7× at 1024 merges, ~4× at GPT-2 scale (50,257 tokens). Fewer tokens ≈ proportionally faster generation.

On to [Chapter 24 — RMSNorm replaces LayerNorm](24_rmsnorm.md).
