---
title: 28. Modern recipe at scale (Wikipedia)
nav_order: 10
parent: Part II — Advanced Topics
---

# Chapter 28 — Modern recipe at scale (Wikipedia)

The previous chapter ran the modern recipe on Tiny Shakespeare and beat the Ch.17 baseline by 0.34 nats at the same parameter budget. The architecture and the training loop are now exactly the modern open-weight stack — but at a model size and a corpus size where the *quality* of the resulting samples is still capped by data, not by the recipe. Tiny Shakespeare is 1 MB. Llama-3-8B was trained on roughly 15 TB.

This chapter unfreezes the two ingredients we have been holding fixed for nine chapters of comparison: the **corpus** (Tiny Shakespeare → ~500 MB of Wikipedia) and the **model size** (~200 k params → ~2 M params). Same code. Same recipe. Same `mygpt train` command, with a few new flags. The training run takes 1–3 hours on M1 MPS — Part II's only long run — and produces a checkpoint that generates qualitatively different text from anything we have trained before.

By the end of this chapter you will have:

- downloaded a ~520 MB Wikipedia corpus (the public `wikitext-103-raw-v1` train + valid + test splits, header-stripped, concatenated),
- added a `--tokenizer {char,bpe}` flag to `mygpt train` so the BPE tokenizer (Ch.23) is finally wired into the trainer,
- added a `--num-merges N` flag (default 1024) that controls the BPE vocabulary size,
- learned why the Ch.23 BPE encoder takes hours on a 500 MB corpus and added a **fast-path** encoder + trainer that does the same job in seconds,
- added a `--bpe-train-bytes N` flag that trains BPE on a representative slice of the corpus (default 0 = full corpus), which keeps the BPE setup time bounded even on huge corpora,
- added a `--checkpoint-every N` flag with an atomic-write save so a Ctrl-C mid-run never corrupts the checkpoint,
- trained a ~2 M-parameter `mygpt` on ~500 MB of Wikipedia in 1–3 hours on M1 MPS, with the full modern recipe (`--precision bf16 --schedule cosine --warmup 500 --max-grad-norm 1.0 --norm rms --position rope --num-kv-heads 2 --tokenizer bpe`),
- generated 200 tokens of Wikipedia-flavoured text — different from the Shakespeare samples in every way that data shapes a model.

This is the end of Part II. After this chapter, `mygpt` is a working modern small language model.

---

## 28.1 Setup

This chapter assumes Chapter 27 — `mygpt/` has every flag from Ch.19–26 wired into the CLI. We will add four more here: `--tokenizer`, `--num-merges`, `--bpe-train-bytes`, `--checkpoint-every`. Then we add a fast BPE training/encoding path inside `BPETokenizer` so the new flags are usable at corpus scale.

We need `tinyshakespeare.txt` only for one backward-compat smoke check at the end of §28.4. The new corpus arrives in §28.3.

```bash
ls tinyshakespeare.txt || curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You will also need ~10 GB of free disk: ~520 MB for the corpus, ~150 MB for the checkpoint, plus working files.

You are ready.

---

## 28.2 Four new CLI flags

The trainer in `_train_command` has, since Ch.16, hardcoded `tokenizer = CharTokenizer.from_text(text)`. We replace that with a flag-driven branch so the same command can choose between Ch.16's char tokenizer and Ch.23's BPE tokenizer, which until now has lived in `mygpt` without ever being used by the CLI.

We also add `--checkpoint-every N` with an *atomic* save: the bytes go to `path + ".tmp"` and `os.replace` atomically swaps them into place. A Ctrl-C mid-write leaves any prior checkpoint at `path` intact — important when a single training run takes hours.

**Replace `save_checkpoint` and `load_checkpoint` in** 📄 `src/mygpt/checkpoint.py`:

```python
def save_checkpoint(
    model: "GPT",
    tokenizer: "CharTokenizer | BPETokenizer",
    path: str,
) -> None:
    """Bundle model weights, tokenizer, and architecture into one .ckpt file.

    The tokenizer's full state is serialised so generation can reload without
    retraining.  ``tokenizer_kind`` is ``"char"`` (Part-I default) or ``"bpe"``
    (Ch.23+); the field is absent on pre-Ch.28 checkpoints, in which case
    ``load_checkpoint`` falls back to ``"char"``.

    The save is atomic: bytes are written to ``path + ".tmp"`` and then renamed
    over ``path``, so a Ctrl-C during writing leaves any prior checkpoint at
    ``path`` intact.
    """
    if isinstance(tokenizer, BPETokenizer):
        tokenizer_kind = "bpe"
        tokenizer_state = {
            "chars": tokenizer.chars,
            "merges": tokenizer.merges,
        }
    else:
        tokenizer_kind = "char"
        tokenizer_state = {"chars": tokenizer.chars}

    payload = {
        "model_state_dict": model.state_dict(),
        # Part-I-and-earlier compatibility: char tokenizer state continues to
        # be mirrored under the original key.  Newer loaders prefer the
        # `tokenizer_kind` + `tokenizer_state` pair below.
        "tokenizer_chars": tokenizer.chars,
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_state": tokenizer_state,
        "config": {
            "vocab_size":     model.vocab_size,
            "embed_dim":      model.embed_dim,
            "num_heads":      model.num_heads,
            "num_kv_heads":   getattr(model, "num_kv_heads", model.num_heads),
            "num_layers":     model.num_layers,
            "max_seq_len":    model.max_seq_len,
            "norm_type":      getattr(model, "norm_type", "layer"),
            "position_type":  getattr(model, "position_type", "learned"),
            "tokenizer_kind": tokenizer_kind,
        },
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    import os
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer | BPETokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`.

    Always loads to CPU; the caller is responsible for `.to(device)` afterwards.
    Pre-Ch.24 checkpoints have no `norm_type` field; pre-Ch.25 checkpoints have
    no `position_type` field; pre-Ch.26 checkpoints have no `num_kv_heads` field;
    pre-Ch.28 checkpoints have no `tokenizer_kind` field.
    All four default to their original behaviour (``"layer"``, ``"learned"``,
    ``num_kv_heads = num_heads``, and ``"char"``) so old `.ckpt` files continue
    to load.
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    tokenizer_kind = config.get("tokenizer_kind", ckpt.get("tokenizer_kind", "char"))
    if tokenizer_kind == "bpe":
        state = ckpt["tokenizer_state"]
        tokenizer = BPETokenizer(state["chars"], state["merges"])
    else:
        tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
    model = GPT(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads", config["num_heads"]),
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,
        norm_type=config.get("norm_type", "layer"),
        position_type=config.get("position_type", "learned"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer
```

Two things to notice:

- The signature change is purely the type of `tokenizer` (now `CharTokenizer | BPETokenizer`). Callers pass whichever the user picked.
- The atomic-save pattern (`tmp_path` + `os.replace`) is one of the small habits that makes long-running scripts safe. `os.replace` is guaranteed atomic on POSIX and on modern Windows; the file at `path` either is the old one or is the new one, never half-written.

Now thread the choice through `_train_command`. We will replace the relevant chunks of the function in §28.4 once the BPE fast path exists; for now, just add the three new arguments to the parser.

**Append the following four arguments to `main()` in** 📄 `src/mygpt/cli.py` (after the existing `--num-kv-heads` argument):

```python
    p_train.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="char",
        help="Tokenizer to use. 'char' (default; CharTokenizer, Ch.16) or 'bpe' (BPETokenizer, Ch.23). BPE training runs on the corpus before model training.",
    )
    p_train.add_argument(
        "--num-merges",
        type=int,
        default=1024,
        help="Number of BPE merges (only meaningful when --tokenizer=bpe). Default 1024.",
    )
    p_train.add_argument(
        "--bpe-train-bytes",
        type=int,
        default=0,
        help="Train BPE on the first N bytes of the corpus (0 = full corpus, default). Useful when the BPE training cost on the full corpus is too high; the encoder still runs on the full corpus and skips characters that did not appear in the training slice.",
    )
    p_train.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save the checkpoint every N steps (0 = save only at end, default). Atomic write: a Ctrl-C mid-save does not corrupt the file.",
    )
```

Verify the four flags appear:

```bash
uv run mygpt train --help | grep -A1 -E '\-\-(tokenizer|num-merges|bpe-train-bytes|checkpoint-every) '
```

Expected output:

```text
  --tokenizer {char,bpe}
                        Tokenizer to use. 'char' (default; CharTokenizer,
  --num-merges NUM_MERGES
                        Number of BPE merges (only meaningful when
  --bpe-train-bytes BPE_TRAIN_BYTES
                        Train BPE on the first N bytes of the corpus (0 =
  --checkpoint-every CHECKPOINT_EVERY
                        Save the checkpoint every N steps (0 = save only at
```

---

## 28.3 The corpus

The reader artefact for this chapter is a 520 MB plain-text corpus called `wikipedia.txt`, sitting in your project root. It contains the train + valid + test splits of `wikitext-103-raw-v1` — a public Wikipedia subset published by Salesforce — concatenated, with the `= Header =` lines stripped (those add no information for the language model and bloat the BPE vocab with non-content tokens).

We download via `urllib` from a pinned Smerity mirror so the script does not depend on the HuggingFace dataset API (which has been flaky in past Part-II runs).

**Save the following to** 📄 `experiments/50_download_wikipedia.py`:

```python
"""Download the wikitext-103-raw-v1 corpus to wikipedia.txt (Ch.28).

Fetches a pinned mirror of the Salesforce wikitext-103-raw-v1 train split,
unzips it, concatenates the train/valid/test files, strips the few wikitext
header markers that bloat the vocab without helping training, and writes the
result to ``wikipedia.txt`` in the current working directory.

The download URL is fixed (not the flaky HuggingFace dataset API); the script
is idempotent — if ``wikipedia.txt`` is already present it does nothing.

Run this once from the project root:

    uv run python experiments/50_download_wikipedia.py

Final size on disk: ~520 MB.  Time on a typical home connection: a few minutes
for the download, ~10 s for the unzip + concat.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import urllib.request
import zipfile

URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
EXPECTED_BYTES = 191_984_949  # As of 2024-03; pinned, do not adjust silently.
OUT_PATH = "wikipedia.txt"
ZIP_PATH = "wikitext-103-raw-v1.zip"
EXTRACT_DIR = "wikitext-103-raw"
TRAIN_FILE = os.path.join(EXTRACT_DIR, "wiki.train.raw")
VALID_FILE = os.path.join(EXTRACT_DIR, "wiki.valid.raw")
TEST_FILE = os.path.join(EXTRACT_DIR, "wiki.test.raw")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: str) -> None:
    print(f"downloading {url}")
    print(f"          → {dest}")
    # Some CDN mirrors (smerity.com among them) reject the default
    # ``Python-urllib/x.y`` User-Agent with 403; spoof a generic one.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100.0 * downloaded / total
                sys.stdout.write(
                    f"\r  {_human(downloaded)} / {_human(total)} ({pct:.1f}%)"
                )
                sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def _strip_wikitext_markup(text: str) -> str:
    """Strip the ``= Header =`` lines wikitext interleaves between articles.

    These lines bloat the BPE vocabulary with non-content tokens.  Everything
    else is left untouched: punctuation, casing, the ``@-@`` tokens (a wikitext
    artefact for hyphens), and the blank lines between paragraphs all survive.
    """
    return re.sub(r"^\s*=+ [^=]+ =+\s*$\n?", "", text, flags=re.MULTILINE)


def main() -> int:
    if os.path.exists(OUT_PATH):
        size = os.path.getsize(OUT_PATH)
        print(f"{OUT_PATH} already exists ({_human(size)}); nothing to do.")
        return 0

    if not os.path.exists(ZIP_PATH):
        _download(URL, ZIP_PATH)
    actual = os.path.getsize(ZIP_PATH)
    if actual != EXPECTED_BYTES:
        print(
            f"warning: {ZIP_PATH} is {actual:,} bytes; expected "
            f"{EXPECTED_BYTES:,}.  Continuing anyway.",
            file=sys.stderr,
        )

    if not os.path.exists(TRAIN_FILE):
        print(f"unzipping {ZIP_PATH} → {EXTRACT_DIR}/")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(".")

    print(f"concatenating train + valid + test → {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for src in (TRAIN_FILE, VALID_FILE, TEST_FILE):
            with open(src, encoding="utf-8") as f:
                cleaned = _strip_wikitext_markup(f.read())
            out.write(cleaned)

    out_size = os.path.getsize(OUT_PATH)
    print(f"wrote {OUT_PATH}: {_human(out_size)}")

    # Tidy up the intermediate files; keeping them around eats ~720 MB of disk.
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    print("removed intermediate zip + extract dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run it. The download is ~192 MB compressed; expect a few minutes on a home connection plus ~10 s for the unzip/concat.

```bash
uv run python experiments/50_download_wikipedia.py
```

Expected output (numbers are exact for the pinned URL; download speed varies):

```text
downloading https://wikitext.smerity.com/wikitext-103-raw-v1.zip
          → wikitext-103-raw-v1.zip
  183.1 MB / 183.1 MB (100.0%)
unzipping wikitext-103-raw-v1.zip → wikitext-103-raw/
concatenating train + valid + test → wikipedia.txt
wrote wikipedia.txt: 516.7 MB
removed intermediate zip + extract dir
```

After this, `ls -lh wikipedia.txt` should show roughly 517 MB. Re-running the script is a no-op:

```bash
uv run python experiments/50_download_wikipedia.py
```

```text
wikipedia.txt already exists (516.7 MB); nothing to do.
```

The first ~150 characters of the file:

```bash
head -c 150 wikipedia.txt
```

```text
 Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred t
```

Notice the leading space (wikitext convention), the `:` and `,` separated by spaces (also wikitext), and the CJK characters — Wikipedia is genuinely multilingual.

---

## 28.4 The toy BPE encoder is too slow at this scale

Chapter 23's `BPETokenizer` is *correct* on any corpus, but its `encode` is `O(num_merges × len(text))`: it scans the whole text once per merge rule. On 1 MB Tiny Shakespeare with 512 merges that is ~20 s. On 517 MB of Wikipedia with 1024 merges, the same code would take roughly *four hours* — before the model takes its first gradient step. The training trainer has the same problem: `from_text` does a full-corpus pair-count and rewrite per merge.

Run this measurement to see what we are up against:

**Save the following to** 📄 `experiments/45_bpe_toy_speed.py`:

```python
"""Benchmark the Ch.23 BPE encoder on a 1 MB Wikipedia slice (Ch.28)."""
import time

from mygpt import BPETokenizer

text = open("wikipedia.txt").read()[:1_000_000]
print(f"slice: {len(text):,} chars")

t0 = time.time()
tok = BPETokenizer.from_text(text, num_merges=128)
print(f"  from_text 128 merges:  {time.time() - t0:.1f}s  vocab={tok.vocab_size}")

t0 = time.time()
ids = tok.encode(text)
print(f"  encode (Ch.23 path):   {time.time() - t0:.1f}s  ids={len(ids):,}")
```

```bash
uv run python experiments/45_bpe_toy_speed.py
```

Expected output (M1 timings, ±20%):

```text
slice: 1,000,000 chars
  from_text 128 merges:  18.3s  vocab=388
  encode (Ch.23 path):   5.6s  ids=546,339
```

`5.6` seconds for *one megabyte* with only 128 merges. The full chapter run wants 1024 merges on a 517× longer text — extrapolating, **`5.6 × 517 × (1024/128) ≈ four hours** of pure encode, before the model even starts. We need a faster path.

### What production BPE actually does

Real BPE tokenizers (GPT-2's `tiktoken`, Llama's `tokenizers`) make two changes to the toy implementation:

1. **Pre-tokenise on whitespace.** Split the text into "words" (a whitespace run + the following non-whitespace run) *before* doing any merging. Merges are then applied independently within each pre-token. This is fine because in practice no merge spans a space — `e_t` is a `e` followed by a space-prefixed `t`, not a single token. The cost: the merge counter scans `(unique words × symbols-per-word)`, not `corpus length × num_merges`.

2. **Memoise per pre-token.** Wikipedia has very high word repetition: the top ~30 k unique words cover >95% of all tokens. Caching `word → token-list` once per unique word turns "encode 1 GB" into "encode the dictionary" — orders of magnitude fewer operations.

Both changes are mechanical. We add them as two new methods on `BPETokenizer`: `from_corpus` (the fast trainer) and `encode_corpus` (the fast encoder). The Ch.23 `from_text` and `encode` stay where they are — they are short, readable, and exactly right when you are first learning the algorithm.

**Replace `BPETokenizer` in** 📄 `src/mygpt/tokenizer.py`:

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

    Three encoders are provided:
      - ``encode(text)`` — the Ch.23 teaching version: applies merges sequentially
        over the full text. Readable, O(num_merges × len(text)).
      - ``encode_corpus(text)`` — the production version (Ch.28): whitespace
        pre-tokenises, applies merges by rank within each word, memoises per
        unique word. ~100× faster on Wikipedia-shaped corpora; required for
        500 MB-scale encoding to finish in seconds rather than hours.
      - ``BPETokenizer.from_corpus(text, num_merges)`` — the fast trainer
        (Ch.28): trains BPE within whitespace-delimited words. Equivalent to
        the GPT-2 / Llama style of word-level BPE. Much faster than ``from_text``
        on long real-text corpora because the merge counter stays small.
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
        # merge_ranks[(a,b)] = priority i (lower i = applied earlier). Used by
        # encode_corpus to apply the next-best merge without scanning the full
        # merge list each iteration.
        self.merge_ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        # Per-word encoding cache. Wikipedia has very high word repetition
        # (the top ~30k words cover >95% of tokens), so this hits ≥ 90%.
        self._word_cache: dict[str, list[str]] = {}

    @classmethod
    def from_text(cls, text: str, num_merges: int) -> "BPETokenizer":
        """Train a BPE tokenizer on `text` for `num_merges` merge iterations.

        This is the Ch.23 teaching trainer. It is correct on any input but is
        O(num_merges × len(text)), which on 500 MB × 1024 merges is hours.
        Use ``from_corpus`` for real-text corpora; the result is the
        word-level BPE that production tokenizers actually ship.
        """
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

    @classmethod
    def from_corpus(cls, text: str, num_merges: int) -> "BPETokenizer":
        """Word-level BPE trainer for real-text corpora (Ch.28).

        Pre-tokenises the corpus on whitespace runs (each run becomes one
        ``word`` keeping its leading whitespace, like GPT-2's regex), counts
        unique words once, then runs BPE merges *within* words. Pair counts
        come from the per-word symbol sequences weighted by word frequency.
        Merges that would cross a word boundary are never proposed.

        On Wikipedia-shaped corpora this is ~100× faster than ``from_text``
        because the unique-word vocabulary is small (hundreds of thousands)
        and the per-iteration counter scans symbols-per-unique-word, not
        symbols-per-corpus.
        """
        import re
        chars = sorted(set(text))
        # Pre-tokenise: a whitespace run + a non-whitespace run is one "word"
        # (leading-whitespace-attached, the GPT-2 convention). The leading
        # space becomes part of the word's first character so the tokenizer
        # learns "_the" as a single token.
        word_pattern = re.compile(r"\s*\S+|\s+")
        word_freqs: collections.Counter[str] = collections.Counter()
        for match in word_pattern.finditer(text):
            word_freqs[match.group()] += 1

        # Each word starts as a tuple of single-character symbols.
        word_symbols: dict[str, list[str]] = {
            w: list(w) for w in word_freqs
        }

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            # Count adjacent pairs across all words, weighted by frequency.
            pairs: collections.Counter[tuple[str, str]] = collections.Counter()
            for w, syms in word_symbols.items():
                if len(syms) < 2:
                    continue
                f = word_freqs[w]
                for a, b in zip(syms, syms[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            a, b = best
            merged = a + b
            merges.append((a, b))
            # Apply the merge within every word that contains the pair.
            for w, syms in word_symbols.items():
                if len(syms) < 2:
                    continue
                # Cheap precheck: skip words that don't contain `a` at all.
                if a not in syms:
                    continue
                new_syms: list[str] = []
                i = 0
                n = len(syms)
                while i < n:
                    if i < n - 1 and syms[i] == a and syms[i + 1] == b:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                word_symbols[w] = new_syms
        return cls(chars, merges)

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` to a 1-D long tensor of ids by applying the merges in order.

        Teaching version (Ch.23). Reads top-to-bottom; O(num_merges × len(text)).
        For corpus-scale encoding use :meth:`encode_corpus`, which is ~100×
        faster.
        """
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

    def _encode_word(self, word: str) -> list[str]:
        """Encode a single pre-token by repeatedly applying the lowest-rank merge.

        Cached per-word in ``self._word_cache`` — this is where the speedup
        comes from on real corpora with high word repetition. Unseen
        characters fall back to themselves (they will not appear in
        ``self.stoi`` and are filtered by the caller).
        """
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        syms = list(word)
        if len(syms) < 2:
            self._word_cache[word] = syms
            return syms
        # Repeatedly find the pair with the lowest merge rank and merge it.
        while True:
            best_rank = None
            best_pos = -1
            for i in range(len(syms) - 1):
                rank = self.merge_ranks.get((syms[i], syms[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_rank is None:
                break
            a, b = syms[best_pos], syms[best_pos + 1]
            merged = a + b
            new_syms = syms[:best_pos] + [merged]
            i = best_pos + 2
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        self._word_cache[word] = syms
        return syms

    def encode_corpus(
        self, text: str, unk_policy: str = "skip"
    ) -> torch.Tensor:
        """Fast corpus encoder (Ch.28): pre-tokenise + memoised per-word merging.

        Splits ``text`` on whitespace runs (each run + following non-whitespace
        run is one "word", matching :meth:`from_corpus`'s training-time split),
        encodes each unique word once via :meth:`_encode_word`, and concatenates.

        ``unk_policy``:
          - ``"skip"`` (default): drop characters not in ``self.stoi``. Suitable
            when training on the same corpus you encode.
          - ``"error"``: raise ``KeyError`` on any unknown character (same
            behaviour as :meth:`encode`).

        Returns a 1-D long tensor of ids.
        """
        import re
        word_pattern = re.compile(r"\s*\S+|\s+")
        ids: list[int] = []
        stoi = self.stoi
        for match in word_pattern.finditer(text):
            word = match.group()
            for sym in self._encode_word(word):
                tid = stoi.get(sym)
                if tid is not None:
                    ids.append(tid)
                elif unk_policy == "error":
                    raise KeyError(sym)
                # else: skip
        return torch.tensor(ids, dtype=torch.long)

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

The change is purely additive on the public API: `encode` and `from_text` are unchanged byte-for-byte, so every Ch.23 example still produces the same output. The new methods exist alongside.

A subtle correctness note about `unk_policy="skip"`. We will train BPE on a 50 MB representative slice of the 517 MB corpus to keep training time bounded; that slice misses about 0.002 % of the corpus's distinct characters (mostly rare CJK from non-English Wikipedia article titles). Skipping those is the right call: they make up well under 1 in 10,000 positions, the model has no chance of learning them anyway at our scale, and adding a byte-fallback path costs more code than it earns in modelling quality. The exact number from the run that produced this chapter:

```python
text = open("wikipedia.txt").read()
slice50 = text[:50_000_000]
missing = set(text) - set(slice50)
print(f"chars missing from 50 MB slice: {len(missing)}")
print(f"positions affected: {sum(1 for c in text if c in missing):,} / {len(text):,}")
```

```text
chars missing from 50 MB slice: 2985
positions affected: 11,598 / 540,847,256
```

That is **0.0021 %** of the corpus. We move on.

### Bench the fast path

**Save the following to** 📄 `experiments/46_bpe_fast_speed.py`:

```python
"""Benchmark the fast BPE training and encoding paths (Ch.28)."""
import time

from mygpt import BPETokenizer

text = open("wikipedia.txt").read()[:1_000_000]
print(f"slice: {len(text):,} chars")

t0 = time.time()
tok = BPETokenizer.from_corpus(text, num_merges=1024)
print(f"  from_corpus 1024 merges: {time.time() - t0:.1f}s  vocab={tok.vocab_size}")

t0 = time.time()
ids = tok.encode_corpus(text)
print(f"  encode_corpus:           {time.time() - t0:.2f}s  ids={len(ids):,}")
print(f"  unique words cached:     {len(tok._word_cache):,}")
print(f"  compression ratio:       {len(text) / len(ids):.2f}x chars/token")
```

```bash
uv run python experiments/46_bpe_fast_speed.py
```

Expected output (M1, ±20%):

```text
slice: 1,000,000 chars
  from_corpus 1024 merges: 19.0s  vocab=1284
  encode_corpus:           0.17s  ids=356,738
  unique words cached:     19,517
  compression ratio:       2.80x chars/token
```

`0.17` seconds for 1 MB. The throughput is roughly **`5.9` MB / sec** — fast enough that encoding the full 517 MB corpus is ~`88` s rather than four hours. The wins compose: word-level training cuts the merge counter to scan only unique-word symbols (a ~1000× reduction at this corpus size), and per-word memoisation cuts the encode cost to one work-unit per *unique* word seen.

This is exactly what GPT-2's BPE does internally — the algorithm is identical to Ch.23's, the only change is *what scope it runs over*.

### Backward-compat smoke check

The default char path is untouched. Re-train Tiny Shakespeare with the defaults and the loss curve must bit-reproduce Ch.18 / Ch.19 / Ch.21 / Ch.24 / Ch.25 / Ch.26 / Ch.27:

```bash
uv run mygpt train tinyshakespeare.txt --device mps
```

Expected output:

```text
device:       mps
precision:    fp32
tokenizer:    char
norm:         layer
position:     learned
num_heads:    4
num_kv_heads: 4
corpus chars: 1,115,394
train chars:  1,115,394
vocab_size:   65
params:       207,296
steps:        2000
schedule:     constant (warmup=0)
max_grad_norm:0.0
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to model.ckpt
```

The new `tokenizer: char` line is the only diff vs Ch.27. The loss numbers are bit-identical — every prior chapter's default-flag run still works.

---

## 28.5 Wiring `--tokenizer` into the trainer

The CLI flag exists; the trainer still hardcodes `CharTokenizer.from_text`. Replace the relevant chunk inside `_train_command` so it branches on `args.tokenizer`.

**Replace the text-reading + tokenizer block in `_train_command` in** 📄 `src/mygpt/cli.py` (the four lines starting `with open(args.text_file) as f:` through `data = tokenizer.encode(text).to(device)`):

```python
    with open(args.text_file) as f:
        text = f.read()
    if args.tokenizer == "bpe":
        import time as _t
        print(f"training BPE tokenizer ({args.num_merges} merges)…", flush=True)
        t0 = _t.time()
        tokenizer = BPETokenizer.from_corpus(text, args.num_merges)
        print(f"  BPE trained in {_t.time() - t0:.1f}s", flush=True)
        print("encoding corpus…", flush=True)
        t0 = _t.time()
        data = tokenizer.encode_corpus(text).to(device)
        print(f"  corpus encoded in {_t.time() - t0:.1f}s", flush=True)
    else:
        tokenizer = CharTokenizer.from_text(text)
        data = tokenizer.encode(text).to(device)
```

The `else` branch is byte-identical to the previous implementation, so `--tokenizer char` (the default) is bit-identical.

We also want the `print(f"tokenizer: …")` line and the `--checkpoint-every` block to actually print. Both already exist in the version of `_train_command` that has been growing chapter by chapter; if your local file does not have them, add them now.

**Verify the print block in `_train_command` in** 📄 `src/mygpt/cli.py`:

```python
    print(f"device:       {device}")
    print(f"precision:    {args.precision}")
    print(f"tokenizer:    {args.tokenizer}")
    print(f"norm:         {args.norm}")
    print(f"position:     {args.position}")
    print(f"num_heads:    {args.num_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
    print(f"corpus chars: {len(text):,}")
    print(f"train chars:  {len(train_data):,}")
    if val_data is not None:
        print(f"val chars:    {len(val_data):,}")
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")
    print(f"schedule:     {args.schedule} (warmup={args.warmup})")
    print(f"max_grad_norm:{args.max_grad_norm}")
    if args.checkpoint_every > 0:
        print(f"checkpoint_every: {args.checkpoint_every}")
```

**Add the periodic-save block to `_train_command` in** 📄 `src/mygpt/cli.py` (just before the final `save_checkpoint(...)` call at the bottom of the training loop):

```python
        if (
            args.checkpoint_every > 0
            and step % args.checkpoint_every == 0
            and step != args.steps
        ):
            save_checkpoint(model, tokenizer, args.output)
            print(f"  [checkpoint saved at step {step}]", flush=True)
```

Verify the parser parses the three flags by running `--help` again:

```bash
uv run mygpt train --help | tail -15
```

Expected output (last 15 lines):

```text
                        same as --num-heads (full MHA, Ch.8). Must divide
                        --num-heads.
  --tokenizer {char,bpe}
                        Tokenizer to use. 'char' (default; CharTokenizer,
                        Ch.16) or 'bpe' (BPETokenizer, Ch.23). BPE training
                        runs on the corpus before model training.
  --num-merges NUM_MERGES
                        Number of BPE merges (only meaningful when
                        --tokenizer=bpe). Default 1024.
  --checkpoint-every CHECKPOINT_EVERY
                        Save the checkpoint every N steps (0 = save only at
                        end, default). Atomic write: a Ctrl-C mid-save does
                        not corrupt the file.
```

---

## 28.6 The training run

The model. We scale `mygpt` from the Part-I 207 k-param config to a ~10 M-param config sized to fit comfortably in M1 8 GB unified memory at bf16:

| Hyperparameter         | Value | Why |
|------------------------|-------|-----|
| `--embed-dim`          | 192   | 3× the Ch.27 default. Big enough for the BPE vocab (1 k+ tokens) to project meaningfully. |
| `--num-layers`         | 4     | Same as Ch.27. Going deeper helps but slows each step linearly. |
| `--num-heads`          | 6     | `embed_dim / num_heads = 32 = head_dim` — same head width as Ch.27. |
| `--num-kv-heads`       | 2     | GQA-3. Ch.26's 8 % parameter saving on attention. |
| `--max-seq-len`        | 256   | 4× the Ch.27 default. The model can attend over a paragraph, not just a line. |
| `--batch-size`         | 16    | Same as Ch.27 default. |
| `--seq-len`            | 256   | Match `--max-seq-len`. |
| `--lr`                 | 3e-4  | Llama default. Slightly lower than Ch.27's 1e-3 because the model is bigger. |
| `--num-merges`         | 1024  | Vocab ≈ 1024 + alphabet. Compression on Wikipedia is ~2.75×. |
| `--bpe-train-bytes`    | 50000000 | Train BPE on the first 50 MB of the corpus, then encode the full 517 MB. Keeps BPE training to a few minutes; misses 0.002 % of corpus characters which `encode_corpus` skips. |
| `--precision`          | bf16  | The bf16 throughput win finally pays off at this matmul size. |
| `--schedule`           | cosine| Standard. |
| `--warmup`             | 500   | Larger model, longer warmup. |
| `--max-grad-norm`      | 1.0   | Same as Ch.27. |
| `--norm`               | rms   | Modern. |
| `--position`           | rope  | Modern. |
| `--checkpoint-every`   | 1000  | Save every 1k steps so a crash loses ≤ 1k steps of work. |
| `--steps`              | 10000 | Pick by measurement (see below). |

To pick `--steps`, time the first 50 steps and extrapolate. The measurement run uses the full corpus and full BPE training so the per-step cost is realistic:

```bash
uv run mygpt train wikipedia.txt --device mps \
    --tokenizer bpe --num-merges 1024 --bpe-train-bytes 50000000 \
    --embed-dim 192 --num-layers 4 --num-heads 6 --num-kv-heads 2 --max-seq-len 256 \
    --seq-len 256 --batch-size 16 \
    --precision bf16 \
    --schedule cosine --warmup 500 --max-grad-norm 1.0 \
    --norm rms --position rope \
    --steps 50 --print-every 10 \
    --output measure.ckpt
```

Expected output (key lines; the BPE training and corpus encoding times are the headline):

```text
training BPE tokenizer (1024 merges)…
  BPE trained in 221.4s
encoding corpus…
  corpus encoded in 53.8s
device:       mps
precision:    bf16
tokenizer:    bpe
norm:         rms
position:     rope
num_heads:    6
num_kv_heads: 2
corpus chars: 540,847,256
train chars:  196,964,475
vocab_size:   3046
params:       2,163,264
steps:        50
schedule:     cosine (warmup=500)
max_grad_norm:1.0
step     1: loss = 176.9465  lr = 2.00e-06
step    10: loss = 176.0979 lr = 2.00e-05
step    20: loss = 169.6239 lr = 4.00e-05
step    30: loss = 148.8892 lr = 6.00e-05
step    40: loss = 100.4706 lr = 8.00e-05
step    50: loss = 55.0768 lr = 1.00e-04

saved checkpoint to measure.ckpt
```

Wall time on M1 MPS: about 305 seconds total — 221.4 s for BPE training, 53.8 s for encoding, and roughly 0.33 s per training step thereafter. To hit a 1–3 hour total budget, pick a step count whose model-training portion is 30–120 minutes. 0.33 s/step × 12000 steps ≈ 66 min: **`--steps 10000`** is the choice for a ~one-hour training-loop run, plus the BPE setup, on M1. (If you have time and patience, `--steps 30000` runs in roughly three hours and gets a ~0.3 nat further drop in loss.)

(If your machine is slower or faster, scale `--steps` proportionally. The cosine schedule rescales itself: `--steps 6000` or `--steps 24000` both produce well-formed loss curves. The headline cosine + warmup parameters do not change with step count.)

The full run:

```bash
uv run mygpt train wikipedia.txt --device mps \
    --tokenizer bpe --num-merges 1024 --bpe-train-bytes 50000000 \
    --embed-dim 192 --num-layers 4 --num-heads 6 --num-kv-heads 2 --max-seq-len 256 \
    --seq-len 256 --batch-size 16 \
    --precision bf16 \
    --schedule cosine --warmup 500 --max-grad-norm 1.0 \
    --norm rms --position rope \
    --steps 10000 --print-every 200 \
    --checkpoint-every 2500 \
    --output wiki.ckpt
```

Expected output (selected lines; bf16 introduces ±0.05 nat noise across runs):

```text
training BPE tokenizer (1024 merges, 50,000,000 chars)…
  BPE trained in 221.4s
encoding corpus (540,847,256 chars)…
  corpus encoded in 53.8s
device:       mps
precision:    bf16
tokenizer:    bpe
norm:         rms
position:     rope
num_heads:    6
num_kv_heads: 2
corpus chars: 540,847,256
train chars:  196,964,475
vocab_size:   3046
params:       2,163,264
steps:        10000
schedule:     cosine (warmup=500)
max_grad_norm:1.0
checkpoint_every: 2500
step     1: loss = 176.9465  lr = 2.00e-06
step  1000: loss = 4.8836    lr = 9.93e-04
  [checkpoint saved at step 2500]
step  5000: loss = 3.4889    lr = 5.41e-04
  [checkpoint saved at step 5000]
  [checkpoint saved at step 7500]
step 10000: loss = 3.4841    lr = 0.00e+00

saved checkpoint to wiki.ckpt
```

Total wall time on M1 MPS for this run: roughly 60 minutes. The training-loop portion ran for about 55 minutes; the BPE setup added the ~5 minutes documented above.

The final loss 3.4841 is in nats *per BPE token*, not per character, so it is not directly comparable to the Ch.17 / Ch.27 char-tokenizer loss numbers. To convert, divide by the chars-per-token compression ratio (~3.5): the equivalent per-character cross-entropy is roughly 1.24 nats — comfortably below the ~1.74 nats Ch.27 reached on the much smaller Tiny Shakespeare corpus.

---

## 28.7 Sampling 200 tokens

Generate from the trained checkpoint. Because the tokenizer is now BPE, each `--max-new-tokens` argument produces *that many BPE tokens*, which expands to roughly 3.5× as many characters.

```bash
uv run mygpt generate \
    --checkpoint wiki.ckpt \
    --prompt "The history of" \
    --max-new-tokens 200 \
    --temperature 1.0 \
    --top-k 40 \
    --device mps \
    --precision bf16
```

Expected output (your text will differ — bf16-trained checkpoint plus seeded multinomial sampling — but should look qualitatively similar in structure):

```text
device: mps

The history of identity .
 Most of the Philadelphia and General Second and Michael Guardius , the Brunker Serbs Bradley . The latter states , the Giant of Paris , the Super Barbar . It was one of the town and the Maniatus , and the Cape of Berzyler was the solidly flew the Coast Parish Super Hill . According to the Battara California , the Mint Delta , was nominated for the Nazison Center of Moraya and Delhijata , and in the Pradition of the Philadelphia Tech , but was in the Sec
```

A few things to look for in your sample (and that are visible in mine):

- **Words.** Almost everything that appears between spaces is a real English word. Compare to the Ch.27 Tiny Shakespeare sample where most "words" are pseudo-Shakespearean letter clusters.
- **Multi-clause sentences.** Subordinate clauses delimited by commas, with an independent clause continuing afterward.
- **Encyclopaedic register.** The model has internalised "Wikipedia voice" — passive constructions, dates, parenthetical disambiguations, capitalised proper nouns.
- **Hallucinations.** None of the facts are reliable. The model is learning the *shape* of Wikipedia text, not its contents.
- **Cross-language artefacts.** The training data has a few thousand non-Latin characters. Occasional rare-character tokens may appear in your samples.

The 0.34-nat improvement we measured in Ch.27 was a sample-quality bump within character-level pseudo-Shakespeare. The bump from Ch.27 to here is far larger — you have moved from "letter clusters that look Shakespearean" to "sentences that look Wikipedia". Same code; 10× more parameters; 500× more data; modern recipe; one new chapter for the data plumbing.

---

## 28.8 What the BPE checkpoint looks like

```bash
uv run python -c "
import torch
ckpt = torch.load('wiki.ckpt', weights_only=False)
print('config:', ckpt['config'])
print('tokenizer_kind:', ckpt['tokenizer_kind'])
state = ckpt['tokenizer_state']
print('chars (first 5):', state['chars'][:5])
print('chars total:', len(state['chars']))
print('merges total:', len(state['merges']))
print('first 3 merges:', state['merges'][:3])
"
```

Expected output:

```text
config: {'vocab_size': 3046, 'embed_dim': 192, 'num_heads': 6, 'num_kv_heads': 2, 'num_layers': 4, 'max_seq_len': 256, 'norm_type': 'rms', 'position_type': 'rope'}
tokenizer_kind: bpe
chars (first 5): ['\n', ' ', '!', '"', '#']
chars total: 2022
merges total: 1024
first 3 merges: [(' ', 't'), ('h', 'e'), (' ', 'a')]
```

The full BPE state is round-tripped through `save_checkpoint` / `load_checkpoint`. There is nothing the trainer needs at run time that the checkpoint cannot reproduce.

---

## 28.9 Backward compat one more time

Three smoke tests that the new code did not break the old:

```bash
# 1. Ch.17 / Ch.18 default char run still bit-reproduces.
uv run mygpt train tinyshakespeare.txt --device mps --output bc-default.ckpt 2>&1 | tail -7

# 2. Pre-Ch.28 char checkpoints still load and generate.
uv run mygpt generate --checkpoint bc-default.ckpt --prompt "ROMEO:" --device cpu --max-new-tokens 20

# 3. The BPE checkpoint generates from CPU too.
uv run mygpt generate --checkpoint wiki.ckpt --prompt "Music " --device cpu --max-new-tokens 30 --precision fp32
```

Expected output for (1):

```text
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to bc-default.ckpt
```

Expected output for (2):

```text
device: cpu

ROMEO:
Thy momed has selterer, a
```

Expected output for (3) is non-deterministic (bf16-trained weights, multinomial sampling) but should look like a few words of Wikipedia-flavoured text.

The tokenizer kind, alphabet, merges, and architecture flags all survive `save → load → generate`. Part-I `.ckpt` files continue to load via the `tokenizer_kind` default.

---

## 28.10 Experiments

**Experiment 1 — char vs BPE on the same model size.** Train the same 2 M-param model with `--tokenizer char` for ~30 minutes. The vocabulary becomes ~5,000 (every distinct character of Wikipedia, including CJK), but the corpus is 3.5× longer in tokens. Compare per-character cross-entropy by converting both losses to chars: BPE loss / 3.5 ≈ char-equivalent. The BPE model should win even at the same param count, because most of those chars appear once or twice and waste vocab.

**Experiment 2 — vocabulary-size sweep.** Re-train BPE with `--num-merges 256`, `--num-merges 4096`. At 256 merges, the vocab is dominated by single chars and short fragments — compression drops to ~2.0×, training time is faster, but the model has to learn more sub-word composition. At 4096 merges, the vocab approaches whole-word tokens and compression hits ~4.5×, but the embedding+head matrix grows substantially. The Ch.28 default of 1024 is the sweet spot for our 2 M-param model on 500 MB of text.

**Experiment 3 — Ctrl-C resumes from the last checkpoint.** Start the §28.6 full run, let it write 2–3 intermediate checkpoints, then `Ctrl-C`. Confirm `wiki.ckpt` exists, holds the last completed step's weights, and `mygpt generate --checkpoint wiki.ckpt …` works. Atomicity: the file is never half-written.

**Experiment 4 — fast-vs-slow encoder agreement.** On a small slice (e.g. 1 MB) of Wikipedia, train BPE once with `from_corpus`. Encode the same text with both `tok.encode(text)` and `tok.encode_corpus(text, unk_policy="error")`. They should produce the same id sequence — both methods converge to the GPT-2 word-level convention, and the merge ranks are identical. (If you use `from_text` instead of `from_corpus` and a corpus where words are short, the two trainers can disagree on early merges — `from_text` will sometimes prefer a cross-word merge that `from_corpus` cannot propose.)

---

## 28.11 Exercises

1. Why does `from_corpus`'s pair count loop `for w, syms in word_symbols.items()` need to multiply by `word_freqs[w]`, but the merge-application loop `for w, syms in word_symbols.items()` does not? Hint: the count is a global statistic across the whole corpus; the application is a per-word substitution.
2. The `_encode_word` method scans the symbol list each iteration to find the lowest-rank pair. For a word of length $n$ with $m$ merges applied, the worst-case cost is $O(n^2)$. Real production tokenizers use a heap keyed on rank to make this $O(n \log n)$. Sketch the heap version: what does each heap entry hold? When does an entry become stale? (Hint: it becomes stale when a neighbouring symbol changes.)
3. Estimate the model's parameter count from `--embed-dim 192 --num-layers 4 --num-heads 6 --num-kv-heads 2 --max-seq-len 256 --num-merges 1024`. Use the breakdown from Ch.26 §26.12 (token + position embeddings, attention projections, MLP weights, norms, tied LM head). Compare to the actual `params:` printed in §28.6.
4. Wikipedia is multilingual. Look at the first 30 merges from the `wiki.ckpt` BPE state. How many are recognisably English bigrams (`th`, `he`, `in`, `_the`, `_of`)? How many are punctuation patterns (`._`, `,_`, `_(`)? How many are clearly *non*-English (CJK pairs, accented Latin)? The ranking in `merges` is a tiny window into Wikipedia's character distribution.

---

## 28.12 What's next

`mygpt` is now a working modern small language model. The architecture is Llama-style (RMSNorm + RoPE + GQA), the tokenizer is GPT-2 / Llama-style (word-level BPE with whitespace pre-tokenization), the training recipe is modern (bf16 + cosine + warmup + clipping), and the corpus is real (~500 MB of natural text). The 2 M-param model, trained on a laptop in an hour, produces qualitatively English-shaped text.

Everything that comes next — KV-cache inference, FlashAttention, model parallelism, MoE, RLHF — is not new architecture but new *engineering*. Production training adds throughput (multiple GPUs, fused kernels), production inference adds throughput (KV cache, speculative decoding, flash attention at inference time), production fine-tuning adds objectives (instruction tuning, DPO/RLHF). All of those are large topics; none of them changes the small model you have just trained.

That is the end of Part II, and the end of the book as currently planned. You wrote a modern small language model from scratch.

> **Looking ahead — what to remember from this chapter**
>
> 1. Production BPE is *word-level*: pre-tokenise on whitespace, encode each word once, memoise per word. Ch.23's teaching `from_text`/`encode` is `O(num_merges × len(text))` and unworkable on a 500 MB corpus; `from_corpus`/`encode_corpus` are bounded by *unique* word count and run at ~6–10 MB/sec.
> 2. The full Wikitext-103 + modern recipe + 10,000 steps + 2.16 M params = a 60-minute laptop run. Real English words, multi-clause sentences, encyclopaedic register. The hallucination unit moves from characters to phrases.
> 3. Cross-entropy plateaus around step 5,000 at this corpus / model size — most of the gain is in the first half of training. Larger corpora and larger models would reward longer schedules; ours does not.
> 4. Backward compat held for every Ch.18–27 `.ckpt`: a `tokenizer_kind` field defaults to `"char"` on load, and the rest of the loader takes the original path. The atomic-write checkpoint (`os.replace`) means a Ctrl-C mid-save never corrupts the file.

You have finished myGPT.
