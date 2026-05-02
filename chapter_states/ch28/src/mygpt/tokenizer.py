import collections
import json
import re

import torch


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
