---
title: 22. BPE from scratch (algorithm)
nav_order: 4
parent: Part II — Advanced Topics
---

# Chapter 22 — Byte-pair encoding from scratch

The `CharTokenizer` from Chapter 16 has a 65-character vocabulary and turns every Tiny Shakespeare character into one token. That is the simplest tokenizer imaginable, and it has two costs:

1. **Sequences are long.** A 100-character paragraph is 100 tokens, not 20. Every `mygpt` operation that scales with sequence length pays for the redundancy.
2. **The model has to learn long-range character chains.** "Q is followed by U" is a *fact about English* that the model has to rediscover from gradient updates instead of getting it for free.

Real LLMs use **byte-pair encoding** (BPE): a tokenizer that starts from characters, then *iteratively merges the most-frequent adjacent pair* into a single new token, and repeats until the vocabulary reaches a target size. The algorithm was originally a data-compression technique (Gage, 1994) and was adapted to NLP by Sennrich, Haddow, and Birch in 2016. Modern variants (GPT-2, Llama) all start from BPE.

This chapter teaches the algorithm. It does **not** modify `mygpt` — Chapter 23 will wire it in. Here we hand-trace BPE on Sennrich's canonical four-word example and watch ten merges build it up.

By the end you will have:

- understood the BPE algorithm in five lines (initialise, count pairs, find most-frequent, merge, repeat),
- traced ten merges on the example `"low low low low low lower newer newer newer newer wider"` by hand and by Python,
- seen the final tokenization where `"low"` and `"newer"` are each *one token*, demonstrating compression,
- understood how production BPE (GPT-2, tiktoken) differs from the textbook version.

---

## 22.1 Setup

This chapter assumes Chapter 21 — `mygpt/` exists with `--device`, `--precision`, `--val-split`, `--schedule`, `--warmup`, `--max-grad-norm`, plus the `cosine_warmup_lr` and `estimate_val_loss` helpers. None of those are touched in this chapter.

If you skipped, the only piece you actually need from Part I is `experiments/`:

```bash
mkdir -p experiments
```

You are ready.

---

## 22.2 Why merge pairs?

Pick any English text and look at the character pair frequencies. `t` followed by `h` happens constantly (every `the`, `that`, `this`, `with`). `q` followed by `u` happens whenever `q` shows up. If we replace those pairs with a single token, two things happen:

- **Sequences shrink.** "the cat" stops being 7 tokens (`t-h-e- -c-a-t`) and becomes 5 tokens (`th-e- -c-a-t`) — or 3 tokens (`the- -cat`) if we keep merging.
- **The model gets a head-start.** It no longer has to learn "after `t` predict `h` with high probability" — `th` is one token, atomic, and the next-token prediction skips that step.

The trick is to *let the data decide which pairs to merge*. Instead of "merge `th` because English-speakers know it's common", we count pair frequencies in the corpus and merge whichever is most-frequent right now. After that merge, count pairs again — but now the previously-merged pair is a single symbol, so new pairs (e.g. `th` paired with `e` to make `the`) become candidates. Repeat until you have as many merges as you want.

The vocabulary after `N` merges has size `(initial chars) + N`. Tiny Shakespeare's 65 starting characters plus 1024 BPE merges give a vocabulary of `1089` tokens — a 17× reduction in *number* of tokens compared to character-level on the same corpus, and a roughly 3× reduction in tokens-per-character of text.

---

## 22.3 The algorithm in five steps

Notation: each "word" in the corpus is represented as an ordered tuple of *symbols*, where each symbol is initially one character. We append a special end-of-word marker `</w>` to every word so the algorithm can tell `low` (the word) apart from `low` (the prefix of `lower`).

```text
Init:     for each word w in the corpus, count words[w]; represent w as
          (c1, c2, ..., cn, </w>) — its characters plus the marker.
Step 1:   count pair frequencies. For each (symbols, freq) in the vocab, and each
          adjacent pair (sym_i, sym_i+1), add freq to that pair's count.
Step 2:   find the most-frequent pair (a, b).
Step 3:   merge: in every (symbols, freq), replace every adjacent (a, b) with
          the new symbol "ab". Vocabulary now has one extra token.
Step 4:   repeat steps 1–3 until the vocab reaches the target size.
Step 5:   the merged tokens — in the order they were created — ARE the BPE rules.
          To tokenize new text, apply the rules in order: scan, merge, scan, merge.
```

That's it. Five steps, no neural network involved. The "training" of a BPE tokenizer is just frequency counting and replacement; it takes seconds even on Tiny Shakespeare.

---

## 22.4 The algorithm in code

We capture the algorithm as one standalone Python script. No `mygpt` import is needed — BPE has nothing to do with PyTorch, models, or training.

**Save the following to** 📄 `experiments/41_bpe_by_hand.py`:

```python
"""Experiment 41 — Byte-pair encoding traced by hand on Sennrich's example.

The algorithm has five steps:
  - Initialise: words → (chars + '</w>') tuples, with their counts.
  - Count adjacent pair frequencies across the whole vocabulary.
  - Pick the most-frequent pair.
  - Merge that pair everywhere it occurs.
  - Repeat steps 2-4 until the desired number of merges.
"""

import collections


def init_vocab(corpus_words: list[str]) -> dict[tuple[str, ...], int]:
    """Each word becomes a tuple of (chars + '</w>'); value is its frequency."""
    counts = collections.Counter(corpus_words)
    return {tuple(list(word) + ["</w>"]): freq for word, freq in counts.items()}


def get_pair_counts(
    vocab: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int]:
    """Count adjacent (a, b) pair frequencies across the whole vocabulary."""
    pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str],
    vocab: dict[tuple[str, ...], int],
) -> dict[tuple[str, ...], int]:
    """Replace every adjacent (a, b) in vocab's keys with 'ab'."""
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


def main() -> None:
    corpus = "low low low low low lower newer newer newer newer wider".split()
    vocab = init_vocab(corpus)

    print("Initial vocab:")
    for k, v in vocab.items():
        print(f"  {v}: {' '.join(k)}")
    print()

    merges = []
    for step in range(10):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair, best_count = max(pairs.items(), key=lambda kv: kv[1])
        merged = best_pair[0] + best_pair[1]
        print(
            f"Step {step + 1}: merge ({best_pair[0]!r}, {best_pair[1]!r}) "
            f"-> {merged!r}  (freq={best_count})"
        )
        merges.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

    print()
    print("Final vocab:")
    for k, v in vocab.items():
        print(f"  {v}: {' '.join(k)}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/41_bpe_by_hand.py
```

**Expected output:**

```text
Initial vocab:
  5: l o w </w>
  1: l o w e r </w>
  4: n e w e r </w>
  1: w i d e r </w>

Step 1: merge ('l', 'o') -> 'lo'  (freq=6)
Step 2: merge ('lo', 'w') -> 'low'  (freq=6)
Step 3: merge ('e', 'r') -> 'er'  (freq=6)
Step 4: merge ('er', '</w>') -> 'er</w>'  (freq=6)
Step 5: merge ('low', '</w>') -> 'low</w>'  (freq=5)
Step 6: merge ('n', 'e') -> 'ne'  (freq=4)
Step 7: merge ('ne', 'w') -> 'new'  (freq=4)
Step 8: merge ('new', 'er</w>') -> 'newer</w>'  (freq=4)
Step 9: merge ('low', 'er</w>') -> 'lower</w>'  (freq=1)
Step 10: merge ('w', 'i') -> 'wi'  (freq=1)

Final vocab:
  5: low</w>
  1: lower</w>
  4: newer</w>
  1: wi d er</w>
```

---

## 22.5 Reading the trace

Step-by-step the algorithm grows the vocabulary by one token at each merge:

- **Step 1** picks `('l', 'o')` because `lo` appears 6 times across the corpus (5 in `low`, 1 in `lower`). That's the most-frequent pair before any merging.
- **Step 2** picks `('lo', 'w')` — same 6 occurrences, but now starting from the just-merged `lo`. The merge turns `lo w </w>` into `low </w>`.
- **Step 3** picks `('e', 'r')`, frequency 6 (1 in `lower`, 4 in `newer`, 1 in `wider`). Even though we have already created `lo` and `low`, this `er` pair was untouched — and it's now the most frequent.
- **Step 4** glues `('er', '</w>')` because every word that ends in `r` in this corpus also ends with `</w>` immediately after — total 6 occurrences. The merged token `er</w>` is "an `er` that ends a word".
- **Steps 5-7** progressively build `low</w>` (5), `ne` (4), `new` (4) — each merge consolidating the most-frequent pair currently visible.
- **Step 8** is the first really big merge: `('new', 'er</w>')`. Both pieces are themselves the result of earlier merges. The whole word `newer</w>` becomes one token, freq 4.
- **Step 9** merges `('low', 'er</w>')` to make `lower</w>` — this happens only once (the corpus has just one `lower`), so its frequency is 1, but it's still the highest-frequency pair remaining.
- **Step 10** dips into the `wider` word: `('w', 'i')` is the only pair with frequency 1 that we haven't merged yet. The whole word `wider` is rare enough that it never gets fully consolidated within 10 merges.

The final vocabulary tokenizes the four word-types as:

| Word type | Frequency | Tokens after 10 merges |
|-----------|-----------|------------------------|
| `low`     | 5         | `low</w>` (1 token)     |
| `lower`   | 1         | `lower</w>` (1 token)   |
| `newer`   | 4         | `newer</w>` (1 token)   |
| `wider`   | 1         | `wi d er</w>` (4 tokens)|

Three of the four word-types collapsed to a single token; the fourth (the rarest) stayed mostly as characters. **This is BPE's central trade**: frequent strings become atomic, rare strings stay broken-up. The model gets a head-start on common words and *can still represent* uncommon words via their character decomposition.

---

## 22.6 The merge order is the rulebook

The list of `(a, b) → ab` merges, in the order they were chosen, *is* the tokenizer. To tokenize a new word, you start from its characters plus `</w>` and apply the merge rules **in the order they were created**, repeatedly, until no more apply. For our 10 merges:

```text
1.  l + o     → lo
2.  lo + w    → low
3.  e + r     → er
4.  er + </w> → er</w>
5.  low + </w>→ low</w>
6.  n + e     → ne
7.  ne + w    → new
8.  new + er</w> → newer</w>
9.  low + er</w> → lower</w>
10. w + i     → wi
```

A new word like `slow` would tokenize as: start with `s l o w </w>`. Apply rule 1 (`l + o → lo`) where it fits: `s lo w </w>`. Apply rule 2 (`lo + w → low`): `s low </w>`. Apply rule 5 (`low + </w> → low</w>`): `s low</w>`. No more rules apply (rule 3 needs `e + r`, etc.). Final: `s low</w>` — 2 tokens.

A new word like `slower` would tokenize as: `s l o w e r </w>` → after rules 1, 2, 3, 4: `s low er</w>` → after rule 9 (`low + er</w> → lower</w>`): `s lower</w>` — 2 tokens.

A word with no rule applying at all (e.g. `xyz`) stays at character level: `x y z </w>` — 4 tokens. **Every** byte sequence has *some* tokenization, even if it's "all characters". BPE never fails to tokenize.

---

## 22.7 What production BPE does differently

Two things separate this textbook BPE from the GPT-2 / tiktoken / Llama variants:

**1. Byte-level, not character-level.** GPT-2's BPE operates on bytes, not Unicode characters. The initial vocabulary has 256 entries (one per byte value). Any UTF-8 input — including emoji, Chinese, Cyrillic, code points the training corpus never saw — encodes to a unique byte sequence first, then runs through BPE. Result: GPT-2's tokenizer can encode *any* Unicode text without an "unknown token", at the cost of representing non-ASCII characters as multiple bytes.

**2. No `</w>` marker.** GPT-2's BPE does not use end-of-word markers. Instead, it pre-splits text with a regex that keeps spaces attached to the *following* word — so " the" (space-prefixed) and "the" (after no space) tokenize differently. This is why you'll see `Ġthe` in GPT-2's vocabulary — `Ġ` is GPT-2's printable representation of the leading-space byte. The end-of-word marker isn't needed because the *leading* space disambiguates.

We will implement the textbook version (with `</w>`) in Chapter 23 because it is simpler to reason about and achieves the same compression ratios on Tiny Shakespeare. The byte-level BPE adds engineering, not pedagogy.

---

## 22.8 Experiments

1. **Different number of merges.** Edit `experiments/41_bpe_by_hand.py` to use `range(20)` instead of `range(10)`. Steps 11–12 finish off `wider`: step 11 is `('wi', 'd') → wid`, step 12 is `('wid', 'er</w>') → wider</w>`. After step 12 every word-type is a single token, so `get_pair_counts` returns an empty dict and the loop's early-exit `if not pairs: break` fires. Total of 12 merges, after which the vocab cannot grow further on this corpus.

2. **A different corpus.** Replace the `corpus = ...` line with `corpus = "the cat sat on the mat the cat ran".split()`. The first merge is `('a', 't')` with frequency 4 — once each in `cat ×2`, `sat`, and `mat`. The pair `('t', 'h')` is the *second*-most-frequent (3 occurrences in `the`). After 6 merges or so, `the</w>`, `cat</w>`, and `at</w>` all show up as single tokens. Watch how quickly common short words become atomic when the corpus has more repetition.

3. **The `</w>` marker matters.** Remove the `+ ["</w>"]` from `init_vocab` so words are pure character tuples without an end marker. Re-run on Sennrich's example. Observe that step 4 disappears (no `er</w>` token) and `low` and `lower` become indistinguishable as prefixes — the algorithm now happily merges across word boundaries. (In a real BPE this is fine because pre-splitting handles word boundaries; in this toy version it is a bug.)

4. **What happens if all pairs are tied?** With a corpus of `["abcd"]` (one occurrence), every adjacent pair has frequency 1. The algorithm picks one based on Python's dict iteration order. `max(...)` in CPython returns the *first* occurrence of the maximum, so it's deterministic given a fixed input. Verify experimentally: run with corpus `["abcd"] * 1` ten times and confirm the merges are identical.

After each experiment, restore any file you changed.

---

## 22.9 Exercises

1. **Tokenize `lowest`.** With our 10-merge tokenizer, walk through what `lowest` becomes. Answer: `lowest</w>` doesn't fit any whole-word rule because `lowest` wasn't in the training corpus. Apply rules 1, 2 (build `low`), then `low e s t </w>`. Rule 5 is `low + </w>`, but the next symbol is `e` not `</w>` so it doesn't fire. Final: `low e s t </w>` — 5 tokens. Confirm by adding a small `tokenize()` function to `experiments/41_bpe_by_hand.py` that applies the merge list to a new word.

2. **Why is the merge *order* significant?** Argue that applying the rules in step-creation order is *not* equivalent to picking, at each tokenization position, the longest applicable merge. (Hint: rule 5 (`low</w>`) was created before rule 9 (`lower</w>`); if we apply rules greedily from the corpus we'd over-merge `lower` into `low</w>` `er` instead of `lower</w>`.)

3. **Vocabulary size after K merges.** Argue that after `K` merges, the BPE vocabulary has exactly `len(initial_chars) + K` entries. Why doesn't a merge ever *reduce* vocabulary size? (Hint: the merged tokens are added; the original component tokens stay in the vocab because they may still be needed for words the merge doesn't apply to.)

4. **A pathological corpus.** Suppose the corpus has only one word: `corpus = ["aaaaaaaa"]` (eight `a`s). Trace 4 merges by hand. (Hint: step 1 merges `('a', 'a') → aa`, step 2 merges `('aa', 'aa') → aaaa`, etc. After 3 merges the whole word is one token `aaaaaaaa`.)

---

## 22.10 What's next

We have the algorithm. The next chapter, **Chapter 23 — `BPETokenizer` in `mygpt`**, packages it up into a `mygpt.BPETokenizer` class with the same shape as `CharTokenizer` (constructor, `encode`, `decode`, `save`, `load`) and adds a `--tokenizer {char, bpe}` flag to the CLI. By the end of Ch.23 we will have trained a BPE tokenizer on Tiny Shakespeare with 1024 merges and confirmed it reduces the encoded sequence length by roughly 2.7×.

Looking ahead — what to remember from this chapter:

1. BPE is "iteratively merge the most-frequent adjacent pair" and nothing more sophisticated.
2. The output of BPE training is a **list of merge rules in order**; that list, plus the initial alphabet, IS the tokenizer.
3. Frequent strings become single tokens; rare strings stay broken-up — this trade-off is what makes the tokenizer compact *and* able to handle arbitrary input.
4. Production BPE (GPT-2, tiktoken) replaces character-level with byte-level and `</w>` with a regex pre-split; the algorithm itself is unchanged.
