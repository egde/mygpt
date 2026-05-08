# Student walkthrough report: Chapter 22 — BPE from scratch (algorithm)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/22_bpe_from_scratch.md`
- Total sections walked: 5 of 5 executable subsections (§22.1 setup, §22.4 experiment, §22.5 trace verification, §22.6 prose tokenization claims, §22.8 + §22.9 spot-checks)
- Files written: 1 — `experiments/41_bpe_by_hand.py`
- Files modified: 0 (chapter does not touch `mygpt/`)
- Shell commands run: 5 (the experiment + 3 ad-hoc verifications + 1 setup smoke test)
- Issues found: **1** (0 blockers, 1 mismatch, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Will gently snag a careful student in §22.8 exp 2** — the first-merge prediction is wrong. Easy one-line fix; chapter is otherwise clean.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- Working directory: `/tmp/code-along-runs/ch22-design-131729/mygpt` (the Ch.21 ending-state uv project)

## Walkthrough

### Section: §22.1 Setup
- Files written: none.
- Commands run: `mkdir -p experiments` (already exists from Ch.21 state).
- Issues raised here: none. The chapter notes that `mygpt/` is untouched, which matches the file system state.

### Section: §22.4 The algorithm in code
- Files written: `experiments/41_bpe_by_hand.py` (verbatim from §22.4).
- Commands run: `uv run python experiments/41_bpe_by_hand.py`
- Output:
  ```
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
- Expected output match: yes — exact, character-for-character.
- Issues raised here: none.

### Section: §22.5 Reading the trace
- Pure prose, no executable content. Each of the 10 step explanations matches the captured trace. The final-vocab table accurately reflects the printed output (5×low, 1×lower, 4×newer, 1×wider with `wi d er</w>` token sequence).
- Issues raised here: none.

### Section: §22.6 The merge-order rulebook
- Verified the three prose tokenization claims with an ad-hoc `tokenize(word, merges)` function (apply merges in order; left-to-right scan; repeat until none apply):
  - `slow` → `['s', 'low</w>']` (2 tokens) ✓ — matches "Final: `s low</w>` — 2 tokens".
  - `slower` → `['s', 'lower</w>']` (2 tokens) ✓ — matches "Final: `s lower</w>` — 2 tokens".
  - `xyz` → `['x', 'y', 'z', '</w>']` (4 tokens) ✓ — matches "stays at character level: `x y z </w>` — 4 tokens".
- Expected output match: yes.
- Issues raised here: none.

### Section: §22.8 — experiment spot-checks
- **Exp 1 (range(20))**: re-ran with the loop expanded to `range(20)`. Output continues:
  ```
  Step 11: ('wi', 'd')   -> 'wid'        freq=1
  Step 12: ('wid', 'er</w>') -> 'wider</w>' freq=1
  ```
  Then the loop exits via `if not pairs: break` because every word-type is now a single token. Total = 12 merges. ✓ Matches the chapter's revised claim.
- **Exp 2 (different corpus `"the cat sat on the mat the cat ran"`)**: verified against the chapter's prediction. Vocab init produces:
  ```
  ('t','h','e','</w>'): 3, ('c','a','t','</w>'): 2, ('s','a','t','</w>'): 1,
  ('o','n','</w>'): 1, ('m','a','t','</w>'): 1, ('r','a','n','</w>'): 1
  ```
  Top pair: **`('a', 't')` with freq 4** (cat ×2 + sat ×1 + mat ×1). The chapter claims the first merge is `('t', 'h')` freq 4. **That's wrong** — `('t', 'h')` has freq 3 (only `the` ×3), not 4, and is not the top pair. Issue #1.
- **Exp 4 (tied frequencies on `["abcd"]`)**: not executed (the experiment is about determinism rather than a numerical claim; the chapter's reasoning ("`max(...)` returns the first occurrence") is a Python-language fact, not a BPE fact).
- Issues raised here: #1.

### Section: §22.9 — exercise spot-check
- **Exercise 1 (`lowest`)**: applied the 10-merge rules to `lowest`. Result: `['low', 'e', 's', 't', '</w>']` — 5 tokens. ✓ Matches the chapter's claim "Final: `low e s t </w>` — 5 tokens".
- **Exercise 4 (`["aaaaaaaa"]`)**: traced with my BPE code (verified during pre-review):
  ```
  step 1: ('a','a') -> 'aa'         freq=7
  step 2: ('aa','aa') -> 'aaaa'     freq=3
  step 3: ('aaaa','aaaa') -> 'aaaaaaaa'  freq=1
  step 4: ('aaaaaaaa','</w>') -> 'aaaaaaaa</w>'  freq=1
  ```
  ✓ Matches the chapter's hint exactly (steps 1–3 produce the merges named; the chapter says "After 3 merges the whole word is one token `aaaaaaaa`" which is true at the symbol level — step 4 just attaches the end-of-word marker).
- Issues raised here: none.

## Issues

### Issue #1 — Mismatch — §22.8 exp 2 first-merge prediction is wrong
- Section: §22.8 — experiment 2 prose.
- Where: *"The first merge is `('t', 'h')` because `th` appears 4 times (3 in `the`, 1 in `the`)."*
- What happened: Empirically, on the corpus `"the cat sat on the mat the cat ran"`, the top pair is **`('a', 't')` with frequency 4** (cat ×2 + sat ×1 + mat ×1), not `('t', 'h')`. The pair `('t', 'h')` has frequency 3 (only `the` ×3). The chapter's claim is wrong on both the pair identity and the frequency-counting for `t-h`.
- Why a student would be stuck: not stuck — the experiment still runs and produces a valid trace — but a student following the prediction will see `('a', 't')` as step 1 instead of `('t', 'h')` and may worry their code is broken.
- Suggested fix: change the prose to: *"The first merge is `('a', 't')` (frequency 4 — once each in `cat ×2`, `sat`, and `mat`). The pair `('t', 'h')` is second-most frequent with 3 occurrences in `the`. After the `('a', 't')` merge, subsequent merges progressively build up the tokens `at</w>`, `cat</w>`, `the</w>`, etc."* The chapter also has a typo "(3 in `the`, 1 in `the`)" that should disappear with this rewrite.

## Confidence and caveats

I walked the chapter end-to-end and re-verified every numerical claim and every prose tokenization. The §22.4 expected-output block, the §22.5 step-by-step interpretation, the §22.6 three new-word tokenizations, the §22.9 exercise 4 trace, and the §22.8 exp 1 step-12 termination claim are all empirically reproduced exactly. The lone failure is §22.8 exp 2's first-merge prediction.

§22.8 exp 3 (removing the `</w>` marker) and exp 4 (tied-frequency determinism) are framed as suggested investigations rather than precise empirical claims; their conceptual argument holds without needing to re-run.

The chapter is one prose fix away from clean.

---

## Review #2 — fix verified

After the fix, §22.8 exp 2 now reads:
> "The first merge is `('a', 't')` with frequency 4 — once each in `cat ×2`, `sat`, and `mat`. The pair `('t', 'h')` is the *second*-most-frequent (3 occurrences in `the`). After 6 merges or so, `the</w>`, `cat</w>`, and `at</w>` all show up as single tokens."

Empirical 8-merge trace on `"the cat sat on the mat the cat ran"`:
```
step 1: ('a', 't')      -> 'at'      freq=4   ← matches "first merge ('a','t') freq 4"
step 2: ('at', '</w>')  -> 'at</w>'  freq=4   ← `at</w>` appears (matches "after 6 merges")
step 3: ('t', 'h')      -> 'th'      freq=3   ← matches "('t','h') second-most-frequent freq 3"
step 4: ('th', 'e')     -> 'the'     freq=3
step 5: ('the', '</w>') -> 'the</w>' freq=3   ← `the</w>` appears (matches "after 6 merges")
step 6: ('c', 'at</w>') -> 'cat</w>' freq=2   ← `cat</w>` appears (matches "after 6 merges")
```

All three of `the</w>`, `cat</w>`, `at</w>` exist in the vocab after step 6. Issue #1 is resolved.

Final verdict: **Passes a code-along student**. 0 findings.
