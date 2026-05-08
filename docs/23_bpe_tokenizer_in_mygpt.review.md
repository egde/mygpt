# Student walkthrough report: Chapter 23 — BPETokenizer in mygpt

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/23_bpe_tokenizer_in_mygpt.md`
- Total sections walked: 5 of 5 executable subsections (§23.1 setup, §23.3 code edits, §23.4 train, §23.5 round-trip, §23.6 full-corpus comparison, §23.7 file-size check)
- Files modified: 1 — `src/mygpt/__init__.py` (add `import collections`, append `BPETokenizer` class)
- Files created: 4 — `experiments/42_bpe_train.py`, `experiments/43_bpe_round_trip.py`, `experiments/44_bpe_vs_char_corpus.py`, `shakespeare.bpe.json`
- Shell commands run: 7 (uv init, uv add, curl, three `uv run python`, one `wc -c`)
- Issues found: **2** (0 blockers, 1 mismatch, 0 clarity, 0 prerequisite gaps, 0 sequencing, 1 polish)
- Final verdict: **Will gently snag a careful student in §23.5** — Expected Output has two wrong rows in the per-token decode listing.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- Working directory: `/tmp/code-along-runs/ch23-review-135837/mygpt`

## Walkthrough

### Sections §23.1 – §23.3 (setup + code edits)
- §23.1: project initialised; tinyshakespeare.txt = 1,115,394 bytes.
- §23.3: `import collections` added between `"""mygpt — …"""` and `import math` (chapter says "before `import math`"); `BPETokenizer` class appended after `CharTokenizer.load`, before `save_checkpoint`. Both edits land cleanly.
- Issues: none.

### Section: §23.4 Training on Tiny Shakespeare
- Files written: `experiments/42_bpe_train.py`
- Commands: `uv run python experiments/42_bpe_train.py`
- Output:
  ```
  training took:    68.8s
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
- Expected output match: yes — exact character-for-character match (training time 68.8s vs chapter's quoted 69.0s — the chapter explicitly says wall-clock will vary).
- Issues raised here: none.

### Section: §23.5 Encoding and decoding round-trip
- Files written: `experiments/43_bpe_round_trip.py`
- Commands: `uv run python experiments/43_bpe_round_trip.py`
- Output (key part):
  ```
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
- Expected output match: **partial**. Top-of-section claims (61 chars, 25 ids, compression 2.44x, round-trip ok) match exactly. Per-token table rows 0–15 match. **Rows 16 and 19 differ** from the chapter:
  - Chapter row 16: `id= 70  symbol='f'`. Actual: `id= 44  symbol='f'`. The id is wrong (44 is the correct id for lowercase 'f' in this alphabet — the alphabet's first 13 entries are `\n` and punctuation, then 26 uppercase, then 'a' at 39, putting 'f' at 39+5=44).
  - Chapter row 19: `id= 70  symbol='f'`. Actual: `id= 70  symbol=', '`. The id 70 is correct for *some* token (`', '` — the comma-space merge, which is the 6th merge appended as `65+5=70`), but the chapter's symbol `'f'` is wrong; the actual token at position 19 of the sample is the comma-space after "further" in the sample text.
- Issues raised here: #1.

### Section: §23.6 Compression vs CharTokenizer (full corpus)
- Files written: `experiments/44_bpe_vs_char_corpus.py`
- Commands: `uv run python experiments/44_bpe_vs_char_corpus.py`
- Output:
  ```
  corpus chars:           1,115,394
  CharTokenizer ids:      1,115,394  (0.1s to encode)
  BPETokenizer  ids:      487,961  (19.4s to encode)
  BPE compression ratio:  2.29x
  Char vocab_size:        65
  BPE  vocab_size:        577
  ```
- Expected output match: yes — every count is exact. BPE encode time 19.4s vs chapter's quoted 19.6s; chapter says "timings vary".
- Issues raised here: none.

### Section: §23.7 Save / load round-trip
- Commands: `wc -c shakespeare.bpe.json`
- Output: `7167 shakespeare.bpe.json`
- Expected output match: chapter says "The file is ~6 KB on disk". Actual is 7167 bytes ≈ 7 KB. Off by ~1 KB.
- Issues raised here: #2.

## Issues

### Issue #1 — Mismatch — §23.5 per-token table has wrong rows 16 and 19
- Section: §23.5 — Expected Output block, the "First 20 ids decoded" listing.
- Where: rows 16 and 19 of the per-token table.
- What happened:
  - Chapter row 16: `16: id= 70  symbol='f'`. Actual run: `16: id= 44  symbol='f'`.
  - Chapter row 19: `19: id= 70  symbol='f'`. Actual run: `19: id= 70  symbol=', '`.
- Why a student would be stuck: not stuck — the script runs and the round-trip claim (`round-trip ok: True`) holds — but a careful student diff-checking the ids will find two mismatches. The pattern (chapter has `'f'` twice with the same id 70) suggests the chapter's expected output was hand-edited and a copy-paste error duplicated row 16 onto row 19, plus picked the wrong id for 'f' (70 is the id for the BPE merge `', '`, not the alphabet character `'f'` which is at id 44).
- Suggested fix: replace rows 16 and 19 with the captured-output values:
  ```
    16: id= 44  symbol='f'
    17: id=142  symbol='ur'
    18: id=177  symbol='ther'
    19: id= 70  symbol=', '
  ```

### Issue #2 — Polish — §23.7 quotes the file size as "~6 KB" but actual is ~7 KB
- Section: §23.7 — first prose paragraph after the `wc -c` example.
- Where: *"The file is ~6 KB on disk — three orders of magnitude smaller than a model checkpoint."*
- What happened: `wc -c shakespeare.bpe.json` returns 7167 bytes ≈ 7 KB.
- Why a student would be stuck: not stuck — the order-of-magnitude argument still holds — but the precise number is off by about 1 KB.
- Suggested fix: change "~6 KB" to "~7 KB", or weaken to "well under 10 KB".

## Confidence and caveats

I walked the chapter end-to-end including the slow training step (~69 s on M1 MPS) and the slow full-corpus encode (~19 s). All exact-numeric Expected Output claims hold except the two per-token rows in §23.5 and the file-size figure in §23.7. The compression ratios (2.44x for the sample, 2.29x for the corpus) and the vocabulary structure (65 alphabet + 512 merges = 577 tokens) are reproduced exactly. The first 10 merges are bit-identical to what the chapter claims.

§23.8 experiments and §23.9 exercises were not exhaustively re-run; their conceptual claims are consistent with the algorithm.

The chapter is two small fixes (a 4-line table correction and a 1-character KB number) away from clean.

---

## Review #2 — fixes verified

After the fixes:
- §23.5 rows 16 and 19 now read `id= 44  symbol='f'` and `id= 70  symbol=', '` — exact match to the captured run.
- §23.7 prose now says "~7 KB" — `wc -c shakespeare.bpe.json` returns 7167 bytes.

Both fixes were applied to text only; no code changed. Final verdict: **Passes a code-along student**. 0 findings.
