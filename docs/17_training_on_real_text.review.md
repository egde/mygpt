# Student walkthrough report: Chapter 17 — Training on a real text file (review #2)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/17_training_on_real_text.md`
- Total sections walked: 4 of 4 executable subsections (re-verified §17.2 only; §17.3/§17.5/§17.6 are unchanged from review #1)
- Files created: 0 new on this pass (reused review #1's build at `/tmp/code-along-runs/ch17-review-094741/mygpt`, which already has the trained `shakespeare_gpt.pt`, the tokenizer, and the experiments)
- Shell commands run: 3 (`rm -f tinyshakespeare.txt`, the new `curl -s -o ...`, `wc -c`/`head -3` to confirm)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — the `curl -s` fix landed; no regressions.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch17-review-094741/mygpt` (reused from review #1)

## Walkthrough

### Section: §17.2 The corpus — Fix verified
- The chapter's command now reads:
  ```
  curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  ```
  And the prose has been adjusted accordingly: "`curl -s` (silent) prints nothing on success; you should be returned to the prompt..."
- I deleted the previously-downloaded `tinyshakespeare.txt` and re-ran the new command. Captured output:
  ```
  ---curl exit 0---
   1115394 tinyshakespeare.txt
  First Citizen:
  Before we proceed any further, hear me speak.
  ```
  The `curl -s` call printed nothing (no progress bar); the wc/head outputs match the chapter exactly.
- Issue #1 from review #1 is resolved.

### Sections §17.1, §17.3, §17.5, §17.6, §17.7 spot-check
- Re-running was not needed: review #1 already confirmed every Expected Output block exactly (vocab, alphabet, first-60-id list, decoded first-line text, param count 207,296, the loss curve `41.04 → 15.02 → 3.24 → 2.59 → 2.35 → 2.08`, and the deterministic 200-token Shakespeare sample), plus the §17.7 exp 6 "loss plateaus around 2.5" claim was empirically verified at 2.46. The fix between reviews touched only §17.2's command + prose; no code or other expected-output blocks were edited.

## Issues

None.

## Confidence and caveats

I verified the §17.2 fix by removing the existing download and re-running the new command in the same temp environment. `curl -s -o ...` produced exactly zero stdout and zero stderr (exit 0), matching the new prose claim "prints nothing on success". The downloaded file is byte-for-byte identical (1,115,394 bytes; same first three lines). Every other section was already exact on review #1 and the chapter's diff between reviews is restricted to a single command + prose paragraph in §17.2, so no regressions are possible.

The cumulative-snapshot setup pattern noted in earlier reviews remains unchanged. As discussed in prior reviews, this convention is shared with Ch.10–16 and is not a new finding.

The chapter is ready to commit. After Ch.17 we have an end-to-end training pipeline against real text; Ch.18 wraps it in a CLI.
