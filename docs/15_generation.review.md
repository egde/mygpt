# Student walkthrough report: Chapter 15 — Generation (review #2)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/15_generation.md`
- Total sections walked: 7 of 11 executable subsections (§15.2 setup, §15.3, §15.5, §15.7 + §15.8, §15.9, plus targeted re-verification of §15.10 #1 and #3)
- Files created: 0 new on this pass (reused review #1's build at `/tmp/code-along-runs/ch15-rev1-070520/mygpt`, which already contains `trained_gpt.pt` and experiments 30–33)
- Shell commands run: 2 (re-run experiment 33 + ad-hoc REPL verification of §15.10 #1)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — all three issues from review #1 are resolved; no regressions introduced.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch15-rev1-070520/mygpt` (reused from review #1)

## Walkthrough

### Section: §15.9 What happens beyond the trained context — Fix verified
- The chapter's expected-output block now reads (with the trailing `'I'` present on the `expected:` line, and a consistent 5-space alignment between `got=...` and `expected=` across all 13 rows):
  ```
  output:   ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!', 'I', 'AI', '!', '!', 'love']
  expected: ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!', 'I', 'love', 'AI', '!', 'I']

    position 0: got='I'     expected='I'     OK
    position 1: got='love'  expected='love'  OK
    position 2: got='AI'    expected='AI'    OK
    position 3: got='!'     expected='!'     OK
    position 4: got='I'     expected='I'     OK
    position 5: got='love'  expected='love'  OK
    position 6: got='AI'    expected='AI'    OK
    position 7: got='!'     expected='!'     OK
    position 8: got='I'     expected='I'     OK
    position 9: got='AI'    expected='love'  DRIFT
    position 10: got='!'     expected='AI'    DRIFT
    position 11: got='!'     expected='!'     OK
    position 12: got='love'  expected='I'     DRIFT
  ```
- Re-running `uv run python experiments/33_beyond_context.py` produces a character-for-character identical output. Issues #1 and #2 from review #1 are resolved.

### Section: §15.10 Experiments — Fix verified
- The chapter's experiment 1 wording now reads (line 481):
  > "...at `temperature=2.0` (and even at `3.0`) the output is still the cycle. Bump it to `temperature=5.0` and you may see one or two tokens drift; only at extreme settings like `temperature=100.0` does the output become noticeably random."
- Empirical verification (`set_seed(0)` per call, model reloaded from checkpoint each time):
  ```
  temp=2.0:   ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!']    ← cycle
  temp=3.0:   ['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!']    ← cycle
  temp=5.0:   ['I', 'love', 'AI', '!', 'I', 'love', '!', 'I']     ← 2 drifts
  temp=100.0: ['I', '!', '!', 'love', 'I', '!', '!', '!']         ← random
  ```
  Each step of the new wording holds: cycle at 2.0 and 3.0, "one or two tokens drift" at 5.0, "noticeably random" at 100.0. Issue #3 from review #1 is resolved.

### Sections §15.2, §15.3, §15.5, §15.7+§15.8
- Re-running was not needed: review #1 already confirmed every numerical/text claim in these sections (the trained loss curve matches `_state_after_ch14.md`; greedy at §15.3 produces `['I', 'love', 'AI', '!', 'I', 'love', 'AI', '!']`; the §15.5 temperature table matches exactly; §15.8 produces the same cycle). The fixes between review #1 and #2 only touched §15.9 and §15.10 #1 — no other code or expected-output blocks were edited.

## Issues

None.

## Confidence and caveats

I walked review #2 by re-running the §15.9 experiment in the same temp directory used for review #1 (the trained checkpoint and earlier experiments are unchanged) and by spot-checking the §15.10 #1 temperature claim with an ad-hoc REPL invocation. The §15.9 output is now an exact match, including whitespace and the 13th element on the `expected:` line. The §15.10 #1 wording is now consistent with empirical behaviour at the four temperatures it mentions (2.0, 3.0, 5.0, 100.0).

The cumulative-snapshot setup pattern noted in review #1's caveats (the §15.2 setup tells the student to consume `_state_after_ch14.md`, but that snapshot only documents the chapter's *delta* and the student must chain back to `_state_after_ch08.md` for the full canonical baseline) remains unchanged. As discussed in review #1, this convention is shared with Ch.10–14 and was accepted in those reviews, so it is not a new finding.

The chapter is ready to commit. After Ch.15 we have a working `mygpt.generate` plus the trained-model demonstration that closes the §1.10 promise; Ch.16 will replace the four-token vocabulary with a character-level tokenizer.
