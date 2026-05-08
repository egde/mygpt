# Student walkthrough report: Chapter 16 — A reusable character tokenizer (review #2)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/16_character_tokenizer.md`
- Total sections walked: 5 of 5 executable subsections (re-verified §16.6 only; §16.2/§16.3/§16.4-§16.5/§16.7 are unchanged from review #1)
- Files created: 0 new on this pass (reused review #1's build at `/tmp/code-along-runs/ch16-review-091825/mygpt`, which already has the appended `CharTokenizer`, all four experiments, and `tokenizer.json`)
- Shell commands run: 1 (`wc -c tokenizer.json` to confirm the byte-count claim)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — the byte-count fix landed; no regressions.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch16-review-091825/mygpt` (reused from review #1)

## Walkthrough

### Section: §16.6 Saving and loading — Fix verified
- The chapter's prose paragraph now reads (line 323):
  > "A tokenizer file for our running example is 51 bytes (the JSON above, no trailing newline). A real-text tokenizer for the Tiny Shakespeare corpus we will use in Chapter 17 has ~65 distinct characters and the JSON file is around 350 bytes."
- `wc -c tokenizer.json` reports **51 bytes**, matching the new claim exactly.
- The "around 350 bytes" Tiny Shakespeare estimate is consistent with the back-of-envelope `11 + 65 × 5 + 2 ≈ 338` calculation noted in review #1, weakened to "around 350" so the pre-Ch.17 estimate has slack.
- Issue #1 from review #1 is resolved.

### Sections §16.2, §16.3, §16.4-§16.5, §16.7, §16.8
- Re-running was not needed: review #1 already confirmed every Expected Output block exactly, and the §16.8 experiment claims (29-char vocab, `KeyError('H')`, deterministic sort). The fix between reviews touched only §16.6's prose paragraph; no code or other expected-output blocks were edited.

## Issues

None.

## Confidence and caveats

I verified the §16.6 byte-count fix by re-reading the updated prose and re-running `wc -c` on the same `tokenizer.json` produced by experiment 36 (51 bytes — matches). Every other section was already exact on review #1 and the chapter's diff between reviews is restricted to one prose sentence in §16.6, so no regressions are possible.

The cumulative-snapshot setup pattern noted in earlier reviews (the §16.2 setup tells the student to consume `_state_after_ch15.md`, but that snapshot only documents the chapter's *delta* and the student must chain back to `_state_after_ch08.md` for the full canonical baseline) remains unchanged. As discussed in prior reviews, this convention is shared with Ch.10–15 and is not a new finding.

The chapter is ready to commit. After Ch.16 we have a reusable `CharTokenizer`; Ch.17 will use it on a real Tiny Shakespeare corpus.
