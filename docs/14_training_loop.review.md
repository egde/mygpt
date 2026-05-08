# Student walkthrough report: Chapter 14 — Training loop (review #2)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/14_training_loop.md`
- Total sections walked: 7 of 9 (§14.1, §14.4, §14.8, §14.9 are prose)
- Files created: 4 — `src/mygpt/__init__.py`, `experiments/27_corpus_and_batch.py`, `experiments/28_train_gpt.py`, `experiments/29_eval_trained.py`
- Shell commands run: 7 (verified on review #1; the §14.6 fix is text-only and does not require re-running)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — the prior Polish finding (§14.6 whitespace alignment) is resolved; no regressions.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: previously `/tmp/code-along-runs/ch14-rev1-20260502-065017/mygpt`

## Walkthrough

### Section: §14.6 Verifying the trained model — Fix verified
- The chapter's expected output block now reads (with 6 spaces between `]` and `predicted=` for the longest prefix, matching the f-string's `:<28` padding plus 2 literal separator spaces):
  ```text
    prefix=[0]                           predicted='love'   expected='love'   OK
    prefix=[0, 1]                        predicted='AI'     expected='AI'     OK
    prefix=[0, 1, 2]                     predicted='!'      expected='!'      OK
    prefix=[0, 1, 2, 3]                  predicted='I'      expected='I'      OK
    prefix=[0, 1, 2, 3, 0]               predicted='love'   expected='love'   OK
    prefix=[0, 1, 2, 3, 0, 1]            predicted='AI'     expected='AI'     OK
    prefix=[0, 1, 2, 3, 0, 1, 2]         predicted='!'      expected='!'      OK
    prefix=[0, 1, 2, 3, 0, 1, 2, 3]      predicted='I'      expected='I'      OK
  ```
- This is the captured output from review #1's run, copied character-for-character into the chapter. The chapter and the script now agree exactly.

### Sections §14.2, §14.3, §14.5, §14.7
- Re-running was not needed: review #1 already confirmed every numerical claim (loss curve `5.27 → 0.0035`, 8/8 correct evals, AdamW vs SGD, lr=1.0 stuck at log(V), B=1/16 both converge). The fix between reviews was text-only — no code changed — so the script outputs are unchanged from review #1.

## Issues

None.

## Confidence and caveats

I walked the chapter twice. Review #1 found one Polish issue: §14.6's expected output block had 4 spaces between `]` and `predicted=` for the longest prefix, but the f-string produces 6 spaces (24-char prefix padded to 28 = 4 trailing spaces, plus 2 literal separator spaces). The chapter has been updated to match the captured output exactly. No regressions; everything else was already passing on review #1.

The chapter is ready to commit. After Ch.14 we have a *trained* `mygpt.GPT` — Ch.15 will sample text from it.
