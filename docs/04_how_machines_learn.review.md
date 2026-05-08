# Student walkthrough report: Chapter 4 — How machines learn (review #2)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/04_how_machines_learn.md`
- Total sections walked: 9 of 10 (§4.10 is forward-looking prose, no executable content)
- Files created: 4 — `src/mygpt/__init__.py` (rewritten twice), `experiments/07_gradient_descent.py`, `experiments/08_sgd.py`, `experiments/09_fit_line.py`
- Shell commands run: 14 (core path + four LR sweep variations + three §4.8 experiment variations + the §4.9 leaf-Variable check)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — all five findings from review #1 resolved, every numerical claim across §§4.2–4.9 held empirically, no regressions introduced.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch04-rev2-20260502-045420/mygpt`

## Walkthrough

### Section: §4.2 Setup
- Files written: `src/mygpt/__init__.py` (Ch.3 ending state)
- Commands run:
  ```bash
  uv init mygpt --package
  cd mygpt
  mkdir -p experiments
  uv add torch numpy
  ```
- Expected output match: **yes** — recapitulating Ch.3 setup runs cleanly.
- Issues raised here: none

### Section: §4.4 Gradient descent on $f(x)=(x-3)^2$
- Files written: `experiments/07_gradient_descent.py`
- Commands run: `uv run python experiments/07_gradient_descent.py`
- Expected output match: **yes** — exact match through step 10; final x = 3.000000 after 100 steps.
- Issues raised here: none

### Section: §4.5 The optimiser: torch.optim.SGD
- Files written: `experiments/08_sgd.py`
- Commands run: `uv run python experiments/08_sgd.py`
- Expected output match: **yes** — exact match. SGD trajectory is identical to §4.4 manual version.
- Issues raised here: none

### Section: §4.6 Fitting a line
- Files written: `experiments/09_fit_line.py` (with the new parameterised `true:` print)
- Commands run: `uv run python experiments/09_fit_line.py`
- Output:
  ```text
  step | w        | b        | loss
  -----+----------+----------+---------
     1 | 0.154490 | 0.106941 | 1.912861
     5 | 0.666696 | 0.359492 | 0.933316
    10 | 1.121395 | 0.477290 | 0.420929
    20 | 1.643009 | 0.528539 | 0.095822
    50 | 2.050989 | 0.534696 | 0.008301

  learned: w = 2.0510, b = 0.5347
  true:    w = 2.0000, b = 0.5000
  ```
- Expected output match: **yes** — exact match. The `true:` line is now produced via `f"true:    w = {true_w:.4f}, b = {true_b:.4f}"`, so it will reflect any change to `true_w`/`true_b` in §4.8 exp 3.
- Issues raised here: none

### Section: §4.7 Extending mygpt: set_seed
- Files written: `src/mygpt/__init__.py` (Ch.4 version with `set_seed`)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4
  to_ids(['I', 'love', 'AI', '!']) = tensor([0, 1, 2, 3])
  after set_seed(0), torch.randn(3) = tensor([ 1.5410, -0.2934, -2.1788])
  ```
- Expected output match: **yes** — exact match, including the leading space inside `tensor([ 1.5410, ...])`. **Fix #5 from review #1 resolved.**
- Issues raised here: none

### Section: §4.8 Experiments
- Files written: temporary edits to `experiments/07_gradient_descent.py`, `08_sgd.py`, `09_fit_line.py` (all restored after).
- Commands run (full LR sweep + three other experiments):
  ```bash
  # exp 1 — LR sweep
  uv run python experiments/08_sgd.py     # lr=0.1   (the original)
  uv run python experiments/08_sgd.py     # lr=0.99
  uv run python experiments/08_sgd.py     # lr=1.0
  uv run python experiments/08_sgd.py     # lr=1.5
  uv run python experiments/08_sgd.py     # lr=2.0
  # exp 2
  uv run python experiments/07_gradient_descent.py    # x = 7.0
  # exp 3
  uv run python experiments/09_fit_line.py            # true_w = -1.0
  # exp 4
  uv run python experiments/08_sgd.py     # both zero_grad commented
  ```
- Output (exp 1 — LR sweep, observed behaviour vs chapter's table):
  | lr     | chapter says                                                            | actually observed                                                                  |
  |--------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------|
  | 0.1    | converges smoothly to $x = 3$                                           | x → 3.000000 at step 100 ✓                                                         |
  | 0.99   | converges, oscillates with shrinking amplitude                          | x oscillates 0.0↔6.0 with shrinking gap; at step 100, x = 2.602 (still descending) ✓ |
  | 1.0    | perfectly oscillates between $x = 0$ and $x = 6$ forever, no progress   | x toggles 0 ↔ 6 every step, no progress ✓                                          |
  | 1.5    | diverges geometrically: 9, -9, 27, -45, 99, -189, …                     | exact sequence: 9, -9, 27, -45, 99, -189, 387, -765, 1539, -3069 ✓                 |
  | 2.0    | diverges even faster: 12, -24, 84, -240, …, hits `nan` quickly          | exact sequence: 12, -24, 84, -240, 732, -2184, …; final = `nan` ✓                  |
- Output (exp 2): grad values 8.0000, 6.4000, 5.1200, 4.0960, … and x descending toward 3 from above (6.2, 5.56, 5.05, 4.64, 4.31, …). Final x = 3.000000. Matches the chapter's prediction "grad values like 8.0000, 6.4000, 5.1200, …".
- Output (exp 3): with `true_w = -1.0`, the printed `true: w = -1.0000, b = 0.5000` correctly reflects the edit (Fix #4 verified). Learned w = -0.8837 at step 50 — exactly matching the chapter's "roughly -0.88 at step 50" prediction.
- Output (exp 4): with **both** `opt.zero_grad()` lines commented, the trajectory is 0, 0.6, 1.68, 3.02 (chapter says "x_3 = 3.02 already past the answer at step 3" ✓), 4.36, 5.43, 6.01, 5.99, 5.37, 4.28, 2.93 (overshoot then damped oscillation ✓). Final x = 2.313596 — matches "ending around 2.31 after 100 steps" exactly.
- Expected output match: **yes** for all four experiments. Every quantitative claim in the chapter — the LR sweep table, the gradient-sequence values for exp 2, the "roughly -0.88" prediction for exp 3, and the trajectory landmarks (x_3 = 3.02, final = 2.31) for exp 4 — held exactly.
- Issues raised here: none. **Fixes #1, #2, #3 from review #1 resolved.**

### Section: §4.9 Exercises
- Files written: temporary edit to `experiments/07_gradient_descent.py` (`with torch.no_grad():` block removed; restored after).
- Commands run: `uv run python experiments/07_gradient_descent.py`
- Output: `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.`
- Expected output match: **yes** — exact substring match for the chapter's hint ("`leaf Variable`" and "`in-place operation`").
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§4.2–4.9 inside a fresh `/tmp/code-along-runs/ch04-rev2-20260502-045420/` directory with `uv 0.8.0`, `torch 2.11.0`, `numpy 2.4.4`. Every "Expected output" block matched actual output, and every prose prediction (`watch X happen`, `you should see Y`) was verified against a real run.

Specifically on the five prior findings:

1. **§4.8 exp 1 LR sweep** — chapter now lists five LR values with the exact behaviour for each, plus the math derivation showing $\eta = 1$ is the oscillation boundary. All five rows verified empirically. Resolved.
2. **§4.8 exp 4 zero_grad demo** — chapter now describes overshoot + damped oscillation rather than "explosion", with specific landmarks (`x_3 = 3.02`, final = 2.31). Verified to match exactly. Resolved.
3. **§4.8 exp 4 ambiguity (which `zero_grad`)** — chapter now says "comment out **both** `opt.zero_grad()` lines". Verified the both-commented behaviour matches the new prose. Resolved.
4. **§4.8 exp 3 hardcoded label** — `09_fit_line.py` now uses `f"true:    w = {true_w:.4f}, b = {true_b:.4f}"`, and the chapter's prediction is now "near $-1$ — at step 50 with `lr=0.1` you will see roughly $-0.88$" (precise rather than misleadingly "near $-1.0$"). Both checked. Resolved.
5. **§4.7 tensor leading space** — chapter now shows `tensor([ 1.5410, -0.2934, -2.1788])`. Matches actual output exactly. Resolved.

No new issues surfaced. The chapter is ready to commit.
