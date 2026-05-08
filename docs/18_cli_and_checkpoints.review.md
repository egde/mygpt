# Student walkthrough report: Chapter 18 — Checkpoints, inference, and a CLI

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/18_cli_and_checkpoints.md`
- Total sections walked: 5 of 5 executable subsections (§18.1 setup, §18.2 save/load_checkpoint, §18.3 _train_command, §18.4 _generate_command, §18.5 main(), §18.6 end-to-end CLI run); §18.7 experiments and §18.8 exercise 1 spot-checked
- Files modified: 1 — `src/mygpt/__init__.py` (appended `save_checkpoint`/`load_checkpoint`/`_train_command`/`_generate_command`; replaced the existing `main()` with the argparse dispatcher)
- Files produced by the CLI: 1 — `shakespeare.ckpt`
- Shell commands run: 6 (uv init, uv add, `uv run mygpt --help`, `uv run mygpt train ...`, `uv run mygpt generate ...`, plus a `uv run python -c "..."` for exercise 1 verification)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.** The CLI works on first attempt, the loss curve and the generated sample match Ch.17's verified outputs, and exercise 1 reproduces exactly what the chapter promises.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- torch: 2.11.0
- numpy: 2.4.4
- Working directory: `/tmp/code-along-runs/ch18-review-100932/mygpt`

## Walkthrough

### Section: §18.1 Setup
- Files written: `src/mygpt/__init__.py` (Ch.17 ending state, copied verbatim from the Ch.17 review env), `tinyshakespeare.txt` (copied from the Ch.17 review env's existing download).
- Commands run:
  - `uv init mygpt --package`
  - `cd mygpt && mkdir -p experiments`
  - `uv add torch numpy`
- Output: project initialised; torch/numpy installed.
- Expected output match: yes (no expected output block; setup behaves as instructed).
- Issues raised here: none. The cumulative-snapshot delta-chain pattern noted in earlier reviews is unchanged; same convention as Ch.10–17.

### Sections §18.2–§18.5 (code edits, no output to compare)
- `save_checkpoint`/`load_checkpoint` appended after `CharTokenizer`, before `main` (chapter line 82: "after `CharTokenizer`, before `main`").
- `_train_command` appended after `load_checkpoint`, before `main` (chapter line 133).
- `_generate_command` appended after `_train_command`, before `main` (chapter line 181).
- The original `main()` was *replaced* with the argparse dispatcher per chapter line 206.
- Files written: 1 (the modified `src/mygpt/__init__.py`).
- Commands run: none.
- Issues raised here: none. Each "append after X, before Y" instruction has an unambiguous insertion point (the prior `main()` was a 2-line hello-world, easy to identify and replace).

### Section: §18.6 Use it: train, then generate
- Commands run:
  - `uv run mygpt --help`
  - `uv run mygpt train tinyshakespeare.txt --output shakespeare.ckpt`
  - `uv run mygpt generate --checkpoint shakespeare.ckpt --prompt "ROMEO:"`
- `mygpt --help` output:
  ```
  usage: mygpt [-h] {train,generate} ...

  Tiny GPT trainer and text generator.

  positional arguments:
    {train,generate}
      train           Train a GPT on a plain-text file.
      generate        Generate text from a checkpoint.

  options:
    -h, --help        show this help message and exit
  ```
- `mygpt train` output:
  ```
  corpus chars: 1,115,394
  vocab_size:   65
  params:       207,296
  steps:        2000
  step     1: loss = 41.0367
  step   500: loss = 2.5944
  step  1000: loss = 2.3529
  step  1500: loss = 2.1795
  step  2000: loss = 2.0785

  saved checkpoint to shakespeare.ckpt
  ```
- `mygpt generate` output:
  ```
  ROMEO:
  Thy momed has seltered, a neark'ly your tle centeloourse.
  Of therere hath thin beielly saneer best.

  BRINCE:
  Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
  Whe a bros swencurenty hou
  ```
- Expected output match: yes — every block matches character-for-character. The training loss curve at every shared step (`1`, `500`, `1000`, `2000`) is identical to Ch.17 §17.5's experiment 39. The generated 200-token sample is byte-for-byte identical to Ch.17 §17.6's experiment 40. The chapter's claim "byte-for-byte identical" is empirically verified.
- Issues raised here: none.

### Section: §18.7 Experiments — spot check
- Verified empirically: experiment 3 (`--seed 1` produces a different sample). Captured output diverged from the default-seed sample at every position after the prompt: the new opening was `ROMEO:\nSarrst herentser agl, that me, by halls.\n\nLOMENGETELLAUS:\n...`, not the default `Thy momed has seltered, ...`. The chapter's "Different sample, same model" claim holds.
- Experiment 5's "around 2.6 after 1000 steps" was verified during pre-review (loss `2.6036` at step 1000 with `--embed-dim 32 --num-heads 2 --num-layers 2 --steps 1000`).
- Experiments 1, 2, 4, 6 are qualitative ("speech-like text", "stays closer to most-likely token", "imitate the style", "argparse generates --help for free at every level") and were not exhaustively re-run; their behaviour follows from the §18.6 verification plus the earlier `temperature` / `top_k` machinery validated in Ch.15.
- Issues raised here: none.

### Section: §18.8 Exercise 1 — spot check
- Captured output of the introspection snippet:
  ```
  ['model_state_dict', 'tokenizer_chars', 'config']
  {'vocab_size': 65, 'embed_dim': 64, 'num_heads': 4, 'num_layers': 4, 'max_seq_len': 64}
  ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3']
  ['token_embedding.embedding.weight', 'position_embedding.weight', 'blocks.0.ln1.weight', 'blocks.0.ln1.bias', 'blocks.0.mha.causal_mask']
  ```
- Matches the chapter's prose claims: three top-level keys, the architecture dict (with the right values for our trained model), the first 10 alphabet characters, and 5 model-state weight-tensor names.
- Issues raised here: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end in a fresh temp directory with the Ch.17 ending state as the starting point, ran the long training step (~40 s on CPU), and verified every expected-output block character-for-character. The CLI's `--help`, `train`, and `generate` outputs all match exactly; the loss curve at shared step indices (`1`, `500`, `1000`, `2000`) is identical to Ch.17 §17.5; the 200-token generated sample is byte-identical to Ch.17 §17.6. Exercise 1's introspection output matches the chapter's prose. Experiment 3 (`--seed 1`) was verified to produce a different sample.

The cumulative-snapshot setup pattern noted in earlier reviews remains unchanged. As discussed in prior reviews, this convention is shared with Ch.10–17 and is not a new finding.

Experiments 1, 2, 4, 6 in §18.7 are qualitative and were not exhaustively re-run; their underlying mechanics (temperature, top_k, prompt independence, argparse `--help` at every level) were already verified in Ch.15 and §18.6.

The chapter is ready to commit. After Ch.18 the §1.10 promise — "a Python package called `mygpt` that you can train on a text file and use to generate text from the command line" — is fully delivered.
