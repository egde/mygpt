# Student walkthrough report: Chapter 2 — Project setup with `uv` (review #2)

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/02_project_setup.md`
- Total sections walked: 9 of 10 (§2.10 is forward-looking prose, no executable content)
- Files created: 2 (`src/mygpt/__init__.py` rewrite, `experiments/01_hello_mygpt.py`)
- Shell commands run: 18
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student** — all four findings from review #1 are resolved, every shell command succeeded, every "Expected output" block matched, and no regressions were introduced.

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11 (auto-installed by `uv`)
- uv: 0.8.0 (0b2357294 2025-07-17)
- Working directory: `/tmp/code-along-runs/ch02-rev2-20260502-031530/`

## Walkthrough

### Section: §2.2 Install `uv`
- Files written: none
- Commands run: `uv --version`
- Output: `uv 0.8.0 (0b2357294 2025-07-17)`
- Expected output match: **yes** — the chapter now shows the parenthetical and explains it as the build commit hash and date. Exact match.
- Issues raised here: none

### Section: §2.3 Initialise the `mygpt` project
- Files written: none
- Commands run:
  ```bash
  uv init mygpt --package
  cd mygpt
  find . -type f -not -path '*/.venv/*' -not -path '*/.git/*' | sort
  cat pyproject.toml
  cat src/mygpt/__init__.py
  ```
- Output (file list):
  ```text
  ./.gitignore
  ./.python-version
  ./pyproject.toml
  ./README.md
  ./src/mygpt/__init__.py
  ```
- Expected output match: **yes** — file list, `pyproject.toml` shape, and the two-line `__init__.py` stub all match.
- Issues raised here: none

### Section: §2.4 Run the auto-generated package
- Files written: none
- Commands run: `uv run mygpt` (twice)
- Output (first run, abridged):
  ```text
  Using CPython 3.12.11
  Creating virtual environment at: .venv
     Building mygpt @ file:///private/tmp/.../mygpt
        Built mygpt @ file:///private/tmp/.../mygpt
  Installed 1 package in 3ms
  Hello from mygpt!
  ```
- Output (second run): `Hello from mygpt!`
- Expected output match: **yes** — first run matches modulo the build path the chapter explicitly says will differ; second run matches exactly.
- Issues raised here: none

### Section: §2.5 Make the package about *our* project
- Files written: `src/mygpt/__init__.py` (overwritten with the new VOCAB constant + main)
- Commands run: `uv run mygpt`
- Output:
  ```text
  Vocabulary: ('I', 'love', 'AI', '!')
  Vocabulary size V = 4
  ```
- Expected output match: **yes** — exact match.
- Issues raised here: none

### Section: §2.6 Your first experiment script
- Files written: `experiments/01_hello_mygpt.py`
- Commands run: `mkdir -p experiments`; `uv run python experiments/01_hello_mygpt.py`
- Output:
  ```text
  Vocabulary size V = 4
    id 0: 'I'
    id 1: 'love'
    id 2: 'AI'
    id 3: '!'
  ```
- Expected output match: **yes** — exact match.
- Issues raised here: none

### Section: §2.7 Experiments
- Files written: edits to `src/mygpt/__init__.py` (toggle to 5-token VOCAB and back; rename `main` ↔ `_main`).
- Commands run:
  ```bash
  uv run python experiments/01_hello_mygpt.py        # exp 1
  uv run mygpt                                        # exp 2 (with broken main)
  cat .venv/bin/mygpt                                 # exp 2 (inspect shim)
  cat uv.lock | head -30                              # exp 3
  ```
- Output (exp 2 — broken main):
  ```text
  Traceback (most recent call last):
    File "/private/tmp/.../mygpt/.venv/bin/mygpt", line 4, in <module>
      from mygpt import main
  ImportError: cannot import name 'main' from 'mygpt' (...). Did you mean: '_main'?
  ```
- Output (exp 2 — `cat .venv/bin/mygpt`):
  ```text
  #!/private/tmp/.../mygpt/.venv/bin/python
  # -*- coding: utf-8 -*-
  import sys
  from mygpt import main
  if __name__ == "__main__":
      ...
      sys.exit(main())
  ```
- Expected output match: **yes** for all three experiments.
  - Exp 1: V=5, `!` shifts to id 4 — confirmed.
  - Exp 2: chapter now says **Python `ImportError`** with exactly the text `cannot import name 'main' from 'mygpt'` and ending with `Did you mean: '_main'?` — all three substrings appear in the actual error verbatim. The shim file at `.venv/bin/mygpt` exists, contains `from mygpt import main`, and `cat` works as the chapter promises. Fix #2 from review #1 is resolved.
  - Exp 3: short TOML lockfile, as expected.
- Issues raised here: none

### Section: §2.8 Exercises
- Files written: edit to `pyproject.toml` (added a second `[project.scripts]` line).
- Commands run:
  ```bash
  uv run hello-mygpt                                            # ex 2
  ls .venv/lib/python*/site-packages/ | head                    # ex 3
  cat .venv/lib/python*/site-packages/mygpt.pth                 # ex 3 (chapter prompts this)
  wc -c < .venv/lib/python*/site-packages/mygpt.pth             # ex 3 (sanity-check the "~60 bytes" claim)
  ls .venv/lib/python*/site-packages/mygpt-0.1.0.dist-info/     # ex 3 (sanity-check dist-info contents)
  ```
- Output (ex 3 `ls site-packages`):
  ```text
  __pycache__
  _virtualenv.pth
  _virtualenv.py
  mygpt-0.1.0.dist-info
  mygpt.pth
  ```
- Output (ex 3 `cat mygpt.pth`): `/private/tmp/.../mygpt/src` (a single absolute path, ending in `/src`)
- Output (ex 3 size): 64 bytes — within the chapter's "about 60 bytes" claim (the exact size depends on how deep the project lives in the filesystem).
- Output (ex 3 `dist-info/`): `direct_url.json`, `entry_points.txt`, `INSTALLER`, `METADATA`, `RECORD`, `REQUESTED`, `uv_cache.json`, `WHEEL` — the chapter mentions `METADATA`, `RECORD`, `entry_points.txt` and adds "etc.", which is honest.
- Expected output match: **yes** for all three exercises.
  - Ex 2: `uv` rebuilds and re-installs because `pyproject.toml` changed, then runs `hello-mygpt`. Behaviour matches the chapter.
  - Ex 3: chapter's claims about `mygpt.pth` (file, ~60 bytes, contains a single absolute path to `src/`) and `mygpt-0.1.0.dist-info/` (folder, contains `METADATA`/`RECORD`/`entry_points.txt`/etc.) all hold. The two extra entries (`_virtualenv.*`, `__pycache__/`) are explicitly mentioned and dismissed as "the virtual environment itself, not our package". Fix #3 from review #1 is resolved.
- Issues raised here: none

### Section: §2.9 Putting the project under version control
- Files written: none
- Commands run:
  ```bash
  git status
  git add .gitignore pyproject.toml README.md src/ experiments/ uv.lock
  git commit -m "initial mygpt scaffold"
  ```
- Output (git status, before any add): `On branch main / No commits yet / Untracked files: ...` — confirms `uv init` had already run `git init`.
- Output (git commit):
  ```text
  [main (root-commit) 2d08b6c] initial mygpt scaffold
   6 files changed, 66 insertions(+)
   create mode 100644 .gitignore
   create mode 100644 README.md
   create mode 100644 experiments/01_hello_mygpt.py
   create mode 100644 pyproject.toml
   create mode 100644 src/mygpt/__init__.py
   create mode 100644 uv.lock
  ```
- Expected output match: **yes** — the chapter's claim that `uv init` already ran `git init` is correct (git accepted `git status` without prior `git init`), and the commit succeeded directly. The fallback `git config --global` instructions are present for any student whose machine lacks a global identity. Fix #4 from review #1 is resolved.
- Issues raised here: none

## Issues

None.

## Confidence and caveats

I walked every executable step in §§2.2–2.9 in a fresh `/tmp/code-along-runs/ch02-rev2-20260502-031530/` directory, with the system's pre-installed `uv 0.8.0`. Every "Expected output" block in the chapter — including the four sites where review #1 had flagged drift — now matches the actual machine output, modulo the differences the chapter itself caveats (build paths, the parenthetical on `uv --version`, build-time milliseconds, the file-system path inside `mygpt.pth`).

Specifically on the four prior findings:

1. **§2.2 `uv --version`** — chapter now shows `uv 0.8.0 (0b2357294 2025-07-17)` and explains the parenthetical. Resolved.
2. **§2.7 exp 2 (broken entry-point)** — chapter now describes a Python `ImportError`, gives the exact substring the message starts with and ends with, and invites the student to `cat .venv/bin/mygpt` to see the shim. Every claim verified against the live error output. Resolved.
3. **§2.8 ex 3 (site-packages layout)** — chapter now describes `mygpt.pth` as a *file* (~60 bytes, containing the absolute path to `src/`) and `mygpt-0.1.0.dist-info/` as a *folder*, and explicitly mentions the unrelated `_virtualenv.*` and `__pycache__/` entries. Live `ls`, `cat`, and `wc -c` all confirm. Resolved.
4. **§2.9 (git init)** — chapter now skips the redundant `git init` and notes that `uv init` already did it; provides a fallback for the `Author identity unknown` case. Live `git status` (no prior explicit init) followed by `git add` and `git commit` succeeded cleanly. Resolved.

No new issues surfaced. The chapter is ready to commit.
