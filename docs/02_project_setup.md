---
title: 2. Project setup with uv
nav_order: 3
parent: LLM Fundamentals
---

# Chapter 2 — Project setup with `uv`

In Chapter 1 we agreed, in words and equations, what we are going to build: a Python package called `mygpt` that produces probability distributions over tokens, and a training loop that adjusts its parameters. In this chapter we create the package — empty for now — and learn the three commands you will type for the rest of the tutorial.

By the end of this chapter you will:

- have **`uv`** installed and verified on your machine,
- have a fresh `mygpt` package created via `uv init`,
- have replaced the auto-generated stub with our first real piece of code: the four-token vocabulary from Chapter 1,
- have run two programs end-to-end — one as a package entry-point, one as a standalone experiment script.

There is no maths in this chapter. It is pure setup. We get it out of the way once.

---

## 2.1 Why `uv`?

Real Python projects need three things working in concert: a *Python interpreter* of a known version, a *virtual environment* that isolates the project's libraries from the rest of your system, and a *dependency manager* that records which libraries the project uses and at which versions. The standard library tools (`venv`, `pip`) handle each of these separately, with their own files, commands, and edge cases.

[`uv`](https://github.com/astral-sh/uv) is a single tool that does all three. It is fast (written in Rust), it produces a reproducible lockfile, and — most relevantly for a tutorial — it lets you run a project's code with a single command, `uv run`, that takes care of installing the right Python version, creating the venv, and syncing dependencies behind the scenes. You will type `uv run` a lot in the next 17 chapters.

We assume nothing about whether you have used `uv` before. Three commands are enough for everything in this tutorial:

- `uv init` — create a new project.
- `uv add <package>` — add a dependency.
- `uv run <command>` — run a command inside the project's environment.

That's it. There are more commands; we'll meet them when we need them.

---

## 2.2 Install `uv`

On macOS and Linux, install `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or, if you already use [Homebrew](https://brew.sh):

```bash
brew install uv
```

After installing, **open a new terminal** so the updated `PATH` takes effect, then verify the installation:

```bash
uv --version
```

**Expected output (your version may be newer; the parenthetical is the build commit hash and date):**

```text
uv 0.8.0 (0b2357294 2025-07-17)
```

The exact version number does not matter for this tutorial; anything `0.4` or newer will work. The parenthetical after the version is the git commit and build date of the binary you installed — it changes from build to build and you can ignore it.

If `uv --version` errors with `command not found`, your shell's `PATH` does not include `~/.local/bin` (or wherever the installer placed `uv`). Re-open the terminal, or follow the installer's printed instructions.

---

## 2.3 Initialise the `mygpt` project

Pick a directory you keep code in (`~/dev`, `~/code`, whatever). From a shell, `cd` into it and run:

```bash
uv init mygpt --package
```

**Expected output (the path will differ on your machine):**

```text
Initialized project `mygpt` at `/Users/you/dev/mygpt`
```

This command creates a directory called `mygpt` and populates it with the standard layout for a Python package. The `--package` flag tells `uv` we want a *real* installable package (not a single-script project); concretely, it lays out the source under `src/mygpt/` and adds a `[project.scripts]` section to `pyproject.toml`.

`cd` into the new project. **Every command from this point onward is run from inside `mygpt/`** unless we say otherwise.

```bash
cd mygpt
```

Inspect what was just created:

```bash
find . -type f -not -path '*/.venv/*' -not -path '*/.git/*' | sort
```

**Expected output:**

```text
./.gitignore
./.python-version
./pyproject.toml
./README.md
./src/mygpt/__init__.py
```

Five files. Let's quickly look at the two that matter most.

### 2.3.1 `pyproject.toml`

```bash
cat pyproject.toml
```

**Expected output (the `authors` field will reflect your local git config):**

```toml
[project]
name = "mygpt"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "your-name", email = "you@example.com" }
]
requires-python = ">=3.12"
dependencies = []

[project.scripts]
mygpt = "mygpt:main"

[build-system]
requires = ["uv_build>=0.8.0,<0.9"]
build-backend = "uv_build"
```

Three blocks worth understanding:

- `[project]` — metadata. `name` is what we type to install and import. `requires-python = ">=3.12"` says we need Python 3.12 or later; if your system Python is older, `uv` will fetch a compatible interpreter for you. `dependencies = []` is the list we'll grow in Chapter 3.
- `[project.scripts]` — declares a console entry-point. `mygpt = "mygpt:main"` means *"when the user runs the command `mygpt`, call `mygpt.main()`"*. We'll use this immediately.
- `[build-system]` — how `uv` should build the package. We never edit this.

### 2.3.2 `src/mygpt/__init__.py`

```bash
cat src/mygpt/__init__.py
```

**Expected output:**

```python
def main() -> None:
    print("Hello from mygpt!")
```

Two-line stub. We'll replace it in §2.5.

---

## 2.4 Run the auto-generated package

`uv` already gave us a runnable program. Try it:

```bash
uv run mygpt
```

**Expected output (the first time you run this — the build/install lines may differ):**

```text
Using CPython 3.12.11
Creating virtual environment at: .venv
   Building mygpt @ file:///Users/you/dev/mygpt
      Built mygpt @ file:///Users/you/dev/mygpt
Installed 1 package in 4ms
Hello from mygpt!
```

What just happened? `uv run mygpt` saw that this project did not yet have a virtual environment, so it:

1. Picked a Python interpreter that satisfies `requires-python = ">=3.12"` (installing one from upstream if needed).
2. Created a virtual environment at `./.venv/`.
3. Built our package and installed it into that venv in **editable** mode (so future edits to the source take effect without reinstalling).
4. Looked up the script `mygpt` in `[project.scripts]`, found that it maps to `mygpt:main`, imported the package, and called `main()`.

If you run it again, it skips steps 1–3 and just calls `main()`:

```bash
uv run mygpt
```

**Expected output:**

```text
Hello from mygpt!
```

That is the basic edit-run loop. Edit a file under `src/mygpt/`, type `uv run mygpt`, see the change.

---

## 2.5 Make the package about *our* project

The auto-generated `Hello from mygpt!` is a placeholder. Let's replace it with something we actually need: the four-token vocabulary from Chapter 1, plus a `main` function that prints it.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

This file holds the package-level constants used in every chapter.
For now there is only one: the four tokens that form our running example.
"""

VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
```

Three things to notice:

- `VOCAB` is a *module-level constant*. Any code that does `from mygpt import VOCAB` will get this exact tuple. We will extend the package with more constants and classes in later chapters; they will all live alongside this one.
- We use a `tuple`, not a `list`, because the vocabulary is fixed for the lifetime of the program — tuples make that intent explicit and prevent accidental mutation.
- `main()` is still the function `pyproject.toml` points its `mygpt` script at. We kept the name the same so `uv run mygpt` still works without editing `pyproject.toml`.

Run it:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4
```

You have just shipped your first piece of `mygpt`.

---

## 2.6 Your first experiment script

The package itself should only contain code that the final user of `mygpt` would care about. Code that exists *only* to demonstrate or explore something belongs outside the package, in **experiments**.

We follow this convention throughout the tutorial. By the end of the book you will have one experiment per chapter under `experiments/`, all of which import from `mygpt` but none of which are part of `mygpt`.

Create the experiments directory and our first script:

```bash
mkdir -p experiments
```

**Save the following to** 📄 `experiments/01_hello_mygpt.py`:

```python
"""Experiment 01 — Hello from mygpt.

Confirms the package is installed and importable from a script outside
`src/mygpt/`. Prints the vocabulary along with each token's integer id.
"""

from mygpt import VOCAB


def main() -> None:
    print(f"Vocabulary size V = {len(VOCAB)}")
    for token_id, token in enumerate(VOCAB):
        print(f"  id {token_id}: {token!r}")


if __name__ == "__main__":
    main()
```

Run it with `uv run`. Note: we now invoke `python <script>` rather than the `mygpt` console script, because this is just a regular Python file, not a registered entry-point.

```bash
uv run python experiments/01_hello_mygpt.py
```

**Expected output:**

```text
Vocabulary size V = 4
  id 0: 'I'
  id 1: 'love'
  id 2: 'AI'
  id 3: '!'
```

This is the **token-id mapping** from §1.3 of Chapter 1, now actually running on your machine. The integers `0, 1, 2, 3` are the only form in which these tokens will enter any neural-network operation in the rest of this tutorial.

### Why does the import work?

`from mygpt import VOCAB` works because `uv run` already installed the package into the project's `.venv` in editable mode (back in §2.4). Editable mode means the venv contains a *pointer* to `src/mygpt/`, so any change you make to the source is picked up immediately by anything that imports `mygpt`. You will rely on this many times.

---

## 2.7 Experiments

Try these. None of them count as "doing it wrong" — they are how you build intuition.

1. **Add a fifth token.** Edit `src/mygpt/__init__.py` so `VOCAB` becomes `("I", "love", "AI", "GPT", "!")`. Re-run `uv run python experiments/01_hello_mygpt.py`. Notice that no reinstall was required: editable mode picks up the change immediately. Confirm the new size is 5 and the id of `"!"` shifts from 3 to 4.
2. **Break the entry-point on purpose.** Rename `main` to `_main` in `__init__.py`. Run `uv run mygpt`. You will see a Python `ImportError` saying `cannot import name 'main' from 'mygpt'`, ending with `Did you mean: '_main'?`. Concretely: `uv` had installed a tiny shim at `.venv/bin/mygpt` whose contents include `from mygpt import main` — renaming the function broke that one import line. `cat .venv/bin/mygpt` if you want to see the shim. Rename `_main` back to `main` before continuing. This is the feedback loop the script declaration is for: we wire up entry points once, then the file system tells us when we break them.
3. **Inspect the lockfile.** Run `cat uv.lock | head -30`. You will see a small TOML file `uv` has produced to record the exact versions of every dependency. Right now there are no dependencies, so it is short — but `uv.lock` is what makes the build reproducible across machines. We commit it to git in §2.9.

**After each experiment, restore your `VOCAB` and `main` to the version in §2.5 before moving on.** The next chapter assumes that exact starting state.

---

## 2.8 Exercises

1. **Read the script declaration.** Open `pyproject.toml` and locate `[project.scripts]`. The line is `mygpt = "mygpt:main"`. In one English sentence, write what each side of the `=` means.

2. **Add a second entry-point.** Add a line `hello-mygpt = "mygpt:main"` directly underneath the existing one. Save. Run `uv run hello-mygpt`. What happens? Why did `uv` need to rebuild the package? (Hint: `pyproject.toml` is part of the package's metadata, so changing it changes what gets installed.)

3. **Find the venv.** Run `ls .venv/lib/python*/site-packages/ | head`. Among the entries you will see two that belong to our package:
   - a **file** called `mygpt.pth` (about 60 bytes) — this is the editable-mode pointer. Cat it (`cat .venv/lib/python*/site-packages/mygpt.pth`) and you will see a single absolute path: the path to your `src/` directory. That one line is what makes `from mygpt import VOCAB` find your code.
   - a **folder** called `mygpt-0.1.0.dist-info/` — this holds the install metadata (`METADATA`, `RECORD`, `entry_points.txt`, etc.). It is what `uv` writes for the package to count as "installed", even though the source lives elsewhere.

   You will also see entries beginning with `_virtualenv` and a `__pycache__/` folder; those belong to the virtual environment itself, not to our package. The interesting fact is that nothing in `site-packages/` contains a *copy* of `src/mygpt/` — only a pointer to it. Why does that pointer-only layout make the edit-run loop you used in §2.5 work without any reinstall?

There are no "answers" for these — the goal is to build a feel for what `uv` actually arranged on your filesystem.

---

## 2.9 Putting the project under version control

Strictly optional, but recommended. Note that `uv init mygpt --package` already ran `git init` for you and wrote a `.gitignore` — there is no need to run `git init` yourself. (If you do, git will harmlessly reply `Reinitialized existing Git repository` and leave your repo unchanged.) Skip straight to staging and committing:

```bash
git add .gitignore pyproject.toml README.md src/ experiments/ uv.lock
git commit -m "initial mygpt scaffold"
```

If git refuses the commit with `Author identity unknown`, set your name and email globally first:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

Note we explicitly do **not** add `.venv/` (the auto-generated `.gitignore` excludes it). We *do* add `uv.lock` — it is the file that lets a collaborator reproduce your exact dependency versions later.

---

## 2.10 What's next

We have a package. We have an experiment script. We have a vocabulary of four tokens, mapped to ids `0, 1, 2, 3`. We have **zero** mathematics inside `mygpt` so far — the package is, in effect, four strings.

In Chapter 3 we add PyTorch. We learn what a **tensor** is, how PyTorch automatically computes derivatives via **autograd**, and what an `nn.Module` is. After Chapter 3, we will be able to talk about the model in terms of vectors, matrices, and gradients — and from Chapter 4 onward, every chapter will add real neural-network components to `mygpt`.

> **Looking ahead — what to remember from this chapter**
>
> 1. `uv init mygpt --package` creates the package skeleton; `cd mygpt` and stay there.
> 2. `uv run mygpt` runs the entry point declared in `pyproject.toml`.
> 3. `uv run python <script>` runs any script inside the project's environment.
> 4. The package source lives in `src/mygpt/`. Experiments live in `experiments/` and import from the package.
> 5. Editing files under `src/mygpt/` requires no re-install — editable mode already pointed the venv at the source.

On to [Chapter 3 — PyTorch in 20 minutes](03_pytorch_primer.md) *(coming soon)*.
