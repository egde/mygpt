---
title: 3. PyTorch in 20 minutes
nav_order: 4
parent: LLM Fundamentals
---

# Chapter 3 — PyTorch in 20 minutes: tensors, autograd, modules

In Chapter 2 we built the *project*. In this chapter we build the *vocabulary of operations* the rest of the tutorial will use. By the end you will know what a tensor is, what shape it has, what `@` and broadcasting mean, how PyTorch computes derivatives for you with **autograd**, and what an `nn.Module` is. You will also have extended the `mygpt` package with its first piece of real numerical code: a function that turns a list of tokens into a tensor of integer ids.

This is the only chapter that is mostly a primer. From Chapter 4 onward every chapter introduces ML-specific ideas (loss, gradients, attention, …) — but they all rest on the four tools introduced here.

There is some maths. None of it goes beyond first-year calculus and linear algebra.

---

## 3.1 Why PyTorch?

We need a library that can do three things well:

1. Hold large multi-dimensional arrays of numbers and do arithmetic on them very fast (ideally on a GPU).
2. Compute derivatives of any expression we write, automatically.
3. Let us bundle parameters and computation into reusable building blocks.

[PyTorch](https://pytorch.org) does all three with one consistent object — a **tensor** — sitting at the centre of everything. We use PyTorch because it is the most readable of the major options: tensor code looks like the maths, and the differentiation step is one method call (`.backward()`) rather than a separate compiler pass. Other libraries have other strengths; for a learning project, readability wins.

---

## 3.2 Setup

This chapter assumes you finished Chapter 2 and have a working `mygpt/` directory. From inside it, your `src/mygpt/__init__.py` should match the version we wrote in §2.5 — a `VOCAB` constant and a `main()` that prints it.

If you skipped Chapter 2 or your state has drifted, run these commands from a parent directory of your choice to recreate it from clean:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
```

(Chapter 2 created `experiments/` for us; if you went through `uv init` again above, you need to recreate it. Every later section that says "Save this to `experiments/...`" assumes the directory exists.)

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 2 ending state:

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time."""

VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
```

Now add PyTorch — and `numpy`, which `torch` uses internally and complains about if it is missing:

```bash
uv add torch numpy
```

**Expected output (the structural shape varies between first-run and re-run, and version numbers will drift; what matters is that all twelve package lines appear and there are no errors):**

```text
Using CPython 3.12.11
Creating virtual environment at: .venv
Resolved 31 packages in 8ms
   Building mygpt @ file:///.../mygpt
      Built mygpt @ file:///.../mygpt
Prepared 1 package in 2ms
Installed 12 packages in 250ms
 + filelock==3.x.x
 + fsspec==2026.x.x
 + jinja2==3.x.x
 + markupsafe==3.x.x
 + mpmath==1.x.x
 + mygpt==0.1.0 (from file:///.../mygpt)
 + networkx==3.x.x
 + numpy==2.x.x
 + setuptools==81.x.x
 + sympy==1.x.x
 + torch==2.x.x
 + typing-extensions==4.x.x
```

A few things you might see differ from the above and that are fine:

- The first two lines (`Using CPython ...`, `Creating virtual environment ...`) only appear the **first** time `uv` makes the venv. On a re-run they vanish.
- The numbers in `Resolved`, `Prepared`, and `Installed` lines reflect the resolution graph and the cache state — they will drift between machines.
- `mygpt==0.1.0 (from file:///.../mygpt)` appears because our package gets installed alongside its dependencies. On later `uv add` invocations it will instead show as an `Uninstalled 1 package` line followed by a re-install.

Confirm both are importable:

```bash
uv run python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

**Expected output (your patch versions will differ):**

```text
torch 2.11.0 | numpy 2.4.4
```

If you see a `Failed to initialize NumPy` warning here, `uv add numpy` did not run — re-run it.

---

## 3.3 Tensors

A **tensor** is PyTorch's name for a multi-dimensional array of numbers, all of the same type. A 0-dimensional tensor is a scalar. A 1-dimensional tensor is a vector. A 2-dimensional tensor is a matrix. Higher-dimensional tensors do not have well-known names; we just call them rank-$k$ tensors, where $k$ is the number of dimensions.

Three pieces of information identify a tensor:

- **`shape`** — the size along each dimension, written as a tuple. A vector of length 4 has shape `(4,)`; a $3 \times 4$ matrix has shape `(3, 4)`.
- **`dtype`** — the type of every element. The most common dtypes we will use are `torch.float32` (the default for floating-point) and `torch.long` (for integer ids).
- **`device`** — where the data physically lives in memory: `cpu` or `cuda` for an Nvidia GPU. Everything in this tutorial runs on `cpu` unless you opt in.

Save the following experiment so you can run the snippets in this chapter as we go:

**Save the following to** 📄 `experiments/02_tensors.py`:

```python
"""Experiment 02 — Tensors: shape, dtype, device, indexing, reshape."""

import torch


def main() -> None:
    # A 1-D tensor (a vector)
    v = torch.tensor([0.0, 1.0, 2.0, 3.0])
    print(repr(v))
    print(f"shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    print()

    # A 2-D tensor (a matrix)
    M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(repr(M))
    print(f"shape={tuple(M.shape)}")
    print()

    # arange and reshape — useful for setting up examples
    x = torch.arange(12)
    print("arange(12) =", x)
    print("reshape(3, 4) =")
    print(x.reshape(3, 4))
    print("reshape(2, 2, 3).shape =", tuple(x.reshape(2, 2, 3).shape))


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/02_tensors.py
```

**Expected output:**

```text
tensor([0., 1., 2., 3.])
shape=(4,), dtype=torch.float32, device=cpu

tensor([[1., 2.],
        [3., 4.]])
shape=(2, 2)

arange(12) = tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
reshape(3, 4) =
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
reshape(2, 2, 3).shape = (2, 2, 3)
```

Two things worth noticing:

- `torch.tensor([0.0, ...])` infers `dtype=float32` because the literals are floats. `torch.arange(12)` returns `int64` (a.k.a. `long`) because the literals are ints. Mixing dtypes in arithmetic is usually a bug; PyTorch will tell you when it happens.
- `reshape` is a *view*: it does not copy the data, only re-interprets it. The total number of elements (12 here) is the same before and after. Reshape's argument list is the new shape.

---

## 3.4 Tensor shapes and the `(B, T, C)` convention

Almost every tensor in this tutorial has at most three dimensions. We will name them with single letters, in this order:

- **`B`** — batch size: how many independent examples are processed in parallel.
- **`T`** — time steps: how many tokens are in each example. ("Time" is the language-modelling word for "position in the sequence".)
- **`C`** — channels: how many numbers describe each token (also called the *embedding dimension*, but we save that word for Chapter 5).

So a tensor of shape `(B, T, C) = (2, 4, 8)` describes a batch of 2 sequences, each of length 4, with each token represented by 8 numbers. Hold on to this picture; you will see it in nearly every chapter from Chapter 5 onward.

For now, a concrete one-batch example: our running sentence `"I love AI !"` has $T = 4$. Pretending each token is described by 1 number — its id — that becomes a tensor of shape `(1, 4)`:

$$
\begin{pmatrix} 0 & 1 & 2 & 3 \end{pmatrix}.
$$

We will turn that into a `(1, 4, C)` tensor in Chapter 5.

---

## 3.5 Operations: matrix multiplication and broadcasting

Two operations carry most of the arithmetic in a transformer: matrix multiplication and elementwise addition with broadcasting. You already know both from linear algebra; PyTorch just spells them in a particular way.

**Matrix multiplication** in PyTorch is the `@` operator (or `torch.matmul`):

$$
A B \;=\; \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} \;=\; \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}.
$$

**Broadcasting** is PyTorch's rule for combining tensors with different shapes. When you write `M + v`, PyTorch lines up the shapes from the right and "stretches" any size-1 (or missing) dimension to match. So a vector of shape `(3,)` added to a matrix of shape `(2, 3)` is added to *every row*:

$$
\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \;+\; \begin{pmatrix} 1 & 2 & 3 \end{pmatrix} \;=\; \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \end{pmatrix}.
$$

The rule is precise: dimensions are compatible if they are equal or one of them is 1. We will use it constantly to add a learned bias to a whole batch in one line.

**Save the following to** 📄 `experiments/03_ops.py`:

```python
"""Experiment 03 — Matrix multiplication, broadcasting, and a seeded random tensor."""

import torch


def main() -> None:
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print("A @ B =")
    print(A @ B)
    print()

    # Broadcasting: a (3,) vector added to a (2, 3) matrix
    M = torch.zeros(2, 3)
    v = torch.tensor([1.0, 2.0, 3.0])
    print("M + v =")
    print(M + v)
    print()

    # Seeded random tensor — reproducible across runs and machines
    torch.manual_seed(0)
    R = torch.randn(2, 3)
    print("torch.randn(2, 3) with seed 0:")
    print(R)


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/03_ops.py
```

**Expected output:**

```text
A @ B =
tensor([[19., 22.],
        [43., 50.]])

M + v =
tensor([[1., 2., 3.],
        [1., 2., 3.]])

torch.randn(2, 3) with seed 0:
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.5684, -1.0845, -1.3986]])
```

`torch.manual_seed(0)` is the PyTorch equivalent of fixing the random seed: any `randn`/`rand` call after it produces the same numbers on every run. Several later chapters depend on it for reproducible output — note the pattern.

---

## 3.6 Autograd: derivatives without effort

Suppose we have an expression $y = f(x)$ implemented with PyTorch operations, and we want $\frac{dy}{dx}$ at a specific value of $x$. Rather than computing the derivative on paper and writing a second function, PyTorch can compute it automatically — a feature called **autograd**.

The mechanism has three steps:

1. Mark the input tensor as one we want gradients with respect to: `x = torch.tensor(3.0, requires_grad=True)`.
2. Compute `y` from `x` using normal PyTorch operations. As you do, PyTorch silently records the *computation graph* — each operation and its operands.
3. Call `y.backward()`. PyTorch traverses the recorded graph from `y` backward to `x`, applying the chain rule, and stores the result in `x.grad`.

After step 3, `x.grad` is the value of $\frac{dy}{dx}$ at the chosen $x$. PyTorch did the differentiation symbolically through the graph; the only calculus *we* did was knowing how to interpret the answer.

A worked example you can verify by hand. Take

$$
f(x) = x^2 + 2x + 1, \qquad f'(x) = 2x + 2.
$$

At $x = 3$: $f(3) = 9 + 6 + 1 = 16$ and $f'(3) = 8$.

**Save the following to** 📄 `experiments/04_autograd.py`:

```python
"""Experiment 04 — Autograd on a quadratic, checked against the analytical derivative."""

import torch


def main() -> None:
    x = torch.tensor(3.0, requires_grad=True)
    y = x * x + 2 * x + 1   # f(x) = x^2 + 2x + 1
    y.backward()
    x_val = x.item()
    print(f"f({x_val:g})  = {y.item()}")
    print(f"f'({x_val:g}) = {x.grad.item()}  (analytical: 2*({x_val:g}) + 2 = {2*x_val + 2:g})")


if __name__ == "__main__":
    main()
```

The labels are written so they track whatever value of `x` we use — that way the printed lines stay consistent if we change `x` later (we will, in the experiments at the end of the chapter).

Run it:

```bash
uv run python experiments/04_autograd.py
```

**Expected output:**

```text
f(3)  = 16.0
f'(3) = 8.0  (analytical: 2*(3) + 2 = 8)
```

Two facts about autograd you should believe rather than memorise:

- Every PyTorch operation has a hand-written derivative inside the library. `backward()` chains them. You will not need to add to that list — every transformer operation we use is already covered.
- `.backward()` only works once per forward pass by default, and it accumulates gradients into `.grad` on every call. We will see both behaviours — and how to reset gradients — in Chapter 4.

---

## 3.7 `nn.Module`: bundling parameters with computation

A neural network is many tensors and many operations. We need a way to package them — to keep the *parameters* of one layer together with the *forward pass* that consumes them, without inventing our own bookkeeping.

PyTorch's answer is `nn.Module`: a base class you subclass to define a network or part of one. The contract is small.

- In `__init__`, create the **parameters** of your module by wrapping tensors in `nn.Parameter`. PyTorch tracks every `nn.Parameter` you assign to `self.<name>` and exposes them through `.parameters()`.
- In `forward`, write the computation. Call the module like a function (`m(x)`) — PyTorch routes that to `forward`.

**Save the following to** 📄 `experiments/05_module.py`:

```python
"""Experiment 05 — A minimal nn.Module: an affine map y = x W^T + b."""

import torch
import torch.nn as nn


class Affine(nn.Module):
    """y = x W^T + b — a single linear-then-shift layer.

    Parameters:
      weight: tensor of shape (out_features, in_features)
      bias:   tensor of shape (out_features,)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


def main() -> None:
    torch.manual_seed(0)

    m = Affine(in_features=3, out_features=2)
    print("Parameters in Affine(3, 2):")
    for name, p in m.named_parameters():
        print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
    print(f"Total: {sum(p.numel() for p in m.parameters())} parameters")

    x = torch.randn(4, 3)   # a "batch" of 4 vectors of length 3
    y = m(x)
    print(f"\nForward pass: input shape {tuple(x.shape)} -> output shape {tuple(y.shape)}")

    loss = y.sum()
    loss.backward()
    print(f"\nAfter backward, gradient shapes:")
    for name, p in m.named_parameters():
        print(f"  {name}.grad: shape={tuple(p.grad.shape)}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/05_module.py
```

**Expected output:**

```text
Parameters in Affine(3, 2):
  weight: shape=(2, 3), requires_grad=True
  bias: shape=(2,), requires_grad=True
Total: 8 parameters

Forward pass: input shape (4, 3) -> output shape (4, 2)

After backward, gradient shapes:
  weight.grad: shape=(2, 3)
  bias.grad: shape=(2,)
```

Three things to take away:

- **`nn.Parameter` is just a tensor with a flag**, but the flag matters: PyTorch's optimisers (Chapter 4) and saving/loading machinery only touch parameters of registered modules. If you store a learnable tensor as a plain attribute (`self.weight = torch.randn(...)`), the optimiser will *silently* ignore it.
- **The number of parameters is exactly $\text{out} \times \text{in} + \text{out}$**. For `Affine(3, 2)` that is $2 \cdot 3 + 2 = 8$, matching the printed total. We will count parameters this way for every layer we add.
- **`backward()` from a scalar populates `.grad` on every reachable parameter.** This is the engine of training. In Chapter 14 we will write `loss.backward()` and feel exactly nothing magical, because we already know what it is doing.

---

## 3.8 First `mygpt` extension: turning tokens into a tensor

The package needs a way to bridge from strings (`"I"`, `"love"`, …) to tensors (the only thing the rest of the model will see). The simplest possible bridge: a function that maps a list of vocabulary tokens to a 1-D tensor of integer ids.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

This file holds the package-level constants and small helpers used in every
chapter. After Chapter 3, it knows about the running-example vocabulary and
how to turn a list of tokens into a PyTorch tensor of ids.
"""

import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids.

    Each entry of `tokens` must appear in `VOCAB`; otherwise `VOCAB.index`
    raises a `ValueError` and the conversion fails.
    """
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
    sample = list(VOCAB)
    ids = to_ids(sample)
    print(f"to_ids({sample}) = {ids}")
```

Re-run the package entry-point to confirm the addition works:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4
to_ids(['I', 'love', 'AI', '!']) = tensor([0, 1, 2, 3])
```

Two design choices worth noting:

- **`dtype=torch.long`** (i.e. 64-bit integer) because token ids are integer indices. The PyTorch operations that take ids — like the embedding lookup we will meet in Chapter 5 — require `long` and refuse `float`.
- **`VOCAB.index(t)` is $\mathcal{O}(V)$**. With $V = 4$ that is fine; with $V = 50{,}000$ it would be terrible. We will replace this with a dictionary in Chapter 16, where we build the real tokenizer. Until then `VOCAB.index` is enough.

---

## 3.9 Experiment 06 — From token ids to one-hot vectors

We have a way to turn `["I", "love", "AI", "!"]` into the integer tensor `[0, 1, 2, 3]`. The next natural step is to turn each id into a *vector*, because vectors are what neural networks consume.

The simplest vector representation of a categorical id is **one-hot encoding**: an id $i$ becomes a length-$V$ vector that is $0$ everywhere except at position $i$, where it is $1$. For our four-token vocabulary the four one-hot vectors are the columns (and rows) of the $V \times V$ identity matrix:

$$
\text{one-hot}(0) = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \;\;
\text{one-hot}(1) = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, \;\;
\text{one-hot}(2) = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, \;\;
\text{one-hot}(3) = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}.
$$

PyTorch ships this as `torch.nn.functional.one_hot`. The convention `import torch.nn.functional as F` is so universal that we adopt it without comment in every later chapter.

**Save the following to** 📄 `experiments/06_one_hot.py`:

```python
"""Experiment 06 — From token ids (Chapter 3 mygpt addition) to one-hot vectors."""

import torch
import torch.nn.functional as F

from mygpt import VOCAB, to_ids


def main() -> None:
    ids = to_ids(["I", "love", "AI", "!"])
    print(f"ids:   {ids}")
    print(f"shape: {tuple(ids.shape)}")

    V = len(VOCAB)
    onehot = F.one_hot(ids, num_classes=V).float()
    print(f"\none-hot shape: {tuple(onehot.shape)}  # (T, V) = (4, {V})")
    print("one-hot rows (one row per token, V columns):")
    print(onehot)


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/06_one_hot.py
```

**Expected output:**

```text
ids:   tensor([0, 1, 2, 3])
shape: (4,)

one-hot shape: (4, 4)  # (T, V) = (4, 4)
one-hot rows (one row per token, V columns):
tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
```

The output is the $V \times V$ identity matrix — for our running example the four tokens happen to be presented in order, so each token's one-hot lands on the corresponding row. The shape `(T, V) = (4, 4)` will reappear in Chapter 5 as `(T, C)` once we replace one-hot with a learned **embedding**: a function that gives every token a denser, smaller, *learnable* vector instead of a one-hot.

---

## 3.10 Experiments

1. **Pick a different scalar for autograd.** Edit `experiments/04_autograd.py` to set `x = torch.tensor(-2.0, requires_grad=True)`. By hand, $f(-2) = 4 - 4 + 1 = 1$ and $f'(-2) = -2$. Run the script; the printed lines should now read `f(-2)  = 1.0` and `f'(-2) = -2.0  (analytical: 2*(-2) + 2 = -2)`. Notice that the labels follow `x` — this is exactly why we wrote the f-strings with `{x_val:g}` instead of hardcoding the number.
2. **Watch broadcasting fail.** In `experiments/03_ops.py`, change `v` to `torch.tensor([1.0, 2.0])` (length 2). Run it and read the resulting `RuntimeError`. PyTorch tells you which two shapes were not compatible. This is how almost every shape bug surfaces.
3. **Confirm `.grad` accumulates.** In `experiments/05_module.py`, after `loss.backward()`, run `loss = m(x).sum(); loss.backward()` *a second time*, then print `m.weight.grad`. The values should be twice what they were before. We will reset gradients with `m.zero_grad()` in Chapter 4 to prevent exactly this.

After each experiment, restore the file you changed before moving on.

---

## 3.11 Exercises

1. **Count the parameters in `Affine(8, 16)`.** Predict the number; then construct the module and verify with `sum(p.numel() for p in m.parameters())`. Generalise: how many parameters does `Affine(d_in, d_out)` have, in terms of `d_in` and `d_out`?
2. **What is the shape of `m.weight.T`?** If `m.weight.shape == (out, in)`, then `m.weight.T.shape == (in, out)`. Why is the transpose necessary in the line `x @ self.weight.T + self.bias`? (Hint: write the shapes of `x @ M` for various `M` and see which one yields a `(batch, out)` output.)
3. **Why one-hot is wasteful.** A one-hot vector of length $V = 50{,}000$ is 99.998% zeros. Multiplying a learned matrix of shape $(V, C)$ by such a vector produces a single row of that matrix. Argue why directly *indexing* the matrix would give the same result with $V$ times less work. (We will use this argument to justify embeddings in Chapter 5.)

---

## 3.12 What's next

Three new tools, one new helper:

- **tensor** — a multi-dimensional array, identified by shape, dtype, device.
- **autograd** — the engine that turns `.backward()` into a populated `.grad` on every parameter.
- **`nn.Module`** — the class that bundles parameters with a forward method.
- **`mygpt.to_ids`** — the bridge from a list of token strings to a 1-D `long` tensor of ids.

In Chapter 4 we use all three to teach a *single number* — yes, a number — to descend toward the bottom of a quadratic. We will introduce a **loss function**, the **optimiser** (`torch.optim.SGD`), and the training-loop pattern that every later chapter follows. After Chapter 4 we know how to make a model *learn*; from Chapter 5 onward we just keep adding capacity.

> **Looking ahead — what to remember from this chapter**
>
> 1. A tensor's shape, dtype, and device tell you almost everything you need to know about it.
> 2. `@` is matrix multiplication; broadcasting handles the rest of the elementwise arithmetic.
> 3. `requires_grad=True` + `.backward()` + `.grad` is the autograd contract.
> 4. `nn.Module` is the class; `nn.Parameter` is the flag that tells it which tensors are learnable.
> 5. Token ids enter PyTorch as a `dtype=torch.long` tensor — `mygpt.to_ids` produces exactly that.

On to [Chapter 4 — How machines learn: loss, gradients, gradient descent](04_how_machines_learn.md) *(coming soon)*.
