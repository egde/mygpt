---
title: 4. How machines learn
nav_order: 5
parent: Part I — LLM Fundamentals
---

# Chapter 4 — How machines learn: loss, gradients, gradient descent

Chapter 3 gave us tensors, autograd, and `nn.Module`. We can now compute and we can differentiate. What we still cannot do is **learn**: take some parameters that are "bad", and adjust them until they are "good".

This chapter closes that gap. We introduce three pieces, in order:

1. **Loss** — a single number that quantifies how wrong the current parameters are.
2. **Gradient descent** — an algorithm that nudges the parameters in the direction that decreases loss.
3. **The optimiser** — PyTorch's `torch.optim` API that wraps the bookkeeping of gradient descent into four lines you will write a thousand times.

By the end you will have trained two things by hand: a single scalar that learns to equal `3`, and a 1-D linear model that learns to fit a line. From Chapter 5 onward we keep this same loop and just plug bigger models into it.

---

## 4.1 The big picture

A "model" is just a function $f_\theta(x)$ that takes some input $x$ and produces a prediction. The function depends on parameters $\theta$ — the numbers PyTorch will let us adjust. **Learning** means: given some examples of how $f$ is *supposed* to behave, find values of $\theta$ that make $f_\theta$ behave that way.

To do that we need to make "supposed to behave" measurable. We invent a **loss function** $\mathcal{L}(\theta)$ — a non-negative scalar that is $0$ when the model is perfect and grows as the model gets worse. Once we have $\mathcal{L}$, the rule writes itself:

$$
\theta^\star \;=\; \underset{\theta}{\arg\min}\; \mathcal{L}(\theta).
$$

In words: the best parameters are the ones that minimise the loss. The whole rest of this tutorial is about (a) choosing $\mathcal{L}$ for the language-modelling task, and (b) running an algorithm that drives $\mathcal{L}$ down. The algorithm is the same one regardless of model size, so we learn it now, on parameters small enough to print on the screen.

---

## 4.2 Setup

This chapter assumes you finished Chapter 3 — `mygpt/` exists, `torch` and `numpy` are installed, and `src/mygpt/__init__.py` has the `VOCAB` constant and `to_ids` function from §3.8. All commands below run from inside the `mygpt/` project root.

If you skipped Chapter 3, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Then overwrite **`src/mygpt/__init__.py`** with the Chapter 3 ending state:

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time."""

import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")


def to_ids(tokens: list[str]) -> torch.Tensor:
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
    sample = list(VOCAB)
    ids = to_ids(sample)
    print(f"to_ids({sample}) = {ids}")
```

You are ready.

---

## 4.3 Loss: a single number that says how wrong we are

The simplest non-trivial loss is the **squared error** of one prediction. Suppose the model says $\hat{y}$ and the truth is $y$. Define

$$
\mathcal{L}(\hat{y}, y) \;=\; (\hat{y} - y)^2.
$$

Three nice properties:

- It is non-negative: $(\hat{y} - y)^2 \geq 0$.
- It is zero exactly when $\hat{y} = y$.
- It is differentiable everywhere — perfect for autograd.

When we have $n$ predictions $\hat{y}_1, \ldots, \hat{y}_n$ and matching targets $y_1, \ldots, y_n$, we average the squared errors. This is **mean squared error** (MSE):

$$
\mathcal{L}_\text{MSE}(\hat{\mathbf{y}}, \mathbf{y}) \;=\; \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2.
$$

We use MSE for the toy regression problem in this chapter. From Chapter 13 onward our actual loss for language modelling is **cross-entropy**, which we introduce when we need it; the abstract pattern is identical.

---

## 4.4 Gradient descent: minimising $f(x) = (x - 3)^2$

Forget models for a moment. Take the simplest possible loss landscape — a parabola with its minimum at $x = 3$:

$$
f(x) = (x - 3)^2.
$$

We know by inspection that $f$ is minimised at $x = 3$, where $f(3) = 0$. Pretend we don't. We have only:

- the ability to evaluate $f$ at any $x$,
- the ability to compute the derivative $f'(x) = 2(x - 3)$ via autograd.

**Gradient descent** is the iterative rule:

$$
x_{t+1} \;=\; x_t \;-\; \eta \cdot f'(x_t),
$$

where $\eta > 0$ is the **learning rate** — a small step size we pick. The minus sign is the whole idea: $f'(x_t)$ points in the direction of *increasing* $f$, so we move the *opposite* way.

A worked first step. Start at $x_0 = 0$, take $\eta = 0.1$:

$$
f'(0) = 2 \cdot (0 - 3) = -6, \qquad x_1 = 0 - 0.1 \cdot (-6) = 0.6.
$$

Each step we get closer to $3$. The gradient shrinks toward zero, so the steps get shorter and we approach the minimum smoothly. Let's verify with autograd.

**Save the following to** 📄 `experiments/07_gradient_descent.py`:

```python
"""Experiment 07 — Gradient descent on f(x) = (x - 3)^2 by hand.

Starts from x = 0, learning rate eta = 0.1. Prints the trajectory of x
for the first 10 steps, then runs to step 100 and prints the final x.
"""

import torch


def main() -> None:
    x = torch.tensor(0.0, requires_grad=True)
    eta = 0.1

    print(f"step | x        | f(x)     | grad")
    print(f"-----+----------+----------+--------")
    print(f"   0 | {x.item():.6f} |          |")

    for step in range(1, 11):
        if x.grad is not None:
            x.grad.zero_()                       # reset gradient
        loss = (x - 3) ** 2                      # forward
        loss.backward()                          # compute grad
        g = x.grad.item()                        # save for printing
        with torch.no_grad():                    # update without tracking
            x.sub_(eta * x.grad)                 # x ← x − η·grad
        print(f"  {step:2d} | {x.item():.6f} | {loss.item():.6f} | {g:.4f}")

    # Continue silently to step 100
    for step in range(11, 101):
        if x.grad is not None:
            x.grad.zero_()
        loss = (x - 3) ** 2
        loss.backward()
        with torch.no_grad():
            x.sub_(eta * x.grad)

    print(f"\nfinal x after 100 steps: {x.item():.6f}")
    print(f"target:                   3.000000")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/07_gradient_descent.py
```

**Expected output:**

```text
step | x        | f(x)     | grad
-----+----------+----------+--------
   0 | 0.000000 |          |
   1 | 0.600000 | 9.000000 | -6.0000
   2 | 1.080000 | 5.760000 | -4.8000
   3 | 1.464000 | 3.686400 | -3.8400
   4 | 1.771200 | 2.359296 | -3.0720
   5 | 2.016960 | 1.509950 | -2.4576
   6 | 2.213568 | 0.966368 | -1.9661
   7 | 2.370854 | 0.618475 | -1.5729
   8 | 2.496684 | 0.395824 | -1.2583
   9 | 2.597347 | 0.253327 | -1.0066
  10 | 2.677877 | 0.162130 | -0.8053

final x after 100 steps: 3.000000
target:                   3.000000
```

Three small but important points about the script:

- **`x.grad.zero_()` clears the previous gradient before each backward.** Recall from Chapter 3: `.backward()` *accumulates* into `.grad`, so without zeroing, step 2 would see the sum of step 1 and step 2 gradients. The leading `if x.grad is not None` is needed only for step 1, when `.grad` has not been allocated yet.
- **`with torch.no_grad():` around the update tells autograd not to track the parameter update itself.** We do not want the assignment $x \leftarrow x - \eta \cdot \mathrm{grad}$ to become part of the graph for the *next* backward. (Skip this and you get a different but equally informative error — feel free to delete the `with` block as an experiment.)
- **`x.sub_(eta * x.grad)` mutates `x` in place.** The trailing underscore is PyTorch's convention for *in-place* operations. Equivalent to `x.data -= eta * x.grad`.

---

## 4.5 The optimiser: `torch.optim.SGD`

Every line of bookkeeping in §4.4 — the `zero_()`, the `no_grad()`, the in-place update — is the same for every parameter of every model we will ever train. PyTorch packages all of it into the **`torch.optim`** module.

The simplest optimiser is **stochastic gradient descent** (SGD). For our purposes it is just gradient descent — the "stochastic" part will become relevant in Chapter 14 when we feed batches of data. Construct one with a list of parameters and a learning rate:

```python
opt = torch.optim.SGD([x], lr=0.1)
```

Then the training loop becomes a four-line ritual:

```python
opt.zero_grad()      # 1. clear previous gradients
loss = compute(...)  # 2. forward pass: compute loss
loss.backward()      # 3. populate .grad for every parameter
opt.step()           # 4. apply x ← x − lr · grad to every parameter
```

That four-line shape — `zero_grad`, *forward*, `backward`, `step` — is the canonical training loop. You will see it in every chapter from Chapter 14 onward, exactly as written.

**Save the following to** 📄 `experiments/08_sgd.py`:

```python
"""Experiment 08 — The same minimisation as exp 07, using torch.optim.SGD.

Verifies that the four-line training loop produces identical x values
to the manual version in exp 07.
"""

import torch


def main() -> None:
    x = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.SGD([x], lr=0.1)

    print(f"step | x        | f(x)")
    print(f"-----+----------+---------")
    print(f"   0 | {x.item():.6f} |")

    for step in range(1, 11):
        opt.zero_grad()
        loss = (x - 3) ** 2
        loss.backward()
        opt.step()
        print(f"  {step:2d} | {x.item():.6f} | {loss.item():.6f}")

    for step in range(11, 101):
        opt.zero_grad()
        loss = (x - 3) ** 2
        loss.backward()
        opt.step()

    print(f"\nfinal x after 100 steps: {x.item():.6f}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python experiments/08_sgd.py
```

**Expected output:**

```text
step | x        | f(x)
-----+----------+---------
   0 | 0.000000 |
   1 | 0.600000 | 9.000000
   2 | 1.080000 | 5.760000
   3 | 1.464000 | 3.686400
   4 | 1.771200 | 2.359296
   5 | 2.016960 | 1.509950
   6 | 2.213568 | 0.966368
   7 | 2.370854 | 0.618475
   8 | 2.496684 | 0.395824
   9 | 2.597347 | 0.253327
  10 | 2.677877 | 0.162130

final x after 100 steps: 3.000000
```

The trajectory is **identical** to experiment 07, byte for byte. SGD is not doing anything you would not have written by hand — it is just keeping the bookkeeping out of your training loop.

---

## 4.6 Fitting a line: `y = w x + b`

Now we have the full toolkit. Let's apply it to a problem with more than one parameter. We will fit a linear model

$$
\hat{y}_i \;=\; w \cdot x_i + b
$$

to a small synthetic dataset. The "true" relationship is $y = 2x + 0.5$ plus a bit of noise; the model has to discover this.

**Save the following to** 📄 `experiments/09_fit_line.py`:

```python
"""Experiment 09 — Fit a linear model y = w x + b to noisy data.

Generates 20 points from y = 2x + 0.5 + noise, then trains w and b to
minimise mean squared error using torch.optim.SGD. Reports the loss at
selected steps and the learned (w, b) at the end.
"""

import torch


def main() -> None:
    torch.manual_seed(0)

    # Synthetic data: 20 points in [-1, 1], with true y = 2x + 0.5 + N(0, 0.1)
    n = 20
    x_data = torch.linspace(-1.0, 1.0, n)
    true_w, true_b = 2.0, 0.5
    y_data = true_w * x_data + true_b + 0.1 * torch.randn(n)

    # Parameters to learn — start at zero
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=0.1)

    print("step | w        | b        | loss")
    print("-----+----------+----------+---------")
    for step in range(1, 51):
        opt.zero_grad()
        y_pred = w * x_data + b                 # forward
        loss = ((y_pred - y_data) ** 2).mean()  # MSE
        loss.backward()                         # populate w.grad, b.grad
        opt.step()                              # w ← w − lr·grad, same for b
        if step in (1, 5, 10, 20, 50):
            print(f"  {step:2d} | {w.item():.6f} | {b.item():.6f} | {loss.item():.6f}")

    print(f"\nlearned: w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"true:    w = {true_w:.4f}, b = {true_b:.4f}")


if __name__ == "__main__":
    main()
```

The `true:` line interpolates `true_w` and `true_b` so the comparison stays honest if you change the targets in the experiments at the end of the chapter.

Run it:

```bash
uv run python experiments/09_fit_line.py
```

**Expected output:**

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

What to read out of this output:

- **The loss decreases monotonically.** From $1.91$ at step 1 to $0.0083$ at step 50 — about 230× lower. That is what "learning" looks like in numbers.
- **The learned parameters approach the true parameters** but do not reach them exactly — and they should not, because the data is noisy. The remaining gap (≈ $0.05$ for $w$, $0.03$ for $b$) is consistent with the noise level.
- **The shape of the training loop is identical to experiment 08.** Same four lines: `zero_grad`, forward (compute predictions and loss), `backward`, `step`. From Chapter 14 onward we drop the manual print statements and replace `(w * x + b)` with a transformer, but the loop body is unchanged.

---

## 4.7 Extending `mygpt`: `set_seed`

Reproducibility matters. The line `torch.manual_seed(0)` appeared in §4.6 and will appear in nearly every later chapter. We add it to the package as a one-liner so we can write `mygpt.set_seed(0)` instead — and so all later chapters use a single canonical entry point.

**Replace the contents of** 📄 `src/mygpt/__init__.py` **with:**

```python
"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 4 the package knows about the running-example vocabulary,
how to turn tokens into id tensors (Chapter 3), and how to seed PyTorch's
random number generator for reproducibility (this chapter).
"""

import torch


VOCAB: tuple[str, ...] = ("I", "love", "AI", "!")
"""The four tokens used as the running example throughout this tutorial."""


def to_ids(tokens: list[str]) -> torch.Tensor:
    """Convert a list of vocabulary tokens to a 1-D tensor of integer ids."""
    return torch.tensor([VOCAB.index(t) for t in tokens], dtype=torch.long)


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's CPU random number generator.

    Call this at the start of any experiment that uses `torch.randn`,
    `torch.rand`, or randomly initialised modules — it makes the run
    reproducible across machines.
    """
    torch.manual_seed(seed)


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
    sample = list(VOCAB)
    ids = to_ids(sample)
    print(f"to_ids({sample}) = {ids}")
    set_seed(0)
    print(f"after set_seed(0), torch.randn(3) = {torch.randn(3)}")
```

Run it:

```bash
uv run mygpt
```

**Expected output:**

```text
Vocabulary: ('I', 'love', 'AI', '!')
Vocabulary size V = 4
to_ids(['I', 'love', 'AI', '!']) = tensor([0, 1, 2, 3])
after set_seed(0), torch.randn(3) = tensor([ 1.5410, -0.2934, -2.1788])
```

The numbers printed at the end are the same first three values you saw from `torch.randn(2, 3)` in Chapter 3 — `set_seed(0)` is the same as `torch.manual_seed(0)`. From this chapter on, we use the wrapper.

---

## 4.8 Experiments

1. **Sweep the learning rate until things break.** In `experiments/08_sgd.py`, change `lr=0.1` to each of the values below in turn, run, and read off what happens in the first 10 steps:

   | learning rate | what happens                                                              |
   |---------------|---------------------------------------------------------------------------|
   | `lr=0.1`      | converges smoothly to $x = 3$ (the original)                              |
   | `lr=0.99`     | converges to $x = 3$, but oscillates around it with shrinking amplitude   |
   | `lr=1.0`      | **perfectly oscillates** between $x = 0$ and $x = 6$ forever, no progress |
   | `lr=1.5`      | diverges geometrically: $9, -9, 27, -45, 99, -189, \ldots$               |
   | `lr=2.0`      | diverges even faster: $12, -24, 84, -240, \ldots$, hits `nan` quickly    |

   The exact threshold is derivable: for $f(x) = (x - x^\star)^2$, the gradient-descent step is $x_{t+1} = (1 - 2\eta) x_t + 2\eta x^\star$, so the iteration is stable iff $|1 - 2\eta| < 1$, i.e. $\eta < 1$. Convergence is fastest at $\eta = 0.5$, oscillates at the boundary $\eta = 1$, and diverges beyond. Gradient descent is correct in principle and fragile in practice; this is the single most common reason a model "fails to learn".

2. **Start from a different `x`.** In `experiments/07_gradient_descent.py`, set `x = torch.tensor(7.0, requires_grad=True)`. By symmetry the trajectory should now approach 3 from the right, with the *signs* of every gradient flipped (positive instead of negative). Verify by running — you should see `grad` values like `8.0000, 6.4000, 5.1200, ...` and `x` values descending toward 3 from above.

3. **Fit a different line.** In `experiments/09_fit_line.py`, change `true_w = 2.0` to `true_w = -1.0`. Re-run; the learned `w` should now converge in the right direction (near $-1$ — at step 50 with `lr=0.1` you will see roughly $-0.88$, since the gradient magnitude shrinks as $w$ approaches the target and convergence is slower for the smaller-magnitude target). The learned `b` should still be near $0.5$. The same training loop, no changes — that is the point.

4. **Skip `opt.zero_grad()`.** In `experiments/08_sgd.py`, comment out **both** `opt.zero_grad()` lines (the one in the printed loop and the one in the silent loop) and run again. Without the reset, every `loss.backward()` *adds* to `.grad` instead of replacing it, so by step $k$ the optimiser is updating with the accumulated sum $g_1 + g_2 + \ldots + g_k$ rather than the current gradient. On this scalar quadratic, the visible symptom is **erratic** training: $x$ overshoots the target on step 1 ($x_1 = 0.6 \to x_3 = 3.02$ already past the answer at step 3), then oscillates with damped amplitude, ending around $2.31$ after 100 steps — close-ish, but not what a clean run produces. The takeaway is not "everything explodes"; it is "convergence guarantees of SGD assume gradients are reset every step, and silently break when they are not." Re-add the lines.

After each experiment, restore the file you changed before moving on. The next chapter assumes the §4.7 ending state.

---

## 4.9 Exercises

1. **Hand-step gradient descent on $f(x) = (x - 3)^2$ from $x_0 = 0$.** Compute $x_1, x_2, x_3$ on paper using the rule $x_{t+1} = x_t - 0.1 \cdot 2(x_t - 3)$. Compare with the trajectory printed by experiment 07. (Hint: the expected first three values are $0.6$, $1.08$, $1.464$.)
2. **Show that with the right learning rate, gradient descent converges in one step on a quadratic.** For $f(x) = a(x - x^\star)^2$, what value of $\eta$ makes $x_1 = x^\star$ regardless of $x_0$? (Answer in terms of $a$.)
3. **Mean squared error vs. sum of squared errors.** If you replace `((y_pred - y_data) ** 2).mean()` in experiment 09 with `((y_pred - y_data) ** 2).sum()`, the gradients are exactly $n$ times larger. To get the same trajectory, what does the learning rate need to become? Run it and verify.
4. **Why `with torch.no_grad():` matters.** In experiment 07, remove the `with torch.no_grad():` block (so the update line is just `x.sub_(eta * x.grad)`). Re-run. PyTorch raises a `RuntimeError`. Read the error message; in one sentence, explain what it is complaining about. (Hint: the error mentions "leaf Variable" and "in-place operation".)

---

## 4.10 What's next

Three new tools, one new helper:

- **loss** — a non-negative scalar that quantifies how wrong the parameters are. We used MSE; from Chapter 13 onward we use cross-entropy.
- **gradient descent** — the iterative rule $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$ that drives loss downward.
- **`torch.optim.SGD`** — the four-line training loop: `zero_grad`, forward, `backward`, `step`. Drop-in for any list of parameters.
- **`mygpt.set_seed`** — a one-liner that we now have as the canonical entry point for reproducibility.

In Chapter 5 we build the bridge from the integer ids `mygpt.to_ids` produces to the dense floating-point vectors a transformer actually consumes. We replace the one-hot encoding from §3.9 with a learned **embedding** — and because embeddings are themselves parameters, we will train them with exactly the loop we wrote in this chapter.

> **Looking ahead — what to remember from this chapter**
>
> 1. Loss is a scalar; gradient descent is a rule; the optimiser is the bookkeeping wrapper.
> 2. The four-line loop (`zero_grad` → forward → `backward` → `step`) is the same shape no matter how big the model is.
> 3. The learning rate is the most important hyperparameter you choose by hand — too small and learning is slow, too large and it diverges.
> 4. `mygpt.set_seed(0)` is how every later chapter starts a deterministic run.

On to [Chapter 5 — From text to numbers: tokens and embeddings](05_tokens_and_embeddings.md) *(coming soon)*.
