---
title: 21. Training-loop hardening
nav_order: 3
parent: Part II — Advanced Topics
---

# Chapter 21 — Training-loop hardening

The training loop we have been running since Chapter 14 has three visible defects that real LLM trainers fix on day one:

1. **It reports only training loss.** A model that perfectly memorises the training data has training loss zero — and zero ability to predict text it hasn't seen. We need *validation* loss too.
2. **It uses a constant learning rate.** Real training uses a **warmup → cosine decay** schedule: ramp up gently at the start (when gradients are noisy) and ramp down at the end (when small steps refine more than they disrupt).
3. **It can occasionally explode.** A single big-gradient batch can push parameters far enough that the next batch's loss spikes. **Gradient clipping** caps the L2 norm of the gradient vector before each optimiser step, eliminating this failure mode.

By the end of this chapter you will have:

- added `--val-split` and `--val-every` so `mygpt train` reports both train and validation loss,
- added `--schedule {constant, cosine}` and `--warmup` so the LR ramps up and decays,
- added `--max-grad-norm` for gradient clipping,
- watched the same Tiny Shakespeare run produce a much more *trustworthy* loss curve — final train and val converge to similar values, instead of train going to zero while we have no idea what's happening on held-out data.

All three flags default off so every Ch.17–20 expected output continues to bit-reproduce.

---

## 21.1 Setup

This chapter assumes Chapter 20 — `mygpt/` has the `pick_device` helper, the `--device` and `--precision` flags, and the `torch.autocast` wrapper.

If you skipped, recreate the state from `docs/_state_after_ch20.md` and download Tiny Shakespeare:

```bash
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 21.2 Validation loss: the diagnostic we have been missing

Up to now, the only number our training loop has reported is the loss on the *current training batch*. That number is a very local view: it tells us how well the model fits the 16 sequences we just sampled. It tells us nothing about how the model will behave on text it hasn't seen.

The standard fix is the **train/val split**. Hold out (typically) 10% of the corpus as a *validation set* the model never trains on. Every N steps, sample a few batches from the val set, run the forward pass (no backward, no optimizer step), and report the average loss.

Two reasons this matters even on a 1 MB tutorial corpus:

- **Detecting overfitting.** If train loss goes to zero while val loss flattens or rises, the model is memorising rather than generalising. Tiny Shakespeare is small enough that overfitting is very visible — and we will see it.
- **Trustworthy stopping criteria.** "Stop when val loss stops decreasing" is the standard rule. Without val loss, you stop on a guess.

Implementation: chop the 1-D `data` tensor into a train slice and a val slice:

```python
n_train = int((1.0 - args.val_split) * len(data))
train_data = data[:n_train]
val_data = data[n_train:]
```

Then a small helper that averages loss over a few batches sampled from `val_data`:

```python
@torch.no_grad()
def estimate_val_loss(
    model: "GPT",
    val_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    n_eval_batches: int = 10,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_eval_batches):
        x, y = get_batch(val_data, batch_size, seq_len)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)
```

Three details worth flagging:

- **`@torch.no_grad()`** disables gradient tracking — there's no backward, so we save memory and time.
- **`model.eval()` then `model.train()`** toggles dropout off then back on. (Our default `dropout=0.0` makes this a no-op for now, but the flip is the correct pattern.)
- **`n_eval_batches = 10`** trades variance for speed. Ten random batches give a reasonable estimate without making val evaluation more expensive than the training step itself.

---

## 21.3 The cosine warmup schedule

Real LLM training uses a learning-rate schedule that does two things:

- **Warmup.** Ramp `lr` linearly from 0 (or near-zero) up to the target `max_lr` over the first ~100 to ~2000 steps. The model's parameters at random initialisation produce wide-magnitude logits (recall §13.4 / §17.5 "confidently wrong"); a full-throttle LR while gradients are wild can knock the model off its initial trajectory before it learns anything. The warmup gives gradients time to settle.
- **Cosine decay.** After warmup, `lr` follows a half-cosine curve from `max_lr` down to a `min_lr` (typically 0 or 10% of max). The intuition: early on the model is far from optimal and big steps help; near convergence small steps refine more than they disrupt.

The formula:

$$
\text{lr}(t) \;=\; \begin{cases}
    \text{max\_lr} \cdot \dfrac{t}{\text{warmup}} & \text{if } t < \text{warmup} \\[6pt]
    \text{min\_lr} + \dfrac{1}{2}(\text{max\_lr} - \text{min\_lr}) \cdot \left(1 + \cos\!\Big(\pi \cdot \dfrac{t - \text{warmup}}{\text{total} - \text{warmup}}\Big)\right) & \text{if } t \ge \text{warmup}
\end{cases}
$$

The cosine branch starts at $1 + \cos(0) = 2$, so a half-amplitude of $\tfrac{1}{2}(\text{max} - \text{min})$ times $2$ equals $\text{max} - \text{min}$, giving `lr = max_lr` at $t = \text{warmup}$. It ends at $1 + \cos(\pi) = 0$, giving `lr = min_lr` at $t = \text{total}$. Smooth between.

**Append the following helper to** 📄 `src/mygpt/utils.py` (right after `get_batch`):

```python
def cosine_warmup_lr(
    step: int, warmup: int, total: int, max_lr: float, min_lr: float = 0.0
) -> float:
    """Cosine learning-rate schedule with linear warmup.

    Step indexing is 1-based: at step 1, returns max_lr / warmup (or max_lr if
    warmup == 0). After step >= total, returns min_lr.
    """
    if warmup > 0 and step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

And `estimate_val_loss` from §21.2:

```python
@torch.no_grad()
def estimate_val_loss(
    model: "GPT",
    val_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    n_eval_batches: int = 10,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_eval_batches):
        x, y = get_batch(val_data, batch_size, seq_len)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)
```

---

## 21.4 Gradient clipping

A single batch with very large gradients can corrupt the model. **Gradient clipping** caps the L2 norm of the gradient vector across all parameters before each optimiser step:

$$
\text{if } \|g\|_2 > c, \quad g \leftarrow g \cdot \dfrac{c}{\|g\|_2}
$$

PyTorch ships this as one call: `torch.nn.utils.clip_grad_norm_(model.parameters(), c)`. Default `c = 1.0` is what GPT-2, Llama, and most open-source training scripts use.

Two details:

- **Apply between `loss.backward()` and `optimizer.step()`.** That's the only window where the gradients exist on `param.grad`; before backward they're zeroed, and after `step()` the optimizer has already used them.
- **Direction-preserving.** Scaling all gradients by the same factor preserves the direction of the gradient vector. Clipping doesn't change *which way* we step, only how far.

---

## 21.5 Wiring the new flags into `_train_command`

Five additions and one `if/else` around the optimiser update. The defaults preserve Ch.20 behavior so existing chapters continue to bit-reproduce.

**Replace `_train_command` in** 📄 `src/mygpt/cli.py`:

```python
def _train_command(args) -> None:
    device = pick_device(args.device)

    with open(args.text_file) as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode(text).to(device)

    # Train/val split (val_split = 0 keeps Ch.17-style "all data is train")
    if args.val_split > 0.0:
        n_train = int((1.0 - args.val_split) * len(data))
        train_data = data[:n_train]
        val_data = data[n_train:]
    else:
        train_data = data
        val_data = None

    set_seed(0)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device:       {device}")
    print(f"precision:    {args.precision}")
    print(f"corpus chars: {len(text):,}")
    print(f"train chars:  {len(train_data):,}")
    if val_data is not None:
        print(f"val chars:    {len(val_data):,}")
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")
    print(f"schedule:     {args.schedule} (warmup={args.warmup})")
    print(f"max_grad_norm:{args.max_grad_norm}")

    set_seed(42)
    for step in range(1, args.steps + 1):
        # LR schedule
        if args.schedule == "cosine":
            lr_t = cosine_warmup_lr(step, args.warmup, args.steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_t

        x, y = get_batch(train_data, args.batch_size, args.seq_len)
        optimizer.zero_grad()
        if args.precision == "bf16":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        if args.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        is_print_step = step == 1 or step % args.print_every == 0 or step == args.steps
        is_val_step = (
            val_data is not None
            and args.val_every > 0
            and (step % args.val_every == 0 or step == args.steps)
        )
        if is_print_step or is_val_step:
            line = f"step {step:>5}: loss = {loss.item():.4f}"
            if is_val_step:
                vl = estimate_val_loss(
                    model, val_data, args.batch_size, args.seq_len
                )
                line += f"  val = {vl:.4f}"
            if args.schedule == "cosine":
                line += f"  lr = {lr_t:.2e}"
            print(line)

    save_checkpoint(model, tokenizer, args.output)
    print(f"\nsaved checkpoint to {args.output}")
```

And add five new flags to `p_train` in `main`:

**In `main`'s argparse setup, add to `p_train`** (right after the `--precision` block, before `set_defaults(...)`):

```python
    p_train.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of the corpus held out as validation data (0.0 = none, default).",
    )
    p_train.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Print val loss every N steps. Requires --val-split > 0.",
    )
    p_train.add_argument(
        "--schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="LR schedule. 'constant' (default) holds at --lr; 'cosine' linearly warms up over --warmup steps then cosine-decays to 0.",
    )
    p_train.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup steps for the cosine schedule (no effect if --schedule=constant).",
    )
    p_train.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Gradient-norm clip threshold. 0.0 (default) disables clipping.",
    )
```

---

## 21.6 Backward-compat: defaults still reproduce Ch.20

First sanity check — `mygpt train` with default flags must still produce the Ch.20 fp32 loss curve.

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output sh-default.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
corpus chars: 1,115,394
train chars:  1,115,394
vocab_size:   65
params:       207,296
steps:        2000
schedule:     constant (warmup=0)
max_grad_norm:0.0
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to sh-default.ckpt
```

The new `train chars:`, `schedule:`, and `max_grad_norm:` lines appear in the header, but the loss values are identical to Ch.20 / Ch.19 / Ch.17. With every new flag at its off-default, the training loop is the same loop. Backward-compat preserved.

---

## 21.7 The hardened recipe

Now run the same training with all three new features enabled:

```bash
uv run mygpt train tinyshakespeare.txt --device mps \
  --val-split 0.1 --val-every 500 \
  --schedule cosine --warmup 100 \
  --max-grad-norm 1.0 \
  --output sh-hardened.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
corpus chars: 1,115,394
train chars:  1,003,854
val chars:    111,540
vocab_size:   65
params:       207,296
steps:        2000
schedule:     cosine (warmup=100)
max_grad_norm:1.0
step     1: loss = 41.5789  lr = 1.00e-05
step   500: loss = 2.4393  val = 2.4744  lr = 8.95e-04
step  1000: loss = 2.2950  val = 2.2975  lr = 5.41e-04
step  1500: loss = 2.1387  val = 2.2136  lr = 1.61e-04
step  2000: loss = 2.1927  val = 2.2152  lr = 0.00e+00

saved checkpoint to sh-hardened.ckpt
```

This run is fully deterministic — re-run the command and you get the same numbers to four decimals. Several things to read off:

**1. The LR schedule is doing what it says.**
- Step 1: `lr = 1.00e-05` — warmup hasn't reached `--lr 1e-3` yet (it's at $1/100 \cdot 10^{-3} = 10^{-5}$).
- Step 500: `lr = 8.95e-04` — well past warmup, ~90% of `--lr` (cosine-decay barely started).
- Step 1000: `lr = 5.41e-04` — halfway through the 1900-step decay phase, roughly at the cosine midpoint.
- Step 1500: `lr = 1.61e-04` — late decay.
- Step 2000: `lr = 0.00e+00` — the schedule's `min_lr=0` end-state.

**2. Train and val loss travel together.** Step 500 train 2.44 vs val 2.47 — a 0.03 gap. Step 2000 train 2.19 vs val 2.22 — also 0.03. The model is *not* overfitting: it generalises to the held-out 10% as well as it fits the training 90%. That gap, more than the absolute loss number, is the diagnostic that matters.

**3. The final loss is *higher* than the Ch.17/19/20 default run (2.19 vs 2.08).** This is by design. The Ch.17 default uses a constant `lr = 1e-3` for all 2000 steps; the hardened run anneals to zero by the end. The constant-LR run reaches a lower *training* loss because the late-stage parameter updates keep nudging the model further into the training distribution. With the cosine schedule the model stops updating once we hit `min_lr=0`, leaving more capacity unused — and producing a model that's less overfit. The right comparison isn't "which gets lower train loss" but "which gives me a trustworthy val signal at the end." The hardened recipe wins on that metric.

**4. Step-1 loss is *different* from the Ch.20 default (41.58 vs 41.04).** Not because of warmup or schedule — at step 1 the optimizer hasn't applied any update yet, so the LR doesn't matter. The cause is `--val-split 0.1`: the training tensor is now 90% of the corpus, so `get_batch` (which samples random indices into `train_data`) sees a different range of indices and picks a different first batch. Loss at step 1 reflects the model's initial state on *that* particular batch.

---

## 21.8 Watch for overfitting (an experiment)

Train *without* a val set for far longer than necessary, and the train loss will keep falling while val loss starts to rise. To see it on Tiny Shakespeare, run:

```bash
uv run mygpt train tinyshakespeare.txt --device mps \
  --val-split 0.1 --val-every 200 \
  --steps 5000 \
  --output sh-overfit.ckpt
```

This uses the *constant* schedule and *no* gradient clipping, so the model is free to keep nudging itself toward the training data well past the point where it should stop. By step ~3500 the train loss continues to drop while val plateaus or rises — the textbook overfitting curve. (We won't paste the full output here because it takes ~75 s on M1 MPS; the experiment is worth running once.)

The hardened recipe (cosine to zero) makes this much harder to do by accident — once the LR hits zero, the model stops updating regardless of how far past convergence you've trained.

---

## 21.9 Experiments

1. **Effect of warmup.** Re-run the hardened recipe with `--warmup 0`. The early loss curve drops a touch faster — at step 500 you get ~2.41 vs warmup-100's 2.44. (Step-1 loss is unchanged because the LR doesn't affect the *forward* pass; only `optimizer.step` consumes it.) On a 207k-parameter model warmup is *insurance* for stability, not a free win; it earns its keep on bigger models with more parameters to destabilise.
2. **Effect of clipping.** Re-run the hardened recipe with `--max-grad-norm 0.0`. On Tiny Shakespeare with this small a model, clipping rarely fires (the gradients aren't dramatic). On a real-text corpus in Ch.28 it does, and you will see the difference.
3. **Cosine vs constant.** Re-run with `--schedule constant --warmup 0 --max-grad-norm 0.0 --val-split 0.1 --val-every 500`. The constant schedule reaches train loss ~2.13 and val loss ~2.16 at step 2000 (note: not exactly 2.0785 like Ch.17, because the val-split shortens `train_data` and shifts the batch sequence). Train and val are still close, but the absolute numbers depend on the train-data slice — the hardened recipe is consistent on this point.
4. **Effect of `n_eval_batches`.** In `estimate_val_loss`, change the default from 10 to 1. Val loss reported per step is now noisier (it sees only one batch). Change to 50 — barely any visible difference, but val evaluation is now 5× slower. The default of 10 is a defensible cost/precision compromise.

After each experiment, restore any file you changed before moving on.

---

## 21.10 Exercises

1. **Train at the LR=0 end.** With the cosine schedule, `lr` at the final step is exactly zero. Argue from the AdamW update rule that an LR=0 step does *no work* — the parameters don't change. Confirm experimentally: run the hardened recipe twice with the same seeds, save each checkpoint, and verify their `state_dict()`s are bitwise identical.
2. **Linear vs cosine decay.** Sketch a "linear decay" schedule that goes from `max_lr` to `0` linearly over `total - warmup` steps. Argue why cosine-decay spends more time near `max_lr` (where most of the learning happens) and more time near `0` (where the final refinement happens) than linear, with the same area under the curve. Hint: integrate $1 + \cos(\pi t/T)$ vs $1 - t/T$ over $[0, T]$.
3. **What does `clip_grad_norm_` actually return?** Read the PyTorch docs for `torch.nn.utils.clip_grad_norm_`. The function clips in-place but also *returns* the pre-clip total norm. Argue why this is useful for monitoring training stability — and add a `print` line in the chapter's training loop that records the gradient norm every `print_every` steps.
4. **n_eval_batches and variance.** With `n_eval_batches = 10`, val loss is averaged over $10 \cdot B \cdot T = 10 \cdot 16 \cdot 64 = 10240$ tokens. Argue why this is enough to keep run-to-run val noise below `0.01` for a converged model on Tiny Shakespeare, but not enough for a converged model on Wikipedia (Ch.28). Hint: variance of the mean scales with $1 / \sqrt{n}$.

---

## 21.11 What's next

We have a training loop that reports trustworthy diagnostics. The next chapter, **Chapter 22 — BPE from scratch**, leaves training behind and revisits *tokenization*: instead of one token per character (vocab ≈ 65), we build a byte-pair-encoding tokenizer that produces one token per *fragment* (vocab ≈ 1024), so each token carries more information and sequences are shorter. Ch.23 wires it into `mygpt`.

> **Looking ahead — what to remember from this chapter**
>
> 1. Train loss without val loss is a *partial* signal — you don't know if you're learning or memorising.
> 2. Cosine warmup-then-decay is the canonical LR schedule for transformer training. Constant LR is the toy mode.
> 3. `clip_grad_norm_(model.parameters(), 1.0)` between `backward()` and `step()` is one line of insurance against gradient explosions.
> 4. Defaults stay off so older chapters keep their reproducibility guarantees. Real training turns all three on.

On to [Chapter 22 — BPE from scratch (algorithm)](22_bpe_from_scratch.md).
