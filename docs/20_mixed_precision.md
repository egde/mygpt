---
title: 20. Mixed precision training (bf16)
nav_order: 2
parent: Part II — Advanced Topics
---

# Chapter 20 — Mixed precision training (bf16)

Modern LLMs do not train in 32-bit float. They train in **brain-float-16** (`bfloat16`, or `bf16`) — a 16-bit format that keeps fp32's wide *exponent* range (so it doesn't overflow on big gradients) and trades away its precision in the mantissa (it has only 7 mantissa bits instead of fp32's 23). Half the bytes per tensor, similar dynamic range, similar convergence behaviour.

By the end of this chapter you will have:

- understood what bf16 is, why GPT-2-class and Llama-class models use it, and what you give up,
- added a `--precision {fp32, bf16}` flag to `mygpt train` and `mygpt generate`,
- wrapped the forward pass in `torch.autocast` so PyTorch handles the dtype conversions automatically,
- measured what bf16 actually costs and gives at toy scale on M1 MPS — a pedagogically honest result, where bf16 is slightly *slower* and the win arrives only at Chapter 28's bigger model.

The default precision stays **fp32**. Every Part-I and Ch.19 expected output continues to bit-reproduce. bf16 is opt-in.

---

## 20.1 Setup

This chapter assumes Chapter 19 — `mygpt/` has the `pick_device` helper, the multi-device `set_seed`, and the `--device` flag on both subcommands.

If you skipped, recreate the state from `docs/_state_after_ch19.md` in a clean directory and download Tiny Shakespeare:

```bash
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 20.2 What bf16 is

A 32-bit float (fp32) splits its bits like this:

```text
fp32:  [sign 1] [exponent 8] [mantissa 23]   →  range ≈ ±3.4 × 10³⁸,  ≈ 7 decimal digits of precision
```

The two 16-bit "half" formats trade off differently:

```text
fp16:  [sign 1] [exponent 5] [mantissa 10]   →  range ≈ ±6.5 × 10⁴ ,  ≈ 3 decimal digits
bf16:  [sign 1] [exponent 8] [mantissa  7]   →  range ≈ ±3.4 × 10³⁸,  ≈ 2 decimal digits
```

bf16 keeps fp32's *exponent* width (8 bits) — so it can represent numbers from $10^{-38}$ to $10^{38}$ without overflow or underflow — but spends only 7 bits on the mantissa. The catch is precision: in fp32 we represent $\pi$ as $3.1415927\ldots$; in bf16 we get something like $3.140625$. Coarse, but for *training a neural net* that turns out to be enough — gradients are noisy anyway, and the savings (half the memory bandwidth, faster matmul on GPUs that have bf16 hardware) more than pay for the precision loss.

`fp16` (the older "half" format) saves precision but **does** overflow during training, which is why bf16 became the standard for transformer training: same compactness, no overflow.

---

## 20.3 `torch.autocast`: bf16 *only inside the forward pass*

You don't manually convert tensors to bf16. PyTorch ships an **autocast** context manager that does it automatically: inside the `with` block, certain ops (matmul, attention, GELU) run in bf16; the values they return are bf16 too; everything outside the block stays fp32.

The key API:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    logits, loss = model(x, y)     # forward pass runs in bf16
loss.backward()                    # gradients computed in fp32 because we exited the context
optimizer.step()                   # weight update runs in fp32
```

The pattern is **forward in bf16, backward + optimizer in fp32**. This is the canonical "mixed precision" recipe — the *parameters* live in fp32, only the *activations* and *intermediate matmul outputs* run in bf16. The fp32 master copy of the weights is what `optimizer.step()` updates; the bf16 path is a transient compute-time optimisation.

Two reasons this works:
- The matmul inside the forward pass is where >90% of the compute goes. If matmuls are 2× cheaper, the whole forward is roughly 2× cheaper.
- The optimizer step is dominated by *parameter count*, not compute. AdamW updates ~1.5 ops per parameter. Doing this in fp32 is essentially free.

A `GradScaler` is needed for fp16 (because fp16 overflows when gradients are big and you have to "scale up the loss" before backprop). bf16 does **not** need a scaler — its dynamic range matches fp32's. Our code therefore never imports `torch.cuda.amp.GradScaler`.

---

## 20.4 Wiring `--precision` into the CLI

Three small edits to `src/mygpt/__init__.py`. First, replace the body of the training loop in `_train_command` to wrap the forward in autocast when `args.precision == "bf16"`. Add a `print(f"precision: ...")` line so the run log says what it did.

**Replace `_train_command` in** 📄 `src/mygpt/__init__.py`:

```python
def _train_command(args) -> None:
    device = pick_device(args.device)

    with open(args.text_file) as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode(text).to(device)

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
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")

    set_seed(42)
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.seq_len)
        optimizer.zero_grad()
        if args.precision == "bf16":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.print_every == 0 or step == args.steps:
            print(f"step {step:>5}: loss = {loss.item():.4f}")

    save_checkpoint(model, tokenizer, args.output)
    print(f"\nsaved checkpoint to {args.output}")
```

The change is one new `print` line and one `if/else` around the forward call. Notice that **`loss.backward()` and `optimizer.step()` are *outside* the autocast block.** That's by design — gradients and weight updates stay in fp32.

**Replace `_generate_command` in** 📄 `src/mygpt/__init__.py`:

```python
def _generate_command(args) -> None:
    device = pick_device(args.device)
    print(f"device: {device}\n")
    model, tokenizer = load_checkpoint(args.checkpoint)
    model.to(device)
    set_seed(args.seed)
    prompt = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    if args.precision == "bf16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = generate(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
    else:
        out = generate(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(out[0]))
```

Finally, add the `--precision` flag to both subparsers:

**In `main`'s argparse setup, add to BOTH `p_train` and `p_gen`** (right after the `--device` block we added in Ch.19, before the matching `set_defaults(...)`):

```python
    p_train.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Forward-pass precision. fp32 (default) is bit-deterministic; bf16 uses torch.autocast.",
    )
```

(and the matching block on `p_gen`).

---

## 20.5 fp32 still bit-reproduces Ch.19

First sanity check — the new code with `--precision fp32` (the default) must still produce the *same* loss curve as Chapter 19. If it didn't, we'd have changed semantics, not just added a feature.

```bash
uv run mygpt train tinyshakespeare.txt --device mps --precision fp32 --output sh-fp32.ckpt
```

**Expected output:**

```text
device:       mps
precision:    fp32
corpus chars: 1,115,394
vocab_size:   65
params:       207,296
steps:        2000
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to sh-fp32.ckpt
```

Same 41.0367 / 2.5944 / 2.3529 / 2.1795 / 2.0785 sequence as Chapter 19 §19.6. Backward compatibility preserved — the autocast block is bypassed entirely when `--precision fp32`.

(Wall-clock on the author's M1: ~29 s.)

---

## 20.6 bf16: close, but not the same

```bash
uv run mygpt train tinyshakespeare.txt --device mps --precision bf16 --output sh-bf16.ckpt
```

**Expected output (within tolerance):**

```text
device:       mps
precision:    bf16
corpus chars: 1,115,394
vocab_size:   65
params:       207,296
steps:        2000
step     1: loss = 41.0393
step   500: loss = 2.5926
step  1000: loss = 2.3532
step  1500: loss = 2.1793
step  2000: loss = 2.0797

saved checkpoint to sh-bf16.ckpt
```

(Wall-clock on the author's M1: ~36 s. **bf16 is slower than fp32 here.**)

Two important things to notice:

1. **The loss values are *close* to fp32 but not identical.** Step 1: 41.0393 vs 41.0367 — a difference of 0.003. Step 2000: 2.0797 vs 2.0785 — a difference of 0.001. The model converges to essentially the same place, just along a slightly different path. This is the bf16 precision tax: 7 mantissa bits is enough for training to *work*, not enough for it to be bit-deterministic against fp32.

2. **bf16 numbers are NOT bit-deterministic across runs.** Run the bf16 command twice in a row and you will get *slightly different* loss curves each time — typically within ±0.01 at any step. This is because MPS bf16 matmul is non-deterministic at the kernel level (different reduction orderings between runs). For the chapter to verify, treat the bf16 expected outputs as **±0.01 tolerance**, not bit-exact.

---

## 20.7 Why bf16 is *slower* here (and won't be in Chapter 28)

The headline result: **bf16 took ~36 s; fp32 took ~29 s.** bf16 is ~25% *slower* at this scale on this Mac.

That's the opposite of what bf16 is famous for. The reason is overhead. `torch.autocast` has to:

- Cast every input tensor entering an autocast-listed op to bf16.
- Cast every output back if the next op isn't autocast-listed.
- Track the cast graph so backward can compute gradients in fp32.

For our 207k-parameter model with `(B=16, T=64)` activations, the *per-op work* is small — the matmul kernel finishes in microseconds — and the *per-op cast cost* is comparable. We pay the autocast overhead on every op and only save a few percent on the matmul itself. Net: slower.

Where does bf16 win? When the matmul is *expensive enough* that even halving its cost dominates the autocast overhead. Concretely:

- Bigger embed_dim (the matmul is $O(C^2)$).
- Bigger batch / sequence length (more elements per matmul).
- A device with *bf16 tensor cores* (NVIDIA Ampere/Hopper, Apple M3 Pro and up). On those, the matmul itself is 2× faster in bf16, not just narrower.

We will see all three effects in Chapter 28 — `embed_dim=192`, ~10M parameters, 500 MB corpus, bf16 actually faster on M1 MPS. The setup we just built is what makes that experiment trivial: same `mygpt train`, just `--precision bf16`.

The honest summary: **bf16 is not a free win at toy scale. It is a free win at production scale.** Our default stays fp32 precisely so the toy chapters keep their reproducibility guarantees; readers turn it on for Ch.28.

---

## 20.8 Backward-compat smoke test

Before moving on, confirm Ch.18 / Ch.19 checkpoints still load and generate as expected — the new code added a flag but didn't change the checkpoint format.

```bash
# Generate from the fp32 checkpoint we just saved (default --precision fp32):
uv run mygpt generate --checkpoint sh-fp32.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output:**

```text
device: cpu

ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

Identical to Ch.17 §17.6 / Ch.19 §19.7. No regression.

---

## 20.9 Experiments

1. **bf16 generation from a fp32 checkpoint.** `uv run mygpt generate --checkpoint sh-fp32.ckpt --prompt "ROMEO:" --device mps --precision bf16`. The `device: mps` line prints, then a sample. Compare it to the fp32-precision MPS sample from Ch.19 §19.7. Most generated tokens will agree (the multinomial top-k pin keeps things close), but bf16's lower precision tilts a few sampling choices.
2. **Time it on your machine.** `time uv run mygpt train tinyshakespeare.txt --device mps --precision fp32 --steps 200` and `time uv run mygpt train tinyshakespeare.txt --device mps --precision bf16 --steps 200`. Compute the ratio. On the author's M1 it is ~1.25× *slower* for bf16. On a CUDA box with bf16 tensor cores (Ampere generation or newer), expect bf16 to be ~1.5× *faster*.
3. **Force CPU bf16.** `uv run mygpt train tinyshakespeare.txt --device cpu --precision bf16 --steps 200`. CPU autocast still works — the loss curve will be close to the MPS-bf16 curve — but there is no speed benefit. CPU autocast is mostly a code-portability convenience.
4. **Run bf16 training twice.** `uv run mygpt train tinyshakespeare.txt --device mps --precision bf16 --steps 200` then again. Compare loss values at each step. They differ by ~±0.001 — the non-determinism §20.6 mentions, made concrete.

After each experiment, restore any file you changed before moving on.

---

## 20.10 Exercises

1. **Why bf16 not fp16?** The chapter says fp16 overflows during training. Sketch a numerical example: a single gradient component of magnitude $7 \times 10^4$ in fp32. What happens when we cast it to fp16? To bf16? (Hint: fp16's max is $\approx 6.5 \times 10^4$.)
2. **Mantissa precision.** A bf16 value has only 7 mantissa bits, so consecutive representable numbers near 1.0 are spaced about $2^{-7} \approx 0.008$ apart. Argue why this is *enough* precision for training a neural net's *weights* but not enough for, say, scientific simulation. (Hint: gradients are inherently noisy; exact arithmetic isn't required.)
3. **Why is the optimiser step *outside* autocast?** Sketch what would happen if `optimizer.step()` ran in bf16. (Hint: the AdamW update is `w := w - lr * m / (sqrt(v) + eps)`. With `lr ≈ 1e-3` and `w` a typical weight of magnitude `0.1`, the update size is `~1e-4` — *below* bf16's resolution at that scale, so the update would silently round to zero.)
4. **Tracing the autocast graph.** PyTorch logs per-op casts when you set the env var `TORCH_AUTOCAST_DEBUG=1` (in some torch versions). Try it — `TORCH_AUTOCAST_DEBUG=1 uv run mygpt train tinyshakespeare.txt --device cpu --precision bf16 --steps 5`. The dispatch log shows which ops cast and which stay fp32.

---

## 20.11 What's next

We have device-aware (Ch.19) and precision-aware (Ch.20) training. The infrastructure is now real-LLM-shaped from a deployment perspective.

The next chapter, **Chapter 21 — Training-loop hardening**, fixes the *training* itself: validation loss, cosine LR schedule with warmup, gradient clipping. After Ch.21 the loss curves stop being noisy aggregations and start being trustworthy diagnostic signals.

Looking ahead — what to remember from this chapter:

1. bf16 is fp32 with the mantissa truncated. Same range, less precision. Half the bytes per tensor.
2. `torch.autocast(device_type=..., dtype=torch.bfloat16)` wraps the forward; backward + optimizer stay in fp32.
3. fp32 stays the default so existing chapters bit-reproduce. bf16 is opt-in.
4. **bf16 is slower than fp32 at toy scale** because autocast overhead dominates. The win arrives at Ch.28's larger model. The infrastructure we built today is what makes that win one flag away.
