---
title: 19. Device-aware training
nav_order: 1
parent: Part II — Advanced Topics
---

# Chapter 19 — Device-aware training (MPS / CUDA / CPU)

The first chapter of Part II is the smallest one: change `mygpt` so the same code runs on three different compute backends, and let the user pick which one with a flag.

By the end you will have:

- a `pick_device(arg)` helper that resolves `"auto"`, `"cuda"`, `"mps"`, or `"cpu"` to a `torch.device`,
- an updated `set_seed` that seeds the RNG on every available device,
- a `--device` flag on `mygpt train` and `mygpt generate`,
- measured the speedup of MPS over CPU on the Chapter 17 training run (it is real but smaller than you might expect at this model size),
- understood why the *same trained model* produces *different* samples when generation runs on a different device.

There is no new mathematics. This chapter is a small refactor that pays off every time the rest of Part II runs a training step.

---

## 19.1 Setup

This chapter assumes you finished Chapter 18 — `mygpt/` exists at the project root with `CharTokenizer`, `GPT`, `generate`, `save_checkpoint`, `load_checkpoint`, and the `mygpt train` / `mygpt generate` CLI.

If you skipped Part I, recreate the state from a clean directory:

```bash
uv init mygpt --package
cd mygpt
mkdir -p experiments
uv add torch numpy
```

Overwrite **`src/mygpt/__init__.py`** with the Chapter 18 ending state from `docs/_state_after_ch18.md`. Then download the Tiny Shakespeare corpus we'll use for the speedup demo:

```bash
curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 19.2 Why device-aware?

PyTorch tensors live on a *device*: CPU memory by default, an NVIDIA GPU (`cuda`) if you have one, or Apple Silicon's GPU (`mps`) on an M-series Mac. Operations on a tensor execute on that tensor's device, and the *same Python code* can dispatch to a CPU, an NVIDIA GPU, or an Apple GPU depending only on where the tensor was placed.

We have been quietly running everything on CPU because no `mygpt` code mentioned a device. The default tensor device is CPU, and PyTorch never moves things off CPU on its own. To change that, we need three tiny additions:

1. A way to *resolve* a string like `"mps"` (from the command line) to a `torch.device`.
2. A way to *seed* the device's RNG so generation stays deterministic.
3. A way to *move* the model and the data to the chosen device before the training loop runs.

That's the whole chapter.

---

## 19.3 `pick_device`: resolving a flag to a device

The new helper:

```python
def pick_device(arg: str = "auto") -> torch.device:
    """Resolve a device spec to a torch.device.

    "auto" prefers CUDA over MPS over CPU. The other strings
    ("cuda", "mps", "cpu") are passed through.
    """
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)
```

Three things:

- **`"auto"` picks the fastest available**, with CUDA > MPS > CPU. On an M1 Mac with no NVIDIA card, `auto` lands on `mps`. On a Linux box with a 4090, `auto` lands on `cuda`. On Anything Else, you get `cpu`.
- **The other three strings are passed through verbatim** so the user can force a specific backend. Useful for benchmarking and for chapters in Part II that need a specific device for a specific point.
- **No `try/except`** — if the user asks for `cuda` on a Mac, `torch.device("cuda")` succeeds (it just constructs a device object) but the *first operation* on a CUDA tensor will fail with a clear PyTorch error. We don't pre-empt the error with a friendlier message; the real one is fine.

**Append the following helper to** 📄 `src/mygpt/__init__.py` (right after `set_seed`, which we will update next):

```python
def pick_device(arg: str = "auto") -> torch.device:
    """Resolve a device spec to a torch.device.

    ``"auto"`` prefers CUDA over MPS over CPU. The other strings
    (``"cuda"``, ``"mps"``, ``"cpu"``) are passed through.
    """
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)
```

---

## 19.4 `set_seed` for every device

`set_seed` from Chapter 4 only seeds the CPU RNG:

```python
def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's CPU random number generator."""
    torch.manual_seed(seed)
```

Each device has its own RNG state. CUDA's lives on the GPU. MPS's lives on the Apple GPU. `torch.manual_seed` updates *only* the CPU one. To make `set_seed(0)` mean "make the run deterministic" regardless of device, we update all three.

**Replace `set_seed` in** 📄 `src/mygpt/__init__.py`:

```python
def set_seed(seed: int = 0) -> None:
    """Seed PyTorch's RNGs across whatever devices are available."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
```

Important: **seeding all three RNGs to the same number does NOT mean the three devices produce the same random values.** CPU and MPS use different pseudo-random algorithms internally, so the same seed produces different sequences. We will see this directly in §19.7 when generation produces different samples on CPU and MPS even with the same `--seed 0`.

---

## 19.5 Wiring the device into the CLI

Two edits to `_train_command` and `_generate_command`. Each takes a single new line at the top — `device = pick_device(args.device)` — and adds `.to(device)` after the model and the data are constructed.

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
    print(f"corpus chars: {len(text):,}")
    print(f"vocab_size:   {tokenizer.vocab_size}")
    print(f"params:       {n_params:,}")
    print(f"steps:        {args.steps}")

    set_seed(42)
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.seq_len)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.print_every == 0 or step == args.steps:
            print(f"step {step:>5}: loss = {loss.item():.4f}")

    save_checkpoint(model, tokenizer, args.output)
    print(f"\nsaved checkpoint to {args.output}")
```

Two changes from Chapter 18: `device = pick_device(args.device)` near the top, and three `.to(device)` calls (`tokenizer.encode(text).to(device)`, `model.to(device)`, and the print of `device:`).

**Replace `_generate_command` in** 📄 `src/mygpt/__init__.py`:

```python
def _generate_command(args) -> None:
    device = pick_device(args.device)
    print(f"device: {device}\n")
    model, tokenizer = load_checkpoint(args.checkpoint)
    model.to(device)
    set_seed(args.seed)
    prompt = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    out = generate(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0]))
```

The leading `device: <name>` line is what the §19.8 exp 1 prose below relies on — without it, `--device auto` gives no on-screen confirmation of which device was picked.

And one matching change inside `load_checkpoint` so checkpoints load cleanly on any device, regardless of where they were saved:

**Replace the first line of `load_checkpoint`'s body in** 📄 `src/mygpt/__init__.py`:

```python
def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`.

    Always loads to CPU; the caller is responsible for `.to(device)` afterwards.
    """
    ckpt = torch.load(path, map_location="cpu")
    # ... rest unchanged
```

The new bit is `map_location="cpu"`. Without it, `torch.load` tries to put the tensors back on whatever device they were saved from; loading a CUDA-saved checkpoint on a Mac then fails. With `map_location="cpu"`, the tensors arrive on CPU and the caller (`_generate_command`) does the `.to(device)`.

Finally, add the `--device` argument to both subparsers in `main`:

**In `main`'s argparse setup, add to BOTH `p_train` and `p_gen`:**

```python
    p_train.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda → mps → cpu in that order.",
    )
    # … then existing p_train.set_defaults(...) line
```

```python
    p_gen.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda → mps → cpu in that order.",
    )
    # … then existing p_gen.set_defaults(...) line
```

That's the entire code change.

---

## 19.6 Verifying it: same model, three runs

Three runs with the same Tiny Shakespeare corpus that Chapter 17 used. We force the device explicitly so the chapter is reproducible whether you have an M1, a CUDA box, or just a CPU.

**Run 1 — CPU baseline (matches Ch.17 §17.5 exactly):**

```bash
uv run mygpt train tinyshakespeare.txt --device cpu --output shakespeare-cpu.ckpt
```

**Expected output:**

```text
device:       cpu
corpus chars: 1,115,394
vocab_size:   65
params:       207,296
steps:        2000
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to shakespeare-cpu.ckpt
```

(Wall-clock on the author's M1: ~42 s. Same loss values as Chapter 17.)

**Run 2 — MPS, same flags except `--device mps` (M1/M2/M3/M4 only):**

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output shakespeare-mps.ckpt
```

**Expected output (MPS):**

```text
device:       mps
corpus chars: 1,115,394
vocab_size:   65
params:       207,296
steps:        2000
step     1: loss = 41.0367
step   500: loss = 2.5944
step  1000: loss = 2.3529
step  1500: loss = 2.1795
step  2000: loss = 2.0785

saved checkpoint to shakespeare-mps.ckpt
```

(Wall-clock on the author's M1: ~28 s. **Speedup: ~1.5×.**) The loss values are *identical* to the CPU run on this Mac and PyTorch version, because the operations involved (matmul, attention, AdamW updates) are deterministic to fp32 across these two devices for this small a model. With bigger models or bf16 (Chapter 20) the numbers will diverge.

The speedup from MPS at this scale is real but smaller than the popular "3–5×" headline. That headline applies to *bigger* models, where the per-kernel-launch overhead on MPS is amortised over more parameters. We will see that play out in Chapter 28's full Wikipedia training run.

---

## 19.7 Same model, two devices, two samples

Now generate from each checkpoint. The interesting result is that the *same trained model* produces *different* text on different devices, even with the same seed.

```bash
# Generate on CPU from the CPU checkpoint
uv run mygpt generate --checkpoint shakespeare-cpu.ckpt --prompt "ROMEO:" --device cpu
```

**Expected output (CPU):**

```text
device: cpu

ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

The generated text below the `device: cpu` line is exactly the output we recorded in Ch.17 §17.6 — every byte. Backward compatibility is preserved.

```bash
# Generate on MPS from the MPS checkpoint
uv run mygpt generate --checkpoint shakespeare-mps.ckpt --prompt "ROMEO:" --device mps
```

**Expected output (MPS):**

```text
device: mps

ROMEO:
That hal nos themst your hallon murd: you tingen
I'stene of therseetele dneelf thall tid, seare hem stery
Felly: wild weare as his shery mongere. stly his whoth,
Thy wit hor and me we of bust hond an
```

Different speaker structure, different gibberish words, different rhythm. **Why?** The trained model is *identical* in both cases — its weights are the same to fp32 (we just verified this via the matching loss curves). What differs is the *random-sampling* step inside `generate`. `torch.multinomial(probs, num_samples=1)` consumes random numbers from the device's RNG, and the CPU's RNG and the MPS RNG produce different sequences from the same seed `0`.

Try the cross-load case for completeness:

```bash
# Same MPS-style sample, but loaded from the CPU checkpoint:
uv run mygpt generate --checkpoint shakespeare-cpu.ckpt --prompt "ROMEO:" --device mps
```

**Expected output (CPU checkpoint, MPS device):**

```text
device: mps

ROMEO:
That hal nos themst your hallon murd: you tingen
…
```

Identical generated text to the MPS-checkpoint MPS-device run, because the model weights are device-independent and the sample diverges only at the multinomial step on the device. This confirms the checkpoint format from Chapter 18 is fully device-portable: train on CUDA, generate on a Mac, deploy from a Linux box — same checkpoint, no conversion.

---

## 19.8 Experiments

1. **`--device auto` on your machine.** Run `uv run mygpt generate --checkpoint shakespeare-cpu.ckpt --prompt "ROMEO:"` *without* `--device`. The CLI prints which device was chosen; on an M1 with no CUDA, it picks `mps`, so the sample matches the MPS run above. On a CUDA box, `auto` picks `cuda` and you get yet another sample.
2. **Force `cuda` on a non-CUDA system to see the failure mode.** `uv run mygpt generate --checkpoint shakespeare-cpu.ckpt --prompt "ROMEO:" --device cuda` raises a PyTorch error. The exact message depends on whether your torch wheel was built with CUDA support: on a macOS arm64 install (no CUDA in the build) you get `AssertionError: Torch not compiled with CUDA enabled`; on a Linux/Windows install built with CUDA but without an NVIDIA driver you get `RuntimeError: Found no NVIDIA driver on your system`. Compare the timing to what `pick_device` itself does — it just constructs `torch.device("cuda")` with no error; the failure happens later, on the first tensor op against that device.
3. **Time it yourself.** `time uv run mygpt train tinyshakespeare.txt --device cpu --steps 200` and `time uv run mygpt train tinyshakespeare.txt --device mps --steps 200`. Record the ratio. On the author's M1 it is ~1.5×; on a more recent M-series Mac with more GPU cores, expect closer to 2–3×.
4. **Same seed, different device, sample comparison.** Run generation twice with `--seed 0 --device cpu` then `--seed 0 --device mps`. The samples diverge within the first few generated tokens — the prompt `"ROMEO:"` is so confidently followed by `"\n"` and the next character that even with different RNGs the most-likely token wins for a moment, but once the distribution flattens the device's RNG decides. (Exercise 4 below extends this to a numerical check on the random number sequence itself.)

---

## 19.9 Exercises

1. **Why `map_location="cpu"`?** Sketch what happens if you `torch.save` a checkpoint on a CUDA box and then `torch.load(path)` (without `map_location`) on a Mac. Walk through PyTorch's deserialisation steps; predict the failure mode. (Hint: the saved tensors carry their device with them.)
2. **What does `model.to(device)` actually do?** Look up the `nn.Module.to(...)` semantics. Argue that it is a *no-op* for parameters already on that device, and that it modifies the module in-place rather than returning a new module. Confirm experimentally with `id(model) == id(model.to('cpu'))`.
3. **Why three RNG seeds?** Argue that calling `torch.manual_seed(0)` is *insufficient* to make a CUDA training run deterministic, even though the loss values would be deterministic anyway thanks to fp32 matmul determinism. Hint: the model's *initial* weights come from CPU sampling, but the *batch sampling* in `get_batch` happens on the same device as `data`.
4. **Verify the MPS RNG is different.** Write a one-liner: `torch.manual_seed(0); torch.rand(3, device='cpu')` versus `torch.mps.manual_seed(0); torch.rand(3, device='mps')`. The two tensors are *not* equal. (This is the root cause of the diverging samples in §19.7.)

---

## 19.10 What's next

We can train and generate on three devices. The next chapter, **Chapter 20 — Mixed precision (bf16)**, adds a `--precision` flag that uses 16-bit floats inside the forward pass. On CUDA this is a free 1.5–2× speedup on top of MPS-vs-CPU; on MPS it is supported with caveats; on CPU it is pointless. Everything else in Part II will start with `--device <something> --precision <something>` baked in.

Looking ahead — what to remember from this chapter:

1. `pick_device("auto")` resolves to CUDA > MPS > CPU. The string forms (`"cuda"`, `"mps"`, `"cpu"`) force a specific backend.
2. `set_seed` now seeds CPU, CUDA, and MPS RNGs together — but seeding three RNGs to the same number does *not* make them produce the same values.
3. Same model + different device → the loss values are typically identical (deterministic matmul) but the *generated samples* diverge (per-device RNG).
4. Checkpoints are device-portable: `load_checkpoint` always lands on CPU, and the caller does `.to(device)`.
