# Student walkthrough report: Chapter 26 — GQA: grouped-query attention

## Summary
- Document path: `/Users/egdeegde/dev/LLM Fundamentals/docs/26_gqa.md`
- Total sections walked: 9 of 9 executable subsections (§26.1 setup, §26.4 REPL, §26.5 + §26.6 + §26.7 code edits, §26.7 --help, §26.8 backward-compat, §26.9 GQA, §26.11 sampling, §26.12 exp 1/2/3)
- Files modified: 1 — `src/mygpt/__init__.py` (replace `MultiHeadAttention`, `TransformerBlock`, `GPT.__init__`, `save_checkpoint`, `load_checkpoint`; insert `num_kv_heads` resolution + 2 print lines into `_train_command`; add `--num-kv-heads` CLI flag)
- Files produced: 3 — `sh-mha.ckpt`, `sh-gqa.ckpt`, `sh-mqa.ckpt`
- Shell commands run: 9 (uv init, uv add, ls, REPL, --help, three `uv run mygpt train`, two `uv run mygpt generate`, one ad-hoc `uv run python -c …` for §26.12 Exp 1, one `--num-kv-heads 3` Exp 3)
- Issues found: **0** (0 blockers, 0 mismatches, 0 clarity, 0 prerequisite gaps, 0 sequencing, 0 polish)
- Final verdict: **Passes a code-along student.**

## Environment
- OS / shell: Darwin 25.2.0 arm64 / GNU bash 3.2.57
- Python: CPython 3.12.11
- uv: 0.8.0
- torch: 2.11.0
- mps available: True; cuda: False
- Working directory: `/tmp/code-along-runs/ch26-review-152913/mygpt`

## Walkthrough

### Sections §26.1 – §26.7 (setup + code edits)
- §26.1: `tinyshakespeare.txt` already present (1,115,394 bytes); `ls` succeeds, `curl` short-circuited by `||`.
- §26.4: `K.repeat_interleave(2, dim=0)` produces exactly the chapter's expected `tensor([[1.,1.,1.,1.],[1.,1.,1.,1.],[2.,2.,2.,2.],[2.,2.,2.,2.]])`. Pedagogical demo lands.
- §26.5: `MultiHeadAttention` replacement applied (new `num_kv_heads` parameter, divisibility check, `kv_repeat` attribute, narrowed $W_K$/$W_V$ Linears, new K/V reshape with `self.num_kv_heads`, `repeat_interleave` guarded by `kv_repeat > 1`, RoPE applied before repeat). `TransformerBlock` and `GPT.__init__` threaded through. Class boundaries unambiguous; chapter's "replace the existing X class" phrasing is clear.
- §26.6: `save_checkpoint` writes `num_kv_heads` via `getattr(model, "num_kv_heads", model.num_heads)`; `load_checkpoint` reads via `config.get("num_kv_heads", config["num_heads"])`. Backward-compat docstring updated.
- §26.7: `_train_command` resolves `num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else args.num_heads`, threads it through `GPT(...)`, prints `num_heads:` and `num_kv_heads:` lines. New `--num-kv-heads` CLI flag added after `--position`.
- `uv run mygpt train --help | tail -5` returns exactly the four expected lines:
  ```
    --num-kv-heads NUM_KV_HEADS
                          Number of K/V heads for grouped-query attention.
                          Default: same as --num-heads (full MHA, Ch.8). Must
                          divide --num-heads.
  ```
- Issues raised here: none.

### Section: §26.8 Backward-compat default (full MHA)
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --output sh-mha.ckpt`
- Output (key lines):
  ```
  device:       mps
  precision:    fp32
  norm:         layer
  position:     learned
  num_heads:    4
  num_kv_heads: 4
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

  saved checkpoint to sh-mha.ckpt
  ```
- Expected output match: yes — byte-exact, including the new `num_heads:` and `num_kv_heads: 4` lines. Loss curve bit-identical to Ch.21/24/25 default. The `if self.kv_repeat > 1` guard works as advertised: when `num_kv_heads == num_heads`, the `repeat_interleave` is skipped and the executed ops are identical to Ch.25's.
- Issues raised here: none.

### Section: §26.9 GQA run (`num_kv_heads = 2`)
- Command: `uv run mygpt train tinyshakespeare.txt --device mps --num-kv-heads 2 --output sh-gqa.ckpt`
- Output (key lines):
  ```
  num_heads:    4
  num_kv_heads: 2
  corpus chars: 1,115,394
  vocab_size:   65
  params:       190,912
  …
  step     1: loss = 41.1659
  step   500: loss = 2.6111
  step  1000: loss = 2.4060
  step  1500: loss = 2.2052
  step  2000: loss = 2.0854

  saved checkpoint to sh-gqa.ckpt
  ```
- Expected output match: yes — byte-exact. Param count `190,912` is exactly `207,296 - 16,384`; the chapter's derivation `4 layers × 2 (W_K + W_V) × (4096 - 2048) = 16,384` reproduces. Final loss `2.0854` is `0.0069` nats above the MHA `2.0785` — within the chapter's "essentially the same final loss … 0.007 nats worse" claim.
- Issues raised here: none.

### Section: §26.11 Sampling
- `mygpt generate --checkpoint sh-mha.ckpt --prompt "ROMEO:" --device cpu` produces:
  ```
  ROMEO:
  Thy momed has seltered, a neark'ly your tle centeloourse.
  Of therere hath thin beielly saneer best.

  BRINCE:
  Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
  Whe a bros swencurenty hou
  ```
  Byte-exact to the chapter's expected MHA sample (also byte-exact to the Ch.17 §17.6 baseline ROMEO sample — backward compat preserved through Ch.18 → Ch.26).
- `mygpt generate --checkpoint sh-gqa.ckpt --prompt "ROMEO:" --device cpu` produces:
  ```
  ROMEO:
  Thy momed haveseltered ad meament, ink helsterere
  Doues this therere hath thigh orell seaneer brie.

  BRIO:
  Whis bee ancesendy thowall a my be tooe spe st to alin hisere,
  By bres han shup the shere wa
  ```
  Byte-exact to the chapter's expected GQA sample. The first 14 chars `ROMEO:\nThy momed ` are indeed byte-identical to the MHA sample, as the chapter notes; divergence begins at the next token.
- Issues raised here: none.

### Section: §26.12 Experiments
- **Exp 1 (saved config):** `torch.load('sh-gqa.ckpt')['config']` returns `{'vocab_size': 65, 'embed_dim': 64, 'num_heads': 4, 'num_kv_heads': 2, 'num_layers': 4, 'max_seq_len': 64, 'norm_type': 'layer', 'position_type': 'learned'}` — byte-exact match to the chapter's quoted dict.
- **Exp 2 (MQA, `--num-kv-heads 1`):** `params: 182,720` — exactly the chapter's predicted `190,912 − 4 × 1024 − 4 × 1024`. Step 1 loss `41.3530`, step 2000 loss `2.2015` — that is `0.116` nats worse than GQA-2's `2.0854` and `0.123` nats worse than MHA's `2.0785`. The chapter's qualitative claim "you should see a measurable loss bump going from GQA-2 to MQA" is borne out: a >5% increase in final loss is well above noise, vs. the GQA-vs-MHA gap which is `0.007` nats (~0.3%).
- **Exp 3 (`--num-kv-heads 3`, non-divisor):** raises exactly `ValueError: num_heads (4) must be divisible by num_kv_heads (3)`, byte-exact to the chapter's predicted message. Validation works.
- Exp 4 (parameter-count formula derivation) is a pen-and-paper exercise; not run here. The hint is correct (token + position embeddings + per-layer attention with split MHA vs MQA shapes + per-layer MLP + 9 norms + tied LM head) and the three checkpoints match the formula at our defaults.
- Issues raised here: none.

## Issues

None.

## Confidence and caveats

I walked the chapter end-to-end including three ~30 s training runs (MHA, GQA-2, MQA) and two generation calls. Every Expected Output block matches character-for-character, including the new `num_heads:` and `num_kv_heads:` print lines, the param counts (207,296 / 190,912 / 182,720 — all matching the chapter's derivations exactly), the bit-identical Ch.25 default loss curve, the GQA loss curve that ends within 0.007 nats of MHA, both ROMEO samples, the saved-config dict, and the `ValueError` text from the non-divisor Exp 3.

§26.10 (KV-cache discussion) is prose-only, no commands; the architectural prerequisite framing is consistent with §26.5–§26.9. §26.13 exercises are conceptual (RoPE-before-repeat equivalence, KV-cache shapes); the hints are correct.

Two small style notes I considered but did not raise as findings:
- §26.7's "replace the two `print(...)` lines for `norm:` and `position:` with these four" works correctly when followed literally — the four shown lines start with the same two originals plus two additions, so a literal paste yields the right code path. The phrasing could equivalently say "add these two lines after `position:`", but it is unambiguous as written.
- §26.5's "replace the existing `GPT.__init__` (the `forward` is unchanged) with this version. The simplest way is to replace the whole class definition" is mildly redundant — the shown code block already contains the full class. A literal student replacing the entire class block gets the correct result.

Neither of these would have stopped me; both are below the polish bar for this reviewer pass.

The chapter is ready to commit.
