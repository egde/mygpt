---
title: 26. GQA — grouped-query attention
nav_order: 8
parent: Part II — Advanced Topics
---

# Chapter 26 — GQA: grouped-query attention

Chapter 8 split attention into `num_heads` parallel heads. Each head got its own $W_Q$, $W_K$, and $W_V$ — three projections per head, each shaped `(embed_dim, head_dim)`. With four heads and `embed_dim = 64` that's 12 small matrices, totalling $4 \times 4 \times 64 \times 16 = 16{,}384$ parameters per attention layer (counting $W_O$).

For tiny Shakespeare this is fine. At Llama-3 scale (`embed_dim = 8192`, `num_heads = 64`) the same structure makes the K/V projections a serious chunk of the model's parameters — and an even more serious chunk of the **KV cache** at inference time, which is the working set of stored keys and values that grows linearly with the generated sequence. **Grouped-query attention** (GQA) is the answer the modern open-weight LLMs converged on: keep the full set of *query* heads (you want many independent "viewing angles" on each token), but use **fewer K and V heads**, repeated across query groups.

By the end of this chapter you will have:

- understood the central GQA picture — `num_query_heads = G × num_kv_heads`, where each KV head is shared across $G$ query heads,
- parameterised `MultiHeadAttention` by `num_kv_heads` (default `= num_heads`, so Part-I behaviour bit-reproduces),
- added a `--num-kv-heads N` CLI flag and a `num_kv_heads` field to checkpoint config,
- watched the parameter count drop by exactly **16,384** (8% of the model) when going from `--num-kv-heads 4` to `--num-kv-heads 2`,
- seen GQA hit essentially the **same final loss** as full MHA on the same corpus and step budget (2.0854 vs 2.0785) at 8% fewer parameters — the characteristic GQA result,
- understood why the inference-time payoff (smaller KV cache) is bigger than the training-time payoff (parameters).

Backward compat: pre-Ch.26 checkpoints have no `num_kv_heads` field and default to `num_heads` on load — every Part-I `.ckpt` continues to work unchanged.

---

## 26.1 Setup

This chapter assumes Chapter 25 — `mygpt/` has `precompute_rope_cache`, `apply_rope`, `position_type` threaded through `MultiHeadAttention` / `TransformerBlock` / `GPT`, and a `--position {learned, rope}` CLI flag. We are about to add one more knob along the same plan.

```bash
ls tinyshakespeare.txt || curl -s -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You are ready.

---

## 26.2 What multi-head attention actually spends parameters on

Recall the shape of a single attention layer at our default config (`embed_dim = 64`, `num_heads = 4`, `head_dim = 16`):

| Matrix | Shape          | Params |
|--------|----------------|--------|
| $W_Q$  | `(64, 64)`     | 4,096  |
| $W_K$  | `(64, 64)`     | 4,096  |
| $W_V$  | `(64, 64)`     | 4,096  |
| $W_O$  | `(64, 64)`     | 4,096  |
| **Total** |             | **16,384** |

Note `(64, 64)` is `(embed_dim, num_heads × head_dim)` — the projection produces all $H$ heads' worth of output in one shot, then we reshape into `(B, T, num_heads, head_dim)`. With four layers we are spending $4 \times 16{,}384 = 65{,}536$ parameters just on attention projections.

Two empirical facts about real models point a way to shrink that:

1. **Query heads need to be diverse.** Each query head learns a different "what am I looking for?" question. Cutting them down (multi-query attention, MQA) measurably hurts modelling quality.
2. **Key/value heads can be shared.** Two query heads can share the same K/V representation — they ask different questions of the same answers. Llama-2 70B, Mistral, and most modern open-weight models use $G = 8$ query heads per KV head with no measurable quality drop versus full multi-head.

GQA is the middle ground: keep all `num_heads` query heads, use `num_kv_heads < num_heads` key/value heads, and **repeat** each KV head $G = \text{num\_heads} / \text{num\_kv\_heads}$ times before the dot-product. The MHA case ($G = 1$, i.e. `num_kv_heads = num_heads`) and the MQA case ($G = \text{num\_heads}$, i.e. `num_kv_heads = 1`) are both special cases of GQA.

---

## 26.3 The shapes, drawn

At `num_heads = 4`, `num_kv_heads = 2`, `head_dim = 16`, sequence length $T$:

```text
Q : (B, num_heads,    T, head_dim)   = (B, 4, T, 16)
K : (B, num_kv_heads, T, head_dim)   = (B, 2, T, 16)
V : (B, num_kv_heads, T, head_dim)   = (B, 2, T, 16)

(repeat K and V along the heads axis by G = num_heads / num_kv_heads = 2)

K': (B, num_heads,    T, head_dim)   = (B, 4, T, 16)
V': (B, num_heads,    T, head_dim)   = (B, 4, T, 16)

scores = Q @ K'.transpose(-2, -1) / sqrt(head_dim)   # (B, 4, T, T)
out    = softmax(scores) @ V'                        # (B, 4, T, 16)
```

The repeat is **interleaved**: KV head 0 serves query heads 0 and 1; KV head 1 serves query heads 2 and 3. (Other layouts exist; the interleaved one composes cleanly with `torch.repeat_interleave` and matches Llama's reference implementation.)

The point: every other line of attention is **unchanged** from MHA. Only the K/V projections shrink and the K/V tensors get repeated before they meet $Q$.

---

## 26.4 `repeat_interleave`, by hand

Skip ahead and verify the repeat in isolation. Open a Python REPL:

```bash
uv run python
```

Then:

```python
import torch
K = torch.tensor([[1., 1., 1., 1.],
                  [2., 2., 2., 2.]])      # 2 KV heads, head_dim=4
print(K.repeat_interleave(2, dim=0))      # repeat along the heads axis, 2 copies each
```

Expected output:

```text
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
```

Read each row as one head's view of the sequence. The two `[1, 1, 1, 1]` rows are query heads 0 and 1, both reading from KV head 0. The two `[2, 2, 2, 2]` rows are query heads 2 and 3, both reading from KV head 1. That is GQA in one operation.

Press `Ctrl-D` to exit the REPL.

---

## 26.5 Threading `num_kv_heads` through attention, blocks, and GPT

The change in `MultiHeadAttention` is small: a new constructor parameter, a new validation check, a new instance attribute, narrower $W_K$ and $W_V$, a different reshape for K and V, and a `repeat_interleave` before the dot-product. Everything else stays identical to Ch.25.

📄 `src/mygpt/__init__.py` — replace the existing `MultiHeadAttention` class with this version:

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 64,
        dropout: float = 0.0,
        position_type: str = "learned",
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.position_type = position_type

        self.W_Q = nn.Linear(embed_dim, num_heads    * self.head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

        if position_type == "rope":
            cos, sin = precompute_rope_cache(self.head_dim, max_seq_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"input length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.position_type == "rope":
            Q = apply_rope(Q, self.rope_cos, self.rope_sin)
            K = apply_rope(K, self.rope_cos, self.rope_sin)

        if self.kv_repeat > 1:
            K = K.repeat_interleave(self.kv_repeat, dim=1)
            V = V.repeat_interleave(self.kv_repeat, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.causal_mask[:T, :T]
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_drop(self.W_O(out))
```

Three things to notice:

- The `if num_kv_heads is None: num_kv_heads = num_heads` line is the backward-compat hinge: every existing call site that does not pass `num_kv_heads` still gets full MHA.
- The `if self.kv_repeat > 1` guard around `repeat_interleave` is a small efficiency: when `num_kv_heads == num_heads`, `kv_repeat == 1` and the repeat would be a no-op. Skipping the call also keeps the default code path **bit-identical** to Ch.25's, which we will verify in §26.8.
- RoPE is applied to K **before** the repeat. That is mathematically equivalent to rotating after (rotation is a per-position operation; copying the rotated row does the same thing as rotating each copy) and is cheaper, since we only rotate the smaller `(B, num_kv_heads, T, head_dim)` tensor.

Now thread `num_kv_heads` through the two callers above `MultiHeadAttention`.

📄 `src/mygpt/__init__.py` — replace the existing `TransformerBlock` class with this version:

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        max_seq_len=64,
        dropout=0.0,
        norm_type="layer",
        position_type="learned",
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.norm_type = norm_type
        self.position_type = position_type
        self.ln1 = _make_norm(embed_dim, norm_type)
        self.mha = MultiHeadAttention(
            embed_dim, num_heads, max_seq_len, dropout,
            position_type=position_type, num_kv_heads=self.num_kv_heads,
        )
        self.ln2 = _make_norm(embed_dim, norm_type)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

📄 `src/mygpt/__init__.py` — replace the existing `GPT.__init__` (the `forward` is unchanged) with this version. The simplest way is to replace the whole class definition; copy the `forward` from Ch.25 unchanged:

```python
class GPT(nn.Module):
    """Full GPT-2-style decoder-only transformer with weight-tied head."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len=64, dropout=0.0, norm_type="layer",
                 position_type="learned", num_kv_heads=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.norm_type = norm_type
        self.position_type = position_type

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        if position_type == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        # If position_type == "rope", positions are applied inside attention;
        # no learned position embedding is allocated here.
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout,
                             norm_type, position_type, self.num_kv_heads)
            for _ in range(num_layers)
        ])
        self.ln_f = _make_norm(embed_dim, norm_type)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"input length T={T} exceeds max_seq_len={self.max_seq_len}")
        x = self.token_embedding(ids)
        if self.position_type == "learned":
            positions = torch.arange(T, device=ids.device)
            x = x + self.position_embedding(positions)
        x = self.embed_drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = x @ self.token_embedding.embedding.weight.T
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
```

The ladder is the same as Ch.24 and Ch.25: a new optional argument with a default that *means* "Part-I behaviour", carried verbatim through `GPT → TransformerBlock → MultiHeadAttention`.

---

## 26.6 Checkpoint format adds `num_kv_heads`

Persist the choice. Pre-Ch.26 checkpoints fall back to the Part-I default (`num_kv_heads = num_heads`).

📄 `src/mygpt/__init__.py` — replace the existing `save_checkpoint` and `load_checkpoint` with these versions:

```python
def save_checkpoint(model: "GPT", tokenizer: "CharTokenizer", path: str) -> None:
    """Bundle model weights, tokenizer, and architecture into one .ckpt file."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_chars": tokenizer.chars,
            "config": {
                "vocab_size":     model.vocab_size,
                "embed_dim":      model.embed_dim,
                "num_heads":      model.num_heads,
                "num_kv_heads":   getattr(model, "num_kv_heads", model.num_heads),
                "num_layers":     model.num_layers,
                "max_seq_len":    model.max_seq_len,
                "norm_type":      getattr(model, "norm_type", "layer"),
                "position_type":  getattr(model, "position_type", "learned"),
            },
        },
        path,
    )


def load_checkpoint(path: str) -> tuple["GPT", "CharTokenizer"]:
    """Reload a (model, tokenizer) pair from a checkpoint produced by `save_checkpoint`.

    Always loads to CPU; the caller is responsible for `.to(device)` afterwards.
    Pre-Ch.24 checkpoints have no `norm_type` field; pre-Ch.25 checkpoints have
    no `position_type` field; pre-Ch.26 checkpoints have no `num_kv_heads` field.
    All three default to their original behaviour (``"layer"``, ``"learned"``,
    and ``num_kv_heads = num_heads``) so old `.ckpt` files continue to load.
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
    model = GPT(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads", config["num_heads"]),
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,
        norm_type=config.get("norm_type", "layer"),
        position_type=config.get("position_type", "learned"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer
```

---

## 26.7 The `--num-kv-heads` CLI flag

Add the flag, print it, and pass it through.

📄 `src/mygpt/__init__.py` — inside `_train_command`, replace the `model = GPT(...)` block with this version (it inserts a `num_kv_heads` resolution step and threads it through):

```python
    set_seed(0)
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else args.num_heads
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        norm_type=args.norm,
        position_type=args.position,
    ).to(device)
```

📄 `src/mygpt/__init__.py` — inside `_train_command`, replace the two `print(...)` lines for `norm:` and `position:` with these four:

```python
    print(f"norm:         {args.norm}")
    print(f"position:     {args.position}")
    print(f"num_heads:    {args.num_heads}")
    print(f"num_kv_heads: {num_kv_heads}")
```

📄 `src/mygpt/__init__.py` — inside `main()`, just after the `--position` argument is added, append the new flag:

```python
    p_train.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of K/V heads for grouped-query attention. Default: same as --num-heads (full MHA, Ch.8). Must divide --num-heads.",
    )
```

Verify the CLI parses:

```bash
uv run mygpt train --help | tail -5
```

Expected output (last few lines):

```text
  --num-kv-heads NUM_KV_HEADS
                        Number of K/V heads for grouped-query attention.
                        Default: same as --num-heads (full MHA, Ch.8). Must
                        divide --num-heads.
```

---

## 26.8 Backward-compat: defaults still reproduce Ch.25

Train with the defaults — same command as Ch.25's §25.8 backward-compat run. Because the default code path is `num_kv_heads = num_heads`, `kv_repeat = 1`, and the `repeat_interleave` is skipped, the operations executed are identical to Ch.25's. The loss curve must be **bit-identical** to the Ch.21/24/25 default.

```bash
uv run mygpt train tinyshakespeare.txt --device mps --output sh-mha.ckpt
```

Expected output:

```text
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

The `params: 207,296` and the loss numbers match every prior chapter's default run exactly. (The two new print lines `num_heads:` and `num_kv_heads:` are the only difference from the Ch.25 default output.)

---

## 26.9 The GQA run

Now train with two K/V heads instead of four:

```bash
uv run mygpt train tinyshakespeare.txt --device mps --num-kv-heads 2 --output sh-gqa.ckpt
```

Expected output (selected lines):

```text
norm:         layer
position:     learned
num_heads:    4
num_kv_heads: 2
…
params:       190,912
…
step     1: loss = 41.1659
step   500: loss = 2.6111
step  1000: loss = 2.4060
step  1500: loss = 2.2052
step  2000: loss = 2.0854

saved checkpoint to sh-gqa.ckpt
```

Two things to check:

1. **`params: 190,912`.** That is exactly $207{,}296 - 16{,}384$. Where do those 16,384 parameters come from? Each of the 4 attention layers loses $W_K$ and $W_V$ rows — going from `(64, 64)` to `(64, 32)` shrinks each by 2,048 params:

    $$
    \underbrace{4 \text{ layers}}_{} \times \underbrace{2}_{W_K + W_V} \times (4{,}096 - 2{,}048) = 16{,}384.
    $$

    The Q and O projections, every norm, every MLP, the embeddings, and the LM head are all unchanged.

2. **`step 2000: loss = 2.0854`.** Compared to MHA's 2.0785 at the same step, the GQA model is just 0.007 nats worse — a fraction of a percent of the loss range. With **8.0% fewer parameters**, the model trained to essentially the same point. This is the characteristic GQA result: KV heads are highly redundant, and sharing them across query groups costs almost nothing in modelling quality.

The story holds (and gets stronger) at scale. Llama-2 70B uses $G = 8$ — eight query heads per KV head — and ships with no measurable quality loss versus a hypothetical full-MHA Llama-2 70B.

---

## 26.10 Where the inference-time payoff lives

Training-time savings on the toy model are 8% of parameters — pleasant but not the headline. The headline is at *inference*.

When `mygpt generate` produces a long sequence, every generated token has to attend back over every previously generated token. Production inference engines reuse work by storing a **KV cache**: keep the K and V tensors for every position you have already seen, append one new row per generated step, and never recompute. The size of that cache scales with `num_kv_heads × head_dim × T_total`. Halving `num_kv_heads` halves the KV cache.

For `mygpt` with `T_total = 64` this is a few kilobytes either way and you will never notice. For a Llama-2 70B serving 8K tokens of context across 80 attention layers, the KV cache is the dominant memory cost of inference. GQA is the difference between fitting a long-context conversation in 32 GB of VRAM and not fitting at all.

We do not implement KV caching in `mygpt` — caching is an inference-engineering concern, and the code change is large enough to belong in its own chapter (a Part III topic). What matters here is the architectural prerequisite: by the time you build a KV cache, the model must already have `num_kv_heads < num_heads`, otherwise you are caching redundant tensors. GQA is what makes the cache savings real.

---

## 26.11 Sampling from each model

```bash
uv run mygpt generate --checkpoint sh-mha.ckpt --prompt "ROMEO:" --device cpu
```

Expected output (last lines after the device line):

```text
ROMEO:
Thy momed has seltered, a neark'ly your tle centeloourse.
Of therere hath thin beielly saneer best.

BRINCE:
Bucker I to my yet, tronen my bety sevene you for mad, bendoth,
Whe a bros swencurenty hou
```

This is the byte-exact Ch.17 §17.6 sample — every prior backward-compat checkpoint produces this same continuation. (Backward compat is mechanical, not stylistic: the same training run produces the same checkpoint produces the same sample.)

```bash
uv run mygpt generate --checkpoint sh-gqa.ckpt --prompt "ROMEO:" --device cpu
```

Expected output:

```text
ROMEO:
Thy momed haveseltered ad meament, ink helsterere
Doues this therere hath thigh orell seaneer brie.

BRIO:
Whis bee ancesendy thowall a my be tooe spe st to alin hisere,
By bres han shup the shere wa
```

The first 14 characters (`ROMEO:\nThy momed `) are byte-identical to the MHA sample — both models, evaluated in low-entropy contexts with the same `top_k`, picked the same token. They diverge after `momed `. The GQA model's output is qualitatively the same kind of pseudo-Shakespeare as the MHA model's; the architectures are interchangeable at this scale.

---

## 26.12 Experiments

**Experiment 1 — confirm the saved config.** Inspect the GQA checkpoint:

```bash
uv run python -c "
import torch
ckpt = torch.load('sh-gqa.ckpt', weights_only=False)
print(ckpt['config'])
"
```

You should see:

```text
{'vocab_size': 65, 'embed_dim': 64, 'num_heads': 4, 'num_kv_heads': 2, 'num_layers': 4, 'max_seq_len': 64, 'norm_type': 'layer', 'position_type': 'learned'}
```

The MHA checkpoint shows `'num_kv_heads': 4`. Pre-Ch.26 checkpoints have no `num_kv_heads` field at all and would fall back to `config["num_heads"]` on load.

**Experiment 2 — multi-query attention (MQA).** Train with `--num-kv-heads 1`. Param count: 190,912 − 4 × (2,048 − 1,024) − 4 × (2,048 − 1,024) = 182,720 (one $W_K$ and one $W_V$ per layer drop from `(64, 32)` to `(64, 16)`). This is the most aggressive sharing — one KV head shared across all four query heads. Compare the final loss to GQA-2 and MHA. At `embed_dim = 64` you should see a measurable loss bump going from GQA-2 to MQA; at large scale Llama's authors report the bump is too large to accept, which is exactly why GQA was published as a middle ground.

**Experiment 3 — pick a non-divisor.** `--num-kv-heads 3`. The CLI accepts the int but `MultiHeadAttention.__init__` raises:

```text
ValueError: num_heads (4) must be divisible by num_kv_heads (3)
```

This is the validation in `__init__.py`. Without it, `kv_repeat = 4 // 3 = 1` and the repeat would silently mis-shape the attention.

**Experiment 4 — the parameter-count formula.** Derive a closed form for the total `mygpt` parameter count as a function of `embed_dim`, `num_heads`, `num_kv_heads`, `num_layers`, `vocab_size`, and `max_seq_len`. Verify it predicts 207,296 (MHA), 190,912 (GQA-2), and 182,720 (MQA) given our defaults. You will need to count: token embedding (`vocab_size × embed_dim`), position embedding (`max_seq_len × embed_dim` if `position_type=learned`, else 0), per-layer attention ($W_Q$ is `embed_dim × num_heads × head_dim`, $W_O$ is `embed_dim × embed_dim`, $W_K$ and $W_V$ are each `embed_dim × num_kv_heads × head_dim`), per-layer MLP ($8 \times \text{embed\_dim}^2 + 5 \times \text{embed\_dim}$ — count `fc1` and `fc2` weights and biases), 9 norms (1 per `ln1` + 1 per `ln2` per layer + 1 final), and the LM head (tied, so 0 new params).

---

## 26.13 Exercises

1. The `if self.kv_repeat > 1:` guard in `MultiHeadAttention.forward` is not strictly necessary for correctness — `repeat_interleave(1, dim=1)` is a no-op. Remove the guard and re-run §26.8. The default loss curve should remain bit-identical (`repeat_interleave(1)` returns the same tensor). Re-add the guard afterwards: it is one branch saved per forward pass for the common Part-I case, and it documents intent.
2. The chapter applies RoPE to K **before** the repeat. Argue (informally) why applying RoPE *after* the repeat would produce the same numerical result. Why is "before" preferred in production code? (Hint: count multiplications.)
3. Sketch a KV cache for `mygpt`. What are the buffer shapes for MHA at our defaults (`num_heads = 4`, `head_dim = 16`, `max_seq_len = 64`, 4 layers)? For GQA-2? What is the byte savings at fp32?

---

## 26.14 What's next

`mygpt` now has every architectural piece of a modern open-weight LLM at toy scale: BPE tokenizer (Ch.23), RMSNorm (Ch.24), RoPE (Ch.25), and GQA (Ch.26). Chapter 27 stops adding flags and starts measuring: same parameter budget as Ch.17, modern recipe vs. baseline recipe, side by side, on the same Tiny Shakespeare corpus. The "aha" of Part II is that the modern stack — with no scale change — measurably beats the baseline.
