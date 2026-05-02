from mygpt.embedding import TokenEmbedding
from mygpt.mlp import MLP
from mygpt.norm import LayerNorm
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    ln = LayerNorm(C)
    mlp = MLP(embed_dim=C, dropout=0.0)
    ln.eval(); mlp.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    x_normed = ln(x)
    out = x + mlp(ln(x))   # GPT-2 pre-norm pattern: residual around mlp(ln(x))

    print(f"\nInput x       shape={tuple(x.shape)}")
    print(f"After LN(x)   shape={tuple(x_normed.shape)}")
    print(f"After residual+MLP+LN  shape={tuple(out.shape)}")
    print()

    # Per-token mean/std of normalised tensor
    print(f"LN(x) per-token means (4 positions): {ln(x).mean(dim=-1).flatten().tolist()}")
    print(f"LN(x) per-token stds  (4 positions): {ln(x).std(dim=-1, unbiased=False).flatten().tolist()}")
    print()

    n_te = sum(p.numel() for p in te.parameters())
    n_ln = sum(p.numel() for p in ln.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"TokenEmbedding parameters:   {n_te}")
    print(f"LayerNorm parameters:        {n_ln}  (= 2 * embed_dim)")
    print(f"MLP parameters:              {n_mlp}")
    print(f"Total parameters:            {n_te + n_ln + n_mlp}")
