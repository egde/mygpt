from mygpt.embedding import TokenEmbedding
from mygpt.mlp import MLP
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    mlp = MLP(embed_dim=C, dropout=0.0)
    mlp.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = mlp(x)
    out_residual = x + mlp(x)

    print(f"\nToken ids shape:                {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):       {tuple(x.shape)}")
    print(f"MLP output shape:               {tuple(out.shape)}")
    print(f"x + MLP(x) shape (residual):    {tuple(out_residual.shape)}")
    print()
    print(f"hidden_dim = 4*C = {4*C}, embed_dim = {C}")

    n_te = sum(p.numel() for p in te.parameters())
    n_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"MLP parameters:                   {n_mlp}")
    print(f"Total parameters:                 {n_te + n_mlp}")
