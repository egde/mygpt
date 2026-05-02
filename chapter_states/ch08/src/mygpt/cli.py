from mygpt.attention import MultiHeadAttention
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    mha = MultiHeadAttention(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0)
    mha.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = mha(x)

    print(f"\nToken ids shape:                {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):       {tuple(x.shape)}")
    print(f"MultiHeadAttention output shape: {tuple(out.shape)}")
    print()
    print(f"num_heads = {mha.num_heads}, head_dim = {mha.head_dim}, embed_dim = {mha.embed_dim}")

    n_te = sum(p.numel() for p in te.parameters())
    n_mha = sum(p.numel() for p in mha.parameters())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"MultiHeadAttention parameters:    {n_mha}")
    print(f"Total parameters:                 {n_te + n_mha}")
