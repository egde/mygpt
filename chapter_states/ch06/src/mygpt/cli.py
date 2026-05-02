from mygpt.attention import SingleHeadAttention
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(vocab_size=V, embed_dim=C)
    attn = SingleHeadAttention(embed_dim=C, head_dim=C)

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)   # (1, T)
    x = te(ids)                                           # (1, T, C)
    out = attn(x)                                         # (1, T, C)

    print(f"\nToken ids shape:           {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):  {tuple(x.shape)}")
    print(f"Attention output (B, T, C): {tuple(out.shape)}")

    n_te = sum(p.numel() for p in te.parameters())
    n_attn = sum(p.numel() for p in attn.parameters())
    print(f"\nTokenEmbedding parameters:        {n_te}")
    print(f"SingleHeadAttention parameters:   {n_attn}")
    print(f"Total:                            {n_te + n_attn}")
