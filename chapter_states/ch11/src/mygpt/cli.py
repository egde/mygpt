from mygpt.block import TransformerBlock
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C = len(VOCAB), 4
    te = TokenEmbedding(V, C)
    block = TransformerBlock(embed_dim=C, num_heads=2, max_seq_len=64, dropout=0.0)
    block.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    x = te(ids)
    out = block(x)

    print(f"\nToken ids shape:           {tuple(ids.shape)}")
    print(f"Embedded shape (B, T, C):  {tuple(x.shape)}")
    print(f"Block output shape:        {tuple(out.shape)}")
    print()
    print(block)
    print()

    n_te = sum(p.numel() for p in te.parameters())
    n_block = sum(p.numel() for p in block.parameters())
    print(f"TokenEmbedding parameters:   {n_te}")
    print(f"TransformerBlock parameters: {n_block}")
    print(f"Total parameters:            {n_te + n_block}")
