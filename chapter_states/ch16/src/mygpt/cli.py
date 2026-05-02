from mygpt.model import GPT
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")

    set_seed(0)
    V, C, h, N = len(VOCAB), 4, 2, 2
    gpt = GPT(vocab_size=V, embed_dim=C, num_heads=h, num_layers=N,
              max_seq_len=64, dropout=0.0)
    gpt.eval()

    ids = to_ids(["I", "love", "AI", "!"]).unsqueeze(0)
    logits = gpt(ids)

    print(f"\nToken ids shape:  {tuple(ids.shape)}")
    print(f"Logits shape:     {tuple(logits.shape)}  (B, T, V)")
    print()

    n_te = sum(p.numel() for p in gpt.token_embedding.parameters())
    n_pe = sum(p.numel() for p in gpt.position_embedding.parameters())
    n_blocks = sum(p.numel() for p in gpt.blocks.parameters())
    n_ln_f = sum(p.numel() for p in gpt.ln_f.parameters())
    n_total = sum(p.numel() for p in gpt.parameters())
    print(f"Token embedding       (V*C):       {n_te:>5}")
    print(f"Position embedding (max_seq*C):    {n_pe:>5}")
    print(f"{N} TransformerBlocks  (N*228):     {n_blocks:>5}")
    print(f"Final LayerNorm       (2*C):        {n_ln_f:>5}")
    print(f"Tied head            (0 extra):     {0:>5}")
    print(f"Total parameters:                  {n_total:>5}")
