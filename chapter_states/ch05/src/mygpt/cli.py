from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids


def main() -> None:
    print("Vocabulary:", VOCAB)
    print(f"Vocabulary size V = {len(VOCAB)}")
    sample = list(VOCAB)
    ids = to_ids(sample)
    print(f"to_ids({sample}) = {ids}")
    set_seed(0)
    te = TokenEmbedding(vocab_size=len(VOCAB), embed_dim=4)
    print(f"\nTokenEmbedding(V={len(VOCAB)}, C=4):")
    print(te)
    print(f"params = {sum(p.numel() for p in te.parameters())}")
    print(f"\nte(ids) shape = {tuple(te(ids).shape)}")
