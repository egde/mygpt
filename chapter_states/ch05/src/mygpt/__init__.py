"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 5 the package knows about: the running-example vocabulary,
how to convert tokens to id tensors (Chapter 3), how to seed PyTorch's RNG
(Chapter 4), and a TokenEmbedding module that maps id tensors to dense
vector tensors (this chapter).
"""


from mygpt.cli import main
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids
