"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 6 the package knows about: the running-example vocabulary,
how to convert tokens to id tensors (Chapter 3), how to seed PyTorch's RNG
(Chapter 4), a TokenEmbedding module (Chapter 5), and a single-head
causal self-attention module (this chapter).
"""


from mygpt.attention import SingleHeadAttention
from mygpt.cli import main
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids
