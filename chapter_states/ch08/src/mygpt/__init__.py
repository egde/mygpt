"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 8 the package gains a MultiHeadAttention module that runs
num_heads single-head computations in parallel via tensor reshape, and
combines them through a final output projection.
"""


from mygpt.attention import MultiHeadAttention, SingleHeadAttention
from mygpt.cli import main
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids
