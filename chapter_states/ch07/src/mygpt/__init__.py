"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 7 the SingleHeadAttention module uses register_buffer for
the causal mask (allocated once, in __init__) and adds dropout layers
on both the attention weights and the output projection.

With dropout=0 (the default), forward output is byte-for-byte identical
to the Chapter 6 version.
"""


from mygpt.attention import SingleHeadAttention
from mygpt.cli import main
from mygpt.embedding import TokenEmbedding
from mygpt.utils import VOCAB, set_seed, to_ids
