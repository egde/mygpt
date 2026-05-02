"""mygpt — a tiny GPT-2-like language model, built one chapter at a time.

After Chapter 9 the package gains an MLP module — the position-wise
feed-forward sub-block of a transformer. Together with the attention
sub-block and (in Chapter 10) layer norm, this is everything we need to
assemble a full transformer block in Chapter 11.
"""


from mygpt.attention import MultiHeadAttention, SingleHeadAttention
from mygpt.block import TransformerBlock
from mygpt.cli import main
from mygpt.embedding import TokenEmbedding
from mygpt.mlp import MLP
from mygpt.model import GPT
from mygpt.norm import LayerNorm
from mygpt.utils import VOCAB, set_seed, to_ids
