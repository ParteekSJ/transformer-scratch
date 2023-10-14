from torch import nn

"""
- Final LINEAR LAYER after DECODER BLOCK expects the output -> dim (seq,d_model).
- Map the word embeddings (seq,d_model) back into the vocabulary (seq,vocab_size). 

LINEAR layer converts the embedding into a position in the vocabulary i.e., (index). 
This is called PROJECTION  layer as it projects embeddings to vocabulary.

Also apply a softmax to this as well to obtain a distribution over all the words in the vocabulary.
"""


class ProjectionLayer(nn.Module):
    """Linear Layer after Decoder Layer (with Softmax Included)"""

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()

        # (seq,d_model) -> (seq,vocab_size)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

    # Here, we used log_softmax for NUMERICAL STABILITY and this gives us a
    # distribution over all the probable words in the vocabulary.