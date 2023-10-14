from torch import nn
import math


class MultiHeadAttentionBlock(nn.Module):
    """
    1. Takes the input X(seq,d_model) of encoder and use it 3 times (Q, K, V)
    2. Apply 3 linear projections via weight matrices W_q, W_k, W_v of
    shape (d_model,d_model)
    3. Results of the linear projections yields 3 matrices (Q',K',V') of
    size (seq,d_model)
    4. These Q',K',V' are split in 'h' (heads) smaller matrices
    5. THEY ARE SPLIT ALONG 'EMBEDDING' DIMENSION not SEQUENCE DIMENSION
    i.e., 512/h
    6. [IMP] Each head will have access to full sentence BUT a different
    part of embedding \ subset of the embedding.
    7. Attention is applied to EACH OF THE smaller matrices which
    yield 'h' smaller matrices i.e., softmax(QK^T/sqrt(d_k)) * V
    8. They're CONCATENTED on the EMEBDDING DIMENSION (resulting back in 512).
    9. Concatenated Matrix is multiplied with weight matrix W_O
    which results in matrix with dimension (seq,d_model) [same dim as input mat]
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h

        assert d_model % h == 0, "d_model is not divisible by h"

        # d_model // h => 512 // 8 = 64
        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod  # can call this method without having an instance of the class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # MHA hidden dimension (512//8 = 64)

        # (b,h,seq,d_k) @ (b,h,d_k,seq) -> (b,h,seq,seq) \ UNNORMALIZED
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Autoregressive Generation / Causal Mask
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Normalized Attention Scores i.e., Applying Softmax [Row Wise]
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (b,h,seq,seq) @ (b,h,seq,d_k) --> (b,h,seq,d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # mask-to avoid interaction of some tokens with other tokens
        # used for ENCODER [avoid PAD tokens]
        # used for DECODER [autoregressive generation + avoid PAD tokens]

        query = self.w_q(q)  # Q' (b, seq, d_model) -> (b, seq, d_model)
        key = self.w_k(k)  # K' (b, seq, d_model) -> (b, seq, d_model)
        value = self.w_v(v)  # V' (b, seq, d_model) -> (b, seq, d_model)

        # DIVIDING Q', K', V' into smaller matrices i.e., MHA. They're divided
        # on the EMBEDDING DIMENSION.

        # RESHAPING: (batch, seq, d_model) -> (batch, seq, num_heads, d_k)
        # (b, seq_len, d_model) --> (b, seq_len, h, d_k) --> (b, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        """
        [IMP] (batch, num_heads(h), seq, d_k)  - Each head will see the full 
        sequence but a SUBSECTION of the total embeddings. 
        """

        # Calculate attention i.e., x=(attention_scores @ value)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        """ CONCATENATE ALL ATTENTION HEADS """
        # (b,h,seq,d_k) -> (b,seq,h,d_k) -> (b,seq,d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # When you use -1 as a dimension value, PyTorch automatically infers
        # the appropriate size based on the other dimensionS

        # OUTPUT ATTENTION MATRIX
        # (b,seq,d_model) @ (d_model, d_model)  -> (b,seq,d_model)
        return self.w_o(x)
