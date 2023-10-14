from torch import nn
import torch
import math

# Positional ENCODING: Adding another vector to the input embedding to keep a track of
# the order of the tokens inside the sequence.


class PositionalEncoding(nn.Module):
    """Keep track of position of tokens inside the sequence."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Need seq_len number of d_model size vectors
        # [positional vector for each token]

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1) - column vector

        # Create a vector of shape (d_model); denominator
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)

        # Apply SINE to even indices \ [seq_len, 256]
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ** (2i / d_model))

        # Apply COSINE to odd indices \ [seq_len, 256]
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):  # x - [batch, seq_len, embedding_dim]
        # PE remains constant throughout and is not learned.
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)
