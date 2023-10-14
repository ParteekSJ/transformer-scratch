from torch import nn
import torch

class FeedForwardBlock(nn.Module):
    """Position Wise Feed Forward Network used in both Encoder & Decoder"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (b, seq_len, d_model) -> (b, seq_len, d_ff) -> (b, seq_len, d_model)
        x = self.linear_1(x)  # (batch, seq_len, d_model) --> (batch, seq_len, d_ff)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)  # (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return x

        # return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

