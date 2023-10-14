from torch import nn
import torch

class LayerNormalization(nn.Module):
    """Normalize the activations of each layer in a neural network"""

    def __init__(self, features: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps = eps  # to avoid division by 0
        self.alpha = nn.Parameter(torch.ones(features))  # multiplied
        self.bias = nn.Parameter(torch.zeros(features))  # added
        # allows the model to amplify values that it wants amplifies.

    def forward(self, x):
        """
        If y=(2,3): 2examples, 3features, then y_mean, y_std = (2,1)
        i.e., sample mean [mean of all features]

        """
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias