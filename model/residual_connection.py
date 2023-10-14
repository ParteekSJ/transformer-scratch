from torch import nn
from .layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

        # Skip connection are between Add&Norm and Previous Layers

    def forward(self, x, sublayer):
        """
        x : Input before it goes to SubLayer (eg: Input to MHA)
        sublayer : Layer right before the Add&Norm (eg: MHA)

        + :  denotes the Add,
        self.norm : denotes the Norm;
        THEREFORE Add&Norm

        Add UNPROCESSED INPUT TO THE OUTPUT [PROCESSED INPUT] of the
        sublayer (with dropout applied).

        """

        INPUT = x

        # PRE-NORMALIZATION as opposed to POST-NORMALIZATION (in paper)
        PROCESSED_INPUT = self.dropout(sublayer(self.norm(x)))

        return INPUT + PROCESSED_INPUT
        # return x + self.dropout(sublayer(self.norm(x)))
