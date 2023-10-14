from torch import nn
from .multihead_attention import MultiHeadAttentionBlock
from .feed_forward_block import FeedForwardBlock
from .residual_connection import ResidualConnection
from .layer_normalization import LayerNormalization

"""
- Each of the Encoder Block is repeated 'N' times where the output of 
encoder block N-1 is sent to encoder block N. 
- Output of encoder block 'N' is sent to the decoder.
"""


class EncoderBlock(nn.Module):
    """Encoder Block -> 1 MHA, 2 Add&Norm, 1 FF"""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # q,k,v from input 'x'
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        """
        src_mask: Mask applied to ENCODER INPUT. This is to avoid the interaction
        of [PAD] tokens with other tokens.
        """

        # FIRST SKIP CONNECTION
        x = self.residual_connections[0](
            x=x,
            sublayer=lambda x: self.self_attention_block(
                q=x,
                k=x,
                v=x,
                mask=src_mask,
            ),
        )

        # SECOND SKIP CONNECTION
        x = self.residual_connections[1](
            x=x,
            sublayer=self.feed_forward_block,
        )

        return x  # Single Encoder Block Output


class Encoder(nn.Module):
    """List of Encoder Blocks"""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # Iterating through Encoder Blocks
        for layer in self.layers:
            # output of the N-1'th encoder block becomes the input
            # for N'th encoder block
            x = layer(x, mask)
        return self.norm(x)  # THIS IS FED TO THE DECODER
