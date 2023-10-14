from torch import nn
from .multihead_attention import MultiHeadAttentionBlock
from .feed_forward_block import FeedForwardBlock
from .residual_connection import ResidualConnection
from .layer_normalization import LayerNormalization

class DecoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,  # Masked MHA
        cross_attention_block: MultiHeadAttentionBlock,  # Cross MHA
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x : Input of the DECODER
        encoder_output : Output of the ENCODER
        src_mask: Mask Applied to ENCODER (to avoid special tokens)
        tgt_mask: Mask Applied to DECODER
        """

        # First Skip/Residual Connection (MaskedMHA & AddNorm)
        x = self.residual_connections[0](
            x=x,
            sublayer=lambda x: self.self_attention_block(
                q=x,  # Decoder Self Attention
                k=x,
                v=x,
                mask=tgt_mask,  # causal mask
            ),
        )

        # Second Skip/Residual Connection (CrossMHA & AddNorm)
        x = self.residual_connections[1](
            x=x,
            sublayer=lambda x: self.cross_attention_block(
                q=x,  # query from Decoder
                k=encoder_output,  # key from Encoder
                v=encoder_output,  # value from Encoder
                mask=src_mask,  # encoder mask to avoid special tokens.
            ),
        )

        # Third & Final Skip/Residual Connection (FeedForward & AddNorm)
        x = self.residual_connections[2](x=x, sublayer=self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

