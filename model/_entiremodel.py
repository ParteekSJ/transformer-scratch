#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:03:01 2023

@author: parteeksj
NOTES:
    - Embedding: Mapping b/w numbers and vectors. Maps the number to the same vectors everytime.
    - Check if seq_len == vocab_size ?
    - contiguous check?
    
    - https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html
    - https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
    
    
"""

import torch
from torch import nn
import math


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


class InputEmbeddings(nn.Module):
    """Takes the Input Sequence and converts it to Embeddings."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Each word in vocab will be embedded to 'd_model' dimensions
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


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


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        print("Initializing Transformer...")

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize parameters using Xavier Uniform Strategy
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
