import torch
from torch import nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        # # Each word in vocab will be embedded to 'd_model' dimensions
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        # (b, seq_len) --> (b, seq_len, d_model)
        return self.embedding_layer(x) * math.sqrt(self.d_model)  # Section 3.4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model); denominator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply SINE to even embedding indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply COSINE to even embedding indices
        pe = pe.unsqueeze(0)  # Add batch dimensionality

        pe.requires_grad = False  # absolute positional encoding

        self.register_buffer("pe", pe)  # Register the positional encoding as a buffer

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        # randomly sets elements of x to zero with probability 'p' and scales the remaining elements by '1/(1 - p)'
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps  # to avoid division by 0

        self.alpha = torch.nn.Parameter(data=torch.ones(1))  # multiplicative parameter
        self.bias = torch.nn.Parameter(data=torch.zeros(1))  # additive parameter

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Linear(d_model, d_ff)  # W1, b1
        self.W2 = nn.Linear(d_ff, d_model)  # W2, b2
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # (b, seq_len, d_model) -> (b, seq_len, d_ff) -> (b, seq_len, d_model)
        return self.W2(self.dropout(self.relu(self.W1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model  # embedding dimensionality.
        self.h = h  # number of heads.
        assert d_model % h == 0, f"{d_model} not divisible by {h}. Choose a different `h` parmater."

        self.d_k = d_model // h

        # Projection Matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def self_attn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # [B, h, seq_len, d_k] * [B, h, d_k, seq_len] = [B, h, seq_len, seq_len]
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn_scores.masked_fill_(mask=mask == 0, value=-1e-9)  # [B, h, seq_len, seq_len]

        attn_scores = attn_scores.softmax(dim=-1)  # [B, h, seq_len, seq_len]
        if dropout is not None:
            attn_scores = dropout(attn_scores)  # [B, h, seq_len, seq_len]

        # [B, h, seq_len, d_k], # [B, h, seq_len, seq_len]
        return (attn_scores @ value), attn_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        query = self.W_q(q)  # [b, seq_len, d_model]
        key = self.W_k(k)  # [b, seq_len, d_model]
        value = self.W_v(v)  # [b, seq_len, d_model]

        # [b, seq_len, d_model] => [b, seq_len, h, d_k] => [b, h, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attn_scores = MultiHeadAttention.self_attn(query, key, value, mask, self.dropout)

        # [b, h, seq_len, d_k] => [b, seq_len, h, d_k] => [b, seq_len, d_model]
        x = x.transpose(2, 1).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # [b, seq_len, d_model] -> [b, seq_len, d_model]
        return self.W_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        # pre-norm / post-norm can be edited here.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attn_block: MultiHeadAttention,
        feed_forward_block: FeedForwardLayer,
        dropout: float,
    ):
        super().__init__()

        self.self_attn_block = self_attn_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        x = self.residual_connections[0](x, sublayer=lambda x: self.self_attn_block(q=x, k=x, v=x, mask=src_mask))
        x = self.residual_connections[1](x, sublayer=self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: FeedForwardLayer,
        dropout: float,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x: self.self_attn(q=x, k=x, v=x, mask=tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attn(q=x, k=enc_output, v=enc_output, mask=src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.W_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # [B, seq_len, d_model] => [B, seq_len, vocab_size]
        return torch.log_softmax(input=self.W_proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbedding,
        tgt_embedding: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: PositionalEncoding,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(args):
    # Create Embedding Layers
    src_embed = InputEmbedding(d_model=args.d_model, vocab_size=args.src_vocab_size)
    tgt_embed = InputEmbedding(d_model=args.d_model, vocab_size=args.tgt_vocab_size)

    # Create the PositionalEncoding Layers
    src_pos = PositionalEncoding(d_model=args.d_model, seq_len=args.src_seq_len, dropout=args.dropout)
    tgt_pos = PositionalEncoding(d_model=args.d_model, seq_len=args.tgt_seq_len, dropout=args.dropout)

    # Create Encoder Blocks
    encoder_blocks = []
    for _ in range(args.enc_layers):
        encoder_self_attn = MultiHeadAttention(d_model=args.d_model, h=args.nheads, dropout=args.dropout)
        feedforward_block = FeedForwardLayer(d_model=args.d_model, d_ff=args.dim_feedforward, dropout=args.dropout)
        encoder_block = EncoderBlock(
            self_attn_block=encoder_self_attn, feed_forward_block=feedforward_block, dropout=args.dropout
        )
        encoder_blocks.append(encoder_block)

    # Create Decoder Blocks
    decoder_blocks = []
    for _ in range(args.dec_layers):
        decoder_self_attn = MultiHeadAttention(d_model=args.d_model, h=args.nheads, dropout=args.dropout)
        decoder_cross_attn = MultiHeadAttention(d_model=args.d_model, h=args.nheads, dropout=args.dropout)
        feedforward_block = FeedForwardLayer(d_model=args.d_model, d_ff=args.dim_feedforward, dropout=args.dropout)
        decoder_block = DecoderBlock(
            self_attn=decoder_self_attn,
            cross_attn=decoder_cross_attn,
            feed_forward=feedforward_block,
            dropout=args.dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    # Create the Projection Layer
    proj_layer = ProjectionLayer(d_model=args.d_model, vocab_size=args.tgt_vocab_size)

    # Create the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
