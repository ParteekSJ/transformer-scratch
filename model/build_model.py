from .input_embeddings import InputEmbeddings
from .transformer import Transformer
from .positional_encoding import PositionalEncoding
from .multihead_attention import MultiHeadAttentionBlock
from .encoder import Encoder, EncoderBlock
from .feed_forward_block import FeedForwardBlock
from .decoder import Decoder, DecoderBlock
from .projection_layer import ProjectionLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn


def build_transformer(
    config,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
) -> Transformer:
    
    d_model = config.MODEL.d_model
    N = config.MODEL.BLOCKS
    h = config.MODEL.HEADS
    dropout = config.MODEL.DROPOUT
    d_ff = config.MODEL.d_ff
    
    
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


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        config,
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config.GLOBAL.SEQ_LEN,
        tgt_seq_len=config.GLOBAL.SEQ_LEN,
    )

    return model


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def de_parallel(model):
    return model.module if is_parallel(model) else model


def parallel_model(model, device, rank, local_rank):
    # DDP mode
    ddp_mode = device.type != "cpu" and rank != -1
    if ddp_mode:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return model