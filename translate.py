import torch
import os
import argparse
from data import get_ds
from models import build_transformer
from models.decode_strategies import greedy_decode, beam_search_decode


def get_args_parser():
    parser = argparse.ArgumentParser(description="Text Translate", add_help=False)
    parser.add_argument("--src_lang", default="en", type=str)
    parser.add_argument("--tgt_lang", default="it", type=str)
    parser.add_argument("--src_tokenizer_path", default="./tokenizer/en.json", type=str)
    parser.add_argument("--tgt_tokenizer_path", default="./tokenizer/it.json", type=str)
    parser.add_argument(
        "--sentence",
        type=str,
        default="This is a test.",
        help="Sentence you want to translate",
    )
    parser.add_argument("--src_seq_len", default=350, type=int)
    parser.add_argument("--tgt_seq_len", default=350, type=int)
    parser.add_argument("--decoding_strategy", type=str, default="greedy")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument(
        "--d_model", default=512, type=int, help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="Intermediate size of the feedforward layers")
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions."
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer.")
    parser.add_argument("--device", default="cpu", help="device to use for training / testing.")
    parser.add_argument("--ckpt_pretrained", default="", help="path to the pretrained checkpoint path.")
    return parser


def translate(args):
    # loading the tokenizer
    _, _, tokenizer_src, tokenizer_tgt = get_ds(args)
    args.src_vocab_size = tokenizer_src.get_vocab_size()
    args.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    # initializing a model
    model = build_transformer(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    # loading the pretrained checkpoint
    # TODO: Load model ckpt here!

    # translate the sentence
    model.eval()
    with torch.inference_mode():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(args.sentence)
        # Making the input sequence uniform length.
        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")] * (args.src_seq_len - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(args.device)

        source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(args.device)
        # Depending on DECODING STRATEGY
        if args.decoding_strategy == "beam":
            sentence = beam_search_decode(
                model,
                5,
                source,
                source_mask,
                tokenizer_src,
                tokenizer_tgt,
                args.src_seq_len,
                args.device,
            )
        else:
            sentence = greedy_decode(
                model,
                source,
                source_mask,
                tokenizer_src,
                tokenizer_tgt,
                args.src_seq_len,
                args.device,
            )

        print(f"Translated Sentence: \n\n{tokenizer_tgt.decode(sentence.tolist())}.")
        return tokenizer_tgt.decode(sentence.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transformer Training", parents=[get_args_parser()])
    args = parser.parse_args()

    translate(args)
