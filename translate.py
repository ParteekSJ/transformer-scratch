import torch
import os
import argparse
from data.dataset import get_ds
from data.decoding_strategies import greedy_search_decode, beam_search_decode
from model.build_model import get_model
from utils.config_utils import yaml_parser
from constants import BASE_DIR


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--yaml",
        default=BASE_DIR + "config/train.yaml",
        type=str,
        help="configuration file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DDP parameter",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="This is a test.",
        help="Sentence you want to translate",
    )

    return parser.parse_args()


def translate(cfg):
    # root dirs of the training experiment, ckps, tensorboard_logs, and logs(text)
    sentence = cfg.GLOBAL.SENTENCE
    seq_len = cfg.GLOBAL.SEQ_LEN
    decoding_strategy = cfg.GLOBAL.DECODING_STRATEGY

    # setup enviroment
    if cfg.GLOBAL.DEVICE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif cfg.GLOBAL.DEVICE:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()

    cuda = cfg.GLOBAL.DEVICE != "cpu" and torch.cuda.is_available()  # bool
    device = torch.device("cuda:0" if cuda else "cpu")

    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(cfg)

    model = get_model(
        cfg,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    ).to(device)

    model = model.to(device)

    # Perform checks whether the sentence is a proper sentence or not.

    # translate the sentence
    model.eval()
    with torch.inference_mode():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        # Making the input sequence uniform length.
        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")]
                    * (seq_len - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)

        source_mask = (
            (source != tokenizer_src.token_to_id("[PAD]"))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )
        # Depending on DECODING STRATEGY
        if decoding_strategy == "beam":
            sentence = greedy_search_decode(
                model,
                source,
                source_mask,
                tokenizer_src,
                tokenizer_tgt,
                seq_len,
                device,
            )
        else:
            sentence = beam_search_decode(
                model,
                source,
                source_mask,
                tokenizer_src,
                tokenizer_tgt,
                seq_len,
                device,
            )
        print(f"Translated Sentence: \n\n{tokenizer_tgt.decode(sentence.tolist())}.")
        return tokenizer_tgt.decode(sentence.tolist())


if __name__ == "__main__":
    args = get_args()  # getting the input arguments
    cfg = yaml_parser(args.yaml)  # parse the configuration file
    translate(cfg)
