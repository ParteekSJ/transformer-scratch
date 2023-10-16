# used to evaluate models based on their checkpoints.

import argparse
from utils.config_utils import yaml_parser
from logger.logger import get_logger
from model.build_model import get_model
from loss.build_loss import get_loss
import os
import torch
from data.decoding_strategies import greedy_search_decode, beam_search_decode
from data.dataset import causal_mask, get_ds
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description="val")
    parser.add_argument(
        "--yaml", default="config/train.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--model_path",
        default="train_log/2022-1018_1411/checkpoints/best.pth",
        type=str,
        help="output model name",
    )

    return parser.parse_args()


def evaluate(model, val_dl, tokenizer_src, tokenizer_tgt, cfg, device):
    loss_function = get_loss(cfg, tokenizer=tokenizer_src)  # create loss function
    mean_loss = torch.zeros(1).to(device)

    with torch.inference_mode():
        model.eval()
        # data_loader = tqdm(data_loader, file=sys.stdout)

        for step, batch in enumerate(val_dl):
            # extract data from the batch
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (b, seq_len)
            # encoder_mask : hide only the padding tokens
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (b, 1, seq_len, seq_len)

            # run the tensors through the transformer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (b, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (b, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (b, seq_len, d_vocab)
            label = batch["label"].to(device)  # (b, seq_len)

            # compute the loss
            loss = loss_function(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )

            # running average of the loss over multiple mini-batches
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        return mean_loss.item()


if __name__ == "__main__":
    args = get_args()
    cfg = yaml_parser(args.yaml)

    logger = get_logger(name="train")  # log message printing

    if cfg.GLOBAL.DEVICE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif cfg.GLOBAL.DEVICE:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()
    cuda = cfg.GLOBAL.DEVICE != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.TRAIN.VAL_BATCHSIZE_PER_CARD,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=cfg.TRAIN.DROP_LAST,
    )

    net = get_model(
        config=cfg,
        vocab_src_len=tokenizer_src.get_vocab_size(),
        vocab_tgt_len=tokenizer_tgt.get_vocab_size(),
    )
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint["state_dict_backbone"])
    net.to(device)

    cross_entropy_loss = evaluate(
        net, val_loader, tokenizer_src, tokenizer_tgt, cfg, device
    )
    
    print(f'VALIDATION MODEL LOSS -> {cross_entropy_loss}')