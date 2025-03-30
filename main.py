import argparse
from models import build_transformer
from data import get_ds
from logger import setup_logger
import logging
from engine import train_one_epoch, validate_one_epoch
from utils import save_checkpoint
import torch
import random
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser("Transformer Training", add_help=False)
    parser.add_argument("--src_vocab_size", default=100, type=int)
    parser.add_argument("--tgt_vocab_size", default=100, type=int)
    parser.add_argument("--src_seq_len", default=350, type=int)
    parser.add_argument("--tgt_seq_len", default=350, type=int)
    parser.add_argument("--src_lang", default="en", type=str)
    parser.add_argument("--tgt_lang", default="it", type=str)
    parser.add_argument(
        "--d_model", default=512, type=int, help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="Intermediate size of the feedforward layers")
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--save_freq", default=6, type=int, help="How often to save a checkpoint?")
    parser.add_argument("--src_tokenizer_path", default="./tokenizer/en.json", type=str)
    parser.add_argument("--tgt_tokenizer_path", default="./tokenizer/it.json", type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    # parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--print_freq", default=10, type=int, help="how often to print training status")
    # parser.add_argument("--pre_norm", action="store_true")

    # dataset parameters
    parser.add_argument("--device", default="cpu", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval", action="store_true")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transformer Training", parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize logger
    logger = setup_logger(
        name="model_training", log_dir=args.checkpoint_dir, console_level=logging.INFO, file_level=logging.DEBUG
    )

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(args, logger)
    args.src_vocab_size = tokenizer_src.get_vocab_size()
    args.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    model = build_transformer(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-9, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)

    train_losses, train_accuracies = [], []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            device,
            tokenizer_tgt,
            epoch,
            logger,
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}")
        # print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        # print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, args, train_loss, name="ckpt")
            logger.info(f"[save_freq] Checkpoint Saved.")

        if train_loss == min(train_losses) and train_acc == max(train_accuracies):
            save_checkpoint(model, optimizer, epoch, args, train_loss, name="best_ckpt")
            logger.info(f"[lowest training loss] Checkpoint Saved.")

        if args.eval:
            cer, wer, bleu = validate_one_epoch(model, val_dataloader, device, tokenizer_src, tokenizer_tgt)
            logger.info(f"Epoch {epoch+1}/{args.epochs} | CER: {cer:.4f} | WER: {wer:.4f} | BLEU: {bleu:.4f}")

    print("> THE END <")
