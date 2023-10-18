import argparse
import os
import torch
from torch.utils.data import DataLoader, distributed
from utils.config_utils import yaml_parser, setup_seed, init_setting, get_envs
from utils.optimizer_utils import get_parameter_number, build_optimizer, build_scheduler
from logger.logger import setup_logging, get_logger
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
from data.dataset import get_ds
from model.build_model import get_model, parallel_model, de_parallel
from loss.build_loss import get_loss
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import sys
from val import evaluate
from copy import deepcopy
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

    return parser.parse_args()


def train(cfg):
    # root dirs of the training experiment, ckps, tensorboard_logs, and logs(text)
    experiment_dir, checkpoints_dir, tensorboard_dir, log_dir = init_setting(cfg)
    setup_logging(save_dir=log_dir)
    setup_seed(2022)

    logger = get_logger(name="train")  # log message printing

    # setup enviroment
    if cfg.GLOBAL.DEVICE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif cfg.GLOBAL.DEVICE:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()

    cuda = cfg.GLOBAL.DEVICE != "cpu" and torch.cuda.is_available()  # bool
    device = torch.device("cuda:0" if cuda else "cpu")
    rank, local_rank, world_size = get_envs()
    logger.info(f"Using device: {device}")

    if local_rank != -1:  # DDP distributed mode
        logger.info("In DDP mode.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            init_method="env://",
            rank=local_rank,
            world_size=world_size,
        )

    tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(
        f'Start Tensorboard with "tensorboard --logdir={tensorboard_dir}", view at http://localhost:6006/'
    )

    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(cfg)

    logger.info(f"Training dataset created with {len(train_ds)} samples.")
    train_sampler = (
        None if rank == -1 else distributed.DistributedSampler(train_ds, shuffle=True)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.TRAIN.TRAIN_BATCHSIZE_PER_CARD // world_size,
        shuffle=True and train_sampler is None,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        sampler=train_sampler,
        drop_last=cfg.TRAIN.DROP_LAST,
    )
    logger.info(f"Training dataloader created with {len(train_loader)} batches.")

    if rank in [-1, 0]:
        logger.info(f"Validation dataset created with {len(val_ds)} samples.")
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.TRAIN.VAL_BATCHSIZE_PER_CARD,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            drop_last=cfg.TRAIN.DROP_LAST,
        )
        logger.info(f"Validation dataloader created with {len(val_loader)} batches.")

    net = get_model(
        cfg,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    ).to(device)

    logger.info(get_parameter_number(net))  # calculate how many parameters
    net = net.to(device)

    optimizer = build_optimizer(net, cfg, logger)  # create optimizer
    scheduler = build_scheduler(optimizer, cfg)  # create lr scheduler
    loss_function = get_loss(cfg, tokenizer=tokenizer_src)  # create loss function
    start_epoch = 0

    # IF RESUMING TRAINING
    if cfg.GLOBAL.RESUME:
        checkpoint = torch.load(cfg.GLOBAL.RESUME_PATH)
        logger.info("loading checkpoint from {}".format(cfg.GLOBAL.RESUME_PATH))
        start_epoch = checkpoint["epoch"]
        state_dict = checkpoint["state_dict_backbone"]
        net.load_state_dict(state_dict, strict=False)
        state_optimizer = checkpoint["state_optimizer"]
        optimizer.load_state_dict(state_optimizer)
        state_lr_scheduler = checkpoint["state_lr_scheduler"]
        scheduler.load_state_dict(state_lr_scheduler)

    # create parallel model
    net = parallel_model(net, device, rank, local_rank)

    scaler = GradScaler(enabled=cfg.GLOBAL.USE_AMP)  # Mixed Precision Training

    pre_ce_loss = 100.0  # starting loss
    early_stop_patience = 0

    for epoch in range(start_epoch, start_epoch + cfg.GLOBAL.EPOCH_NUM):
        net.train()
        if rank != -1:  # if in DDP mode
            train_loader.sampler.set_epoch(epoch)
        mean_loss = torch.zeros(1).to(device)  # initialzing mean loss

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.GLOBAL.USE_AMP):
                # extract data from the batch
                encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
                decoder_input = batch["decoder_input"].to(device)  # (b, seq_len)
                # encoder_mask : hide only the padding tokens
                encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)
                decoder_mask = batch["decoder_mask"].to(
                    device
                )  # (b, 1, seq_len, seq_len)

                # run the tensors through the transformer
                encoder_output = net.encode(
                    encoder_input, encoder_mask
                )  # (b, seq_len, d_model)
                decoder_output = net.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # (b, seq_len, d_model)
                proj_output = net.project(decoder_output)  # (b, seq_len, d_vocab)
                label = batch["label"].to(device)  # (b, seq_len)

                # compute the loss
                loss = loss_function(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                    label.view(-1),
                )

                # distributed synchronization and reduction of loss
                if rank != -1:
                    dist.all_reduce(loss)  # Synchronize and sum losses across GPUs
                    loss /= world_size  # Compute the mean loss
                    # loss *= world_size

                # running average of the loss over multiple mini-batches
                mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

                if step % cfg.GLOBAL.LOG_EPOCH_STEP == 0 and rank in [-1, 0]:
                    logger.info(
                        f"Epoch [{epoch + 1}/{cfg.GLOBAL.EPOCH_NUM}], Step[{step + 1}/{len(train_loader)}], Loss: {mean_loss.item():.4f}"
                    )

                if not torch.isfinite(loss):  # if NaN
                    logger.warning("WARNING: non-finite loss, ending training ", loss)
                    sys.exit(1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Saving training loss values & learning rates to tensorboard
        tb_writer.add_scalar(
            tag="training_loss",
            scalar_value=mean_loss.item(),
            global_step=epoch,
        )

        tb_writer.add_scalar(
            tag="learning_rate",
            scalar_value=optimizer.param_groups[0]["lr"],
            global_step=epoch,
        )

        tb_writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={"training_loss": mean_loss.item()},
            global_step=epoch,
        )

        scheduler.step()

        # VALIDATION (every VAL_EPOCH_STEP(s)) only ranks (-1 or 0)
        if epoch % cfg.GLOBAL.VAL_EPOCH_STEP == 0 and rank in [-1, 0]:
            ce_loss = evaluate(
                net, val_loader, tokenizer_src, tokenizer_tgt, cfg, device
            )

            logger.info(
                f"Validation Epoch: [{epoch + 1}/{cfg.GLOBAL.EPOCH_NUM}], Loss: {round(ce_loss, 3)}"
            )

            # Saving validation loss values to tensorboard
            tb_writer.add_scalar(
                tag="validation_loss",
                scalar_value=ce_loss,
                global_step=epoch,
            )

            tb_writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict={"validation_loss": ce_loss},
                global_step=epoch,
            )

        if epoch % cfg.GLOBAL.SAVE_EPOCH_STEP == 0 and rank in [-1, 0]:
            checkpoint = {
                "epoch": epoch,
                "state_dict_backbone": deepcopy(de_parallel(net)).state_dict(),
                "state_optimizer": optimizer.state_dict(),
                "state_lr_scheduler": scheduler.state_dict(),
            }  # save state dictionary
            torch.save(checkpoint, checkpoints_dir / f"model-{epoch}.pth")

            # if current validation NLL loss is lesser than previous NLL loss, save it
            if ce_loss < pre_ce_loss:
                torch.save(checkpoint, checkpoints_dir / "best.pth".format(epoch))
                logger.info("Model Saved.")
                pre_ce_loss = ce_loss
                early_stop_patience = 0
            else:
                early_stop_patience += 1
                if early_stop_patience > cfg.GLOBAL.EARLY_STOP_PATIENCE:
                    logger.info(
                        # "acc exceeds times without improvement, stopped training early"
                        "no decrease in cross entropy loss, stopped training early"
                    )

                    # destroy process
                    if world_size > 1 and rank == 0:
                        dist.destroy_process_group()

                    sys.exit(1)

    # destroy process
    if world_size > 1 and rank == 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()  # getting the input arguments
    cfg = yaml_parser(args.yaml)  # parse the configuration file
    train(cfg)
