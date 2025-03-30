import os
import torch


# def save_checkpoint(model, optimizer, scheduler, epoch, args, train_loss):
def save_checkpoint(model, optimizer, epoch, args, train_loss, name):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "config": args.__dict__,
    }

    checkpoint_path = os.path.join(args.checkpoint_dir, f"{name}_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
