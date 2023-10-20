import torch


def get_loss(cfg, tokenizer=None):
    if cfg.CRITERION.NAME == "CE":
        loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.token_to_id("[PAD]"),
            label_smoothing=0.1,  # diffuses the probabilities making the model less sure
        )

    return loss_function
