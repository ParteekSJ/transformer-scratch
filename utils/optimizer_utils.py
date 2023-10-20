import math
from torch.optim import Adam, lr_scheduler


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Format numbers with commas for better readability
    total_num_str = "{:,}".format(total_num)
    trainable_num_str = "{:,}".format(trainable_num)

    info = f"Total: {total_num_str} params, Trainable: {trainable_num_str} params"
    return info


def count_optimizer_parameters(optimizer):
    total_params = 0

    for param_group in optimizer.param_groups:
        params = param_group["params"]
        total_params += sum(p.numel() for p in params)

    return total_params


def build_scheduler(optimizer, cfg):
    epochs = cfg.GLOBAL.EPOCH_NUM

    if cfg.OPTIMIZER.LR_NAME == "none":
        # Return a scheduler with a constant learning rate
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: 1.0,
        )
    elif cfg.OPTIMIZER.LR_NAME == "linear_lr":
        lf = (
            lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg.OPTIMIZER.LR_DECAY)
            + cfg.OPTIMIZER.LR_DECAY
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lf,
        )
    elif cfg.OPTIMIZER.LR_NAME == "cosine_lr":
        lf = (
            lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2)
            * (1.0 - cfg.OPTIMIZER.LR_DECAY)
            + cfg.OPTIMIZER.LR_DECAY
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lf,
        )

    return scheduler


def build_optimizer(model, cfg, logger):
    if cfg.OPTIMIZER.NAME == "Adam":
        optimizer = Adam(params=model.parameters(), lr=cfg.OPTIMIZER.LEARNING_RATE)
        return optimizer
    return None


# def build_optimizer(model, cfg, logger):
#     """
#     g0 - store weights of batch normalization layers which do not have any weight decays
#     g1 - store weights (not from batch normalization layers) which will have weight decays
#     g2 - store biases
#     """

#     g0, g1, g2 = [], [], []  # optimizer parameter groups
#     total_params = 0  # Initialize total parameter count

#     for v in model.modules():
#         if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
#             g2.append(v.bias)
#         if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
#             g0.append(v.weight)
#         elif hasattr(v, "weight") and isinstance(
#             v.weight, nn.Parameter
#         ):  # weight (with decay)
#             g1.append(v.weight)

#     # Accumulate the total number of parameters
#     total_params = sum(p.numel() for group in [g0, g1, g2] for p in group)

#     if cfg.OPTIMIZER.NAME == "Adam":
#         optimizer = Adam(
#             g0,
#             lr=cfg.OPTIMIZER.LEARNING_RATE,
#             betas=[cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2],
#         )
#         optimizer.add_param_group(
#             {"params": g1, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}
#         )  # add g1 with weight_decay
#         optimizer.add_param_group({"params": g2})  # add g2 (biases)

#     elif cfg.OPTIMIZER.NAME == "SGD":
#         optimizer = SGD(
#             g0,
#             lr=cfg.OPTIMIZER.LEARNING_RATE,
#             momentum=cfg.OPTIMIZER.MOMENTUM,
#             nesterov=cfg.OPTIMIZER.NESTEROV,
#         )
#         optimizer.add_param_group(
#             {"params": g1, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}
#         )  # add g1 with weight_decay
#         optimizer.add_param_group({"params": g2})  # add g2 (biases)

#     # logger.info(
#     #     f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
#     #     f"g0: {len(g0)} weight, g1: {len(g1)} weight (no decay), g2: {len(g2)} bias"
#     # )
#     del g0, g1, g2

#     return optimizer
