import torch
from typing import Iterable
from models.transformer import Transformer
from models.decode_strategies import greedy_decode
import torchmetrics


def train_one_epoch(
    model: Transformer,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer_tgt,
    epoch: int,
    logger,
):
    train_losses = []
    train_accs = []

    for idx, batch in enumerate(data_loader):
        # EXTRACTING DATA FROM THE DATA LOADER
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        decoder_input = batch["decoder_input"].to(device)

        # FORWARD PASS OF THE TRANSFORMER
        enc_output = model.encode(src=encoder_input, src_mask=encoder_mask)
        dec_output = model.decode(
            enc_output=enc_output,
            src_mask=encoder_mask,
            tgt=decoder_input,
            tgt_mask=decoder_mask,
        )
        proj_output = model.project(dec_output)
        predictions = torch.argmax(proj_output, dim=-1)  # Shape: (batch_size, seq_len)
        predictions = predictions.cpu()

        # COMPUTING THE LOSS
        label = batch["label"].to(device)
        loss = criterion(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=tokenizer_tgt.get_vocab_size(),
            ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        )
        accuracy = accuracy_metric(predictions.view(-1), label.view(-1))  # Flatten for token-level accuracy

        train_losses.append(loss.item())
        train_accs.append(accuracy.item())

        if idx % 1 == 0:
            logger.info(f"EPOCH {epoch}, STEP [{idx}/{len(data_loader)}], LOSS: {loss:.4f}, ACCURACY: {accuracy:.4f}")

        # BACKPROPAGATING GRADIENTS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.mean(torch.Tensor(train_losses)).item(), torch.mean(torch.Tensor(train_accs)).item()


@torch.no_grad
def validate_one_epoch(
    model: Transformer,
    data_loader: Iterable,
    device: torch.device,
    tokenizer_src,
    tokenizer_tgt,
    num_examples=2,
):
    model.eval()
    count = 0

    src_texts = []
    expected = []
    predicted = []

    for batch in data_loader:
        count += 1
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        label = batch["label"].to(device)

        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer_src,
            tokenizer_tgt,
            max_len=350,
            device=device,
        )

        predictions = torch.argmax(model_out, dim=-1)  # Shape: (batch_size, seq_len)
        predictions = predictions.cpu()
        src_text = batch["src_text"][0]
        tgt_text = batch["tgt_text"][0]

        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        src_texts.append(src_text)
        expected.append(tgt_text)
        predicted.append(model_out_text)

        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)

        print("-" * 80)
        print(f"SOURCE: {src_text}")
        print(f"TARGET: {tgt_text}")
        print(f"PREDICTED: {model_out_text}")
        print("\n\n")

        if count == num_examples:
            print("-" * 60)
            break

        return cer, wer, bleu
