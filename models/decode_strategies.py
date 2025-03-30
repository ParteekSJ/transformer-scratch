import torch
from data.dataset import causal_mask


def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse
    encoder_output = model.encode(src, src_mask)

    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target (decoder input)
        dec_mask = causal_mask(decoder_input.size(1)).type_as(src_mask)

        # Calculate the output
        out = model.decode(encoder_output, src_mask, decoder_input, dec_mask)

        # Select the next token
        probs = model.project(out[:, -1])
        _, next_word = torch.max(probs, dim=1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(
    model,
    beam_size,
    src,
    src_mask,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(src, src_mask)

    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]  # (INPUT, SCORE)

    while True:
        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []
        for candidate, score in candidates:
            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

        # Build the candidate's mask
        candidate_mask = causal_mask(candidate.size(1)).type_as(src_mask).to(device)
        # Calculate output
        out = model.decode(encoder_output, src_mask, candidate, candidate_mask)
        # get next token probabilities
        prob = model.project(out[:, -1])
        # get the top k candidates
        topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)

        # Gathering and storing the multiple partial hypotheses.
        for i in range(beam_size):
            # for each of the top k candidates, get the token and its probability
            token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
            token_prob = topk_prob[0][i].item()

            # create a new candidate by appending the token to the current candidate
            new_candidate = torch.cat([candidate, token], dim=1)

            # Sum the log probabilities because the probabilities are in log space
            new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)

        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()
