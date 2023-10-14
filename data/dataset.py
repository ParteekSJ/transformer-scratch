from torch.utils.data import Dataset, random_split
import torch
from .tokenizer import get_or_build_tokenizer
from datasets import load_dataset  # hugging face thing.


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return 20
        # return len(self.ds)  # length from hugging face

    def __getitem__(self, index):
        src_target_pair = self.ds[index]

        # extracting the text from the src_tgt pair
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # padding such that we have uniform examples
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # SOS, EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # SOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        # Add SOS and EOS to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # Add SOS to target text
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # true label for the given example
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # encoder mask - not allow contribution from PAD tokens.

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(size=decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    # returns an upper triangular matrix
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def get_ds(config):
    # loading the datset from hugging face.
    ds_raw = load_dataset(
        "opus_books",
        f'{config.GLOBAL.LANG_SRC}-{config.GLOBAL.LANG_TGT}',
        split="train",
    )

    # Build tokenizer for source and target language
    # TODO: Update where the tokenizer is saved.
    tokenizer_src = get_or_build_tokenizer(config, ds_raw)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, lang_src=False)

    # Keep 90% training and 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # randomly splits data using the given proportions
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.GLOBAL.LANG_SRC,
        config.GLOBAL.LANG_TGT,
        config.GLOBAL.SEQ_LEN
    )


    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.GLOBAL.LANG_SRC,
        config.GLOBAL.LANG_TGT,
        config.GLOBAL.SEQ_LEN
    )

    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config.GLOBAL.LANG_SRC]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config.GLOBAL.LANG_TGT]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max source text length: {max_len_src}")
    print(f"Max target text length: {max_len_tgt}")


    return train_ds, val_ds, tokenizer_src, tokenizer_tgt
