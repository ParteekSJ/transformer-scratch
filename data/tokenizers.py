from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from .dataset import BilingualDataset


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(tokenizer_path, ds, lang, logger):
    tokenizer_file = Path(tokenizer_path)
    if not Path.exists(tokenizer_file):
        # Create parent directories if they don't exist
        tokenizer_file.parent.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_file))
    else:
        if logger is not None:
            logger.info(f"{tokenizer_path} already exists.")
        else:
            print(f"{tokenizer_path} already exists.")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(args, logger=None):
    cache_dir = "./data/"
    ds_raw = load_dataset("opus_books", f"{args.src_lang}-{args.tgt_lang}", split="train", cache_dir=cache_dir)

    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(args.src_tokenizer_path, ds_raw, args.src_lang, logger)
    tokenizer_tgt = get_or_build_tokenizer(args.tgt_tokenizer_path, ds_raw, args.tgt_lang, logger)

    # Keep 90% for training, and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, args.src_lang, args.tgt_lang, args.src_seq_len
    )
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, args.src_lang, args.tgt_lang, args.tgt_seq_len)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][args.src_lang])
        tgt_ids = tokenizer_tgt.encode(item["translation"][args.tgt_lang])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    if logger is not None:
        logger.info(f"Max length of source sentence: {max_len_src}")
        logger.info(f"Max length of target sentence: {max_len_tgt}")
    else:
        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
