from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from utils.dataset_utils import get_all_sentences
from constants import BASE_DIR


def get_or_build_tokenizer(config, ds, lang_src=True):
    if lang_src:
        tokenizer_path = Path(f"{BASE_DIR}/data_dir/{config.GLOBAL.LANG_SRC}")
    else:
        tokenizer_path = Path(f"{BASE_DIR}/data_dir/{config.GLOBAL.LANG_TGT}")

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,  # threshold for a word to appear in our vocab
        )

        if lang_src:
            tokenizer.train_from_iterator(
                iterator=get_all_sentences(ds, config.GLOBAL.LANG_SRC),
                trainer=trainer,
            )
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer.train_from_iterator(
                iterator=get_all_sentences(ds, config.GLOBAL.LANG_TGT),
                trainer=trainer,
            )
            tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer
