import pytorch_lightning as pl
from dataset import BilingualDataset

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


'''
OpusBookDataModule - Class to construct the dataset.
#TODO
'''
class OpusBookDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item["translation"][lang]

    def get_or_build_tokenizer(self, ds, lang):
        tokenizer_path = Path(self.config["tokenizer_file"].format(lang))
        if Path.exists(tokenizer_path):
            return Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.get_all_sentences(ds=ds, lang=lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
            return tokenizer

    def setup(self, stage):

        ds_raw = load_dataset(
            "opus_books", f"{self.config['lang_src']}-{self.config['lang_tgt']}", split="train"
        )

        #Building the tokenizers
        self.tokenizer_src = self.get_or_build_tokenizer(ds=ds_raw, lang=self.config["lang_src"])
        self.tokenizer_tgt = self.get_or_build_tokenizer(ds=ds_raw, lang=self.config["lang_tgt"])

        #Split the train and validation dataset, 90% and 10%
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_raw, val_ds_raw])

        self.train_ds = BilingualDataset(
            train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"]
        )

        self.val_ds = BilingualDataset(
            val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"]
        )

        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = self.tokenizer_src.encode


        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    pass