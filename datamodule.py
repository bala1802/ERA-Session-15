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
        pass

    def get_all_sentences(self, ds, config):
        pass

    def get_or_build_tokenizer(self, ds, lang):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    pass