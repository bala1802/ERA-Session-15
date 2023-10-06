import torch
import warnings
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from datamodule import OpusBookDataModule
from litmodel import LightningTransformer
from config import get_config
cfg = get_config()


from model import build_transformer

if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  # torch.cuda.empty_cache()
  # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"

  pl.seed_everything(1, workers=True)
  data_module = OpusBookDataModule(cfg)
  tokenizer_src, tokenizer_tgt = data_module.setup(stage="None")
  model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        cfg["seq_len"],
        cfg["seq_len"],
        d_model=cfg["d_model"],
  )

  litmodel = LightningTransformer(model=model, learning_rate=cfg["lr"], tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, max_len=cfg["seq_len"], num_examples=2)


  trainer = pl.Trainer(
      max_epochs=10,
      deterministic=True,
      logger=True,
      enable_model_summary=False,
      log_every_n_steps=1,
      precision=16
  )
  trainer.fit(litmodel, data_module)
