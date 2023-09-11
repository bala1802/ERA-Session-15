import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDatset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64)
            ],
            dim=0
        )


        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label":label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_token_len": len(encoder_input),
            "decoder_token_len": len(decoder_input),
            "pad_token": self.pad_token
        }
    
    def casual_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return (mask==0)