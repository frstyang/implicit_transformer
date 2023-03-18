import pickle
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

import torch
from tokenizers import Tokenizer
from datasets.samplers import LengthBasedSampler

class WikiText103(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        with open(f"cache/raw/wikitext-103-raw/wiki.{split}.raw", "rb") as f:
            wikitext = f.readlines()
        wikitext_processed = []
        for txt in wikitext:
            txt = txt.decode().strip()
            if len(txt) > 0:
                wikitext_processed.append(txt)
        with open(f"cache/wikitext-103/sentence_lengths.pkl", "rb") as f:
            self.sentence_lengths = pickle.load(f)[split]
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.tokenizer = Tokenizer.from_file("cache/wikitext-103/wikitext-103-tokenizer.json")
        self.sentences = wikitext_processed

        print(f"WikiText103 {split} split: {len(self.sentences)} sentences")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx][:128]

    def get_sampler(self, batch_size):
        return LengthBasedSampler(self, batch_size)

    def make_collate_fn(self):
        padding_idx = self.tokenizer.get_vocab_size()
        def collate_fn(sentences):
            self.tokenizer.enable_padding(pad_id=padding_idx, pad_token='<pad>')
            output_batch = self.tokenizer.encode_batch(sentences)
            return (
                torch.stack([torch.tensor(output.ids) for output in output_batch]).to(int),
                torch.tensor([sum(output.attention_mask) for output in output_batch]),
            )
        return collate_fn