import os
import pickle
import re
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

import numpy as np
import torch
from tokenizers import Tokenizer

# a Dataset class for the WikiText103 dataset.
class WikiText103(torch.utils.data.Dataset):
    def __init__(self, split='train', seq_length=384, overlap=50):
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.seq_length = seq_length
        with open(f"cache/raw/wikitext-103-raw/wiki.{split}.raw", "rb") as f:
            wikitext = f.read().decode()
        self.sentences = preprocess_wikitext(wikitext)
        self.tokenizer = get_tokenizer()
        self.sentence_lengths = get_sentence_lengths(split, self.tokenizer)
        split_long_sentences(self.sentences, self.sentence_lengths, seq_length, overlap, self.tokenizer)

        print(f"WikiText103 {split} split: {len(self.sentences)} sentences")

    def __len__(self):
        return len(self.sentences)

    # generates input text by taking the sentence at index idx and appending additional
    # sentences until seq_length tokens have been reached.
    def __getitem__(self, idx):
        tokens = []
        current_length = 0
        
        while len(tokens) < self.seq_length:
            sentence = self.sentences[idx]
            tokens.extend(self.tokenizer.encode(sentence).ids)
            idx = (idx + 1) % len(self.sentences)
        
        return torch.tensor(tokens[:self.seq_length])

    def random_sampler(self, num_samples):
        return torch.utils.data.RandomSampler(self, replacement=True, num_samples=num_samples)


def preprocess_wikitext(wikitext):
    # Replace article titles and subtitles with special tokens
    wikitext = re.sub(r'\n = (.*?) = \n \n', r' = \1 = ', wikitext)
    
    # Split the text into paragraphs or sentences
    paragraphs = wikitext.split('\n')
    sentences = []
    for paragraph in paragraphs:
        for sentence in re.split(r'(?<=[.!?]) +', paragraph):
            if sentence.strip():
                sentences.append(sentence)
    
    return sentences

def get_tokenizer():
    if os.path.exists("cache/wikitext-103/wikitext-103-tokenizer.json"):
        return Tokenizer.from_file("cache/wikitext-103/wikitext-103-tokenizer.json")
    print("cache/wikitext-103/wikitext-103-tokenizer.json does not exist, creating tokenizer")
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    wikitext_files = ['cache/raw/wikitext-103-raw/wiki.train.raw',
                     'cache/raw/wikitext-103-raw/wiki.valid.raw',
                     'cache/raw/wikitext-103-raw/wiki.test.raw']
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<unk>", "<eos>"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(wikitext_files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="$A <eos>",
        special_tokens=[
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.save("cache/wikitext-103/wikitext-103-tokenizer.json")
    return tokenizer

def get_sentence_lengths(split, tokenizer):
    if not os.path.exists("cache/wikitext-103/sentence_lengths.pkl"):
        print("cache/wikitext-103/sentence_lengths.pkl does not exist, generating sentence lengths for each split (may take a few min)")
        all_sentence_lengths = {}
        for split in ['valid', 'test']:
            with open(f"cache/raw/wikitext-103-raw/wiki.{split}.raw", "rb") as f:
                wikitext = f.read().decode()
            sentences = preprocess_wikitext(wikitext)
            sentence_lengths = [len(encoded_sentence) for encoded_sentence in tokenizer.encode_batch(sentences)]
            all_sentence_lengths[split] = sentence_lengths
        with open("cache/wikitext-103/sentence_lengths.pkl", "wb") as f:
            pickle.dump(all_sentence_lengths, f)

    with open("cache/wikitext-103/sentence_lengths.pkl", "rb") as f:
        return pickle.load(f)[split]

def split_long_sentences(sentences, sentence_lengths, seq_length, overlap, tokenizer):
    # split sentences that exceed seq_length into chunks
    # modifies sentences and sequence_lengths inplace
    i = 0
    while i < len(sentences):
        if sentence_lengths[i] > seq_length:
            tokens = tokenizer.encode(sentences[i])
            assert len(tokens) > seq_length
            del sentences[i]
            del sentence_lengths[i]
            start = 0
            while start < len(tokens):
                end = min(start + seq_length, len(tokens))
                chunk = tokens.ids[start:end]
                sentence_chunk = tokenizer.decode(chunk)
                sentences.insert(i, sentence_chunk)
                sentence_lengths.insert(i, len(chunk))
                start += seq_length - overlap
                i += 1
        else:
            i += 1