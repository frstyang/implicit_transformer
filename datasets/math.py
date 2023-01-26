from typing import Tuple, List
import torch
from torch.utils.data import Dataset
from .vocabulary import CharVocabulary

def load_file(path: str) -> Tuple[List[str], List[str]]:
    print(f"Loading {path}")
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    q = lines[::2]
    a = lines[1::2]
    assert len(q) == len(a)
    return q, a

class PlaceValueDataset(Dataset):
    def __init__(self, split='train'):
        q, a = [], []
        for difficulty in ['easy', 'medium', 'hard']:
            q_, a_ = load_file(f'cache/dm_math/mathematics_dataset-v1.0/train-{difficulty}/numbers__place_value.txt')
            q.extend(q_); a.extend(a_)
        interpolate_offset = len(q)
        q_, a_ = load_file('cache/dm_math/mathematics_dataset-v1.0/interpolate/numbers__place_value.txt')
        q.extend(q_); a.extend(a_)
        extrapolate_offset = len(q)
        q_, a_ = load_file('cache/dm_math/mathematics_dataset-v1.0/extrapolate/numbers__place_value_big.txt')
        q.extend(q_); a.extend(a_)
        self.q = q; self.a = a

        vocabulary = set()
        for question in q:
            vocabulary.update(set(question))

        for answer in a:
            vocabulary.update(set(answer))

        self.vocabulary = CharVocabulary(vocabulary)
        self.split_index = {'train': 0, 'interpolate': 1, 'extrapolate': 2}[split]
        self.offsets = [0, interpolate_offset, extrapolate_offset, len(q)]
        
    def __len__(self):
        i = self.split_index
        return self.offsets[i+1] - self.offsets[i]
    
    def __getitem__(self, idx):
        i = self.split_index
        idx = idx + self.offsets[i]
        q, a = self.q[idx], self.a[idx]
        return self.vocabulary(q), int(a)

    def collate_fn(self, samples):
        seq_lens = [len(q) for q, a in samples]
        max_len = max(seq_lens)
        pad_idx = len(self.vocabulary)
        x = []
        y = []
        for q, a in samples:
            x.append(q + (max_len - len(q))*[pad_idx])
            y.append(a)
        return torch.tensor(x), torch.tensor(y), torch.tensor(seq_lens)