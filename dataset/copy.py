import numpy as np
import torch

class CopyDataset(torch.utils.data.Dataset):
    def __init__(self, length=40, n=10000, seed=1234):
        self.data = np.random.randint(0, 10, (n, length))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self._index_to_string = {i: str(i) for i in range(10)}
        self._string_to_index = {str(i): i for i in range(10)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def index_to_string(data):
        return [self._index_to_string[i] for i in data]

    def string_to_index(data):
        return [self._string_to_index[c] for c in data]
