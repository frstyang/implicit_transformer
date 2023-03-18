from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

import torch
# modified RandomSampler class from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py
class LengthBasedSampler:
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, batch_size: int, num_samples: Optional[int] = None,
     generator=None) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def noisy_order_by_length(self, generator):
        sentence_lengths  = self.data_source.sentence_lengths
        assert len(sentence_lengths) == len(self.data_source)
        noise = torch.normal(0, 2, size=(len(sentence_lengths),), generator=generator)
        noisy_lengths = torch.tensor(sentence_lengths) + noise
        ordering = torch.argsort(noisy_lengths)

        cut = int(torch.randint(len(sentence_lengths), (1,), generator=generator))
        #print(cut, len(sentence_lengths) - cut, round((len(sentence_lengths) - cut)/self.batch_size), self.batch_size*round((len(sentence_lengths) - cut)/self.batch_size))
        cut = int(len(sentence_lengths) - self.batch_size*round((len(sentence_lengths) - cut)/self.batch_size))
        #print(f"len(ordering[cut:]): {len(ordering[cut:])}, len(ordering[:cut]): {len(ordering[:cut])}")
        ordering = torch.cat((ordering[cut:], ordering[:cut]))
        return ordering

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        for _ in range(self.num_samples // n):
            yield from self.noisy_order_by_length(generator=generator)
        yield from self.noisy_order_by_length(generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples