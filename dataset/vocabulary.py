import re
from typing import List, Union, Optional, Dict, Any, Set

class CharVocabulary:
    def __init__(self, chars: Optional[Set[str]]):
        self.initialized = False
        if chars is not None:
            self.from_set(chars)

    def from_set(self, chars: Set[str]):
        chars = list(sorted(chars))
        self.to_index = {c: i for i, c in enumerate(chars)}
        self.from_index = {i: c for i, c in enumerate(chars)}
        self.initialized = True

    def __len__(self):
        return len(self.to_index)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "chars": set(self.to_index.keys())
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.from_set(state["chars"])

    def str_to_ind(self, data: str) -> List[int]:
        return [self.to_index[c] for c in data]

    def ind_to_str(self, data: List[int]) -> str:
        return "".join([self.from_index[i] for i in data])

    def _is_string(self, i):
        return isinstance(i, str)

    def __call__(self, seq: Union[List[int], str]) -> Union[List[int], str]:
        assert self.initialized
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if self._is_string(seq):
            return self.str_to_ind(seq)
        else:
            return self.ind_to_str(seq)

    def __add__(self, other):
        return self.__class__(set(self.to_index.values()) | set(other.to_index.values()))