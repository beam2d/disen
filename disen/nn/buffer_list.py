import collections
from typing import Sequence, overload

import torch


class BufferList(torch.nn.Module, collections.abc.Sequence[torch.Tensor]):
    def __init__(self, buffers: Sequence[torch.Tensor]) -> None:
        super().__init__()
        self._len = len(buffers)
        for i, buffer in enumerate(buffers):
            self.register_buffer(self._get_buffer_name(i), buffer)

    def __len__(self) -> int:
        return self._len

    @overload
    def __getitem__(self, i: int) -> torch.Tensor:
        ...

    @overload
    def __getitem__(self, s: slice) -> list[torch.Tensor]:
        ...

    def __getitem__(self, i_or_s):
        if isinstance(i_or_s, int):
            if i_or_s >= self._len:
                raise IndexError
            return getattr(self, self._get_buffer_name(i_or_s))
        return [self[i] for i in range(i_or_s.indices(self._len))]

    def _get_buffer_name(self, i: int) -> str:
        return f"buf{i}"
