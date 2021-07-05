from typing import Sequence, overload

from .. import math_util


class StridedIndices(Sequence[int]):
    def __init__(
        self, shape: Sequence[int], strides: Sequence[int], offset: int
    ) -> None:
        super().__init__()
        self.shape, self.strides = _merge_contiguous_axes(shape, strides)
        self.offset = offset
        self.len = math_util.prod(self.shape)

    def __len__(self) -> int:
        return self.len

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, slc: slice) -> Sequence[int]: ...

    def __getitem__(self, index):
        # slice is not supported; Subset does not use it
        assert isinstance(index, int)
        pos = self.offset
        for n, s in zip(reversed(self.shape), reversed(self.strides)):
            pos += s * (index % n)
            index //= n
        assert index == 0
        return pos


def skip_dim(dims: Sequence[int], index: int) -> list[int]:
    return [*dims[:index], *dims[index + 1:]]


def _merge_contiguous_axes(
    shape: Sequence[int], strides: Sequence[int]
) -> tuple[list[int], list[int]]:
    new_shape: list[int] = []
    new_strides: list[int] = []
    extent = 0
    for d, s in zip(reversed(shape), reversed(strides)):
        if extent == s:
            new_shape[-1] *= d
        else:
            new_shape.append(d)
            new_strides.append(s)
            extent = s
    new_shape.reverse()
    new_strides.reverse()
    return new_shape, new_strides
