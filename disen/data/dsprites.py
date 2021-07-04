import pathlib
from typing import Sequence, Union, overload

import numpy
import torch

from . import dataset_with_factors


class _MidStridedIndices(Sequence[int]):
    def __init__(self, outer_len: int, mid_stride: int, inner_len: int) -> None:
        super().__init__()
        self.inner_len = inner_len
        self.mid_stride = mid_stride
        self.size = outer_len * inner_len

    def __len__(self) -> int:
        return self.size

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[int]:
        ...

    def __getitem__(self, index):
        # Subset only passes int
        assert isinstance(index, int)
        o, i = divmod(index, self.inner_len)
        return o * self.mid_stride + i


class DSprites(dataset_with_factors.DatasetWithFactors):
    n_factors = 5
    n_factor_values = (3, 6, 40, 32, 32)

    def __init__(self, path: Union[pathlib.Path, str]) -> None:
        super().__init__()
        d = numpy.load(path, mmap_mode="r")
        self._images = d["imgs"]
        self._factors = d["latents_classes"]
        self._strides = (
            torch.as_tensor(self.n_factor_values[1:] + (1,)).flip(0).cumprod(0).flip(0)
        )

    def __len__(self) -> int:
        return self._images.shape[0]

    def __getitem__(self, index: int) -> dataset_with_factors.ImageWithFactors:
        image = self._images[index, None].astype(numpy.float32)
        factor = self._factors[index, 1:]
        return (torch.as_tensor(image), torch.as_tensor(factor))

    def get_index(self, factors: torch.Tensor) -> torch.Tensor:
        return factors @ self._strides

    def fix_factor(
        self, factor: int, value: int
    ) -> torch.utils.data.Subset[dataset_with_factors.ImageWithFactors]:
        assert 0 <= factor < self.n_factors
        assert 0 <= value < self.n_factor_values[factor]
        outer_len = int(numpy.prod(self.n_factor_values[:factor]))
        mid_stride = int(numpy.prod(self.n_factor_values[factor:]))
        inner_len = int(numpy.prod(self.n_factor_values[factor + 1 :]))
        indices = _MidStridedIndices(outer_len, mid_stride, inner_len)
        return torch.utils.data.Subset(self, indices)
