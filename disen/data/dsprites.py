import pathlib
from typing import Union

import numpy
import torch

from . import dataset_with_factors, index_util


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
        self._runtime_test()

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
        strides = self._strides.tolist()
        indices = index_util.StridedIndices(
            index_util.skip_dim(self.n_factor_values, factor),
            index_util.skip_dim(strides, factor),
            value * strides[factor],
        )
        return torch.utils.data.Subset(self, indices)

    def _runtime_test(self) -> None:
        ff = self.fix_factor(2, 5)
        assert (ff[0][1] == torch.as_tensor([0, 0, 5, 0, 0])).all()
        assert (ff[12345][1] == torch.as_tensor([2, 0, 5, 1, 25])).all()
