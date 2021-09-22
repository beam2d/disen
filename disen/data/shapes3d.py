import logging
import pathlib
from typing import Union

import h5py
import numpy
import torch

from . import dataset_with_factors, index_util


_logger = logging.getLogger(__name__)


class Shapes3d(dataset_with_factors.DatasetWithFactors):
    n_factors = 6
    n_factor_values = (10, 10, 10, 8, 4, 15)

    def __init__(self, path: Union[pathlib.Path, str]) -> None:
        super().__init__()
        _logger.info("loading 3dshapes dataset...")

        d = h5py.File(path, "r")
        # Loading with numpy is faster... (numpy -> torch conversion is zero-copy)
        self._images = torch.from_numpy(numpy.asarray(d["images"]))

        # We assume that the factors are ordered lexicographically.
        factors: list[torch.Tensor] = []
        for k, n in enumerate(self.n_factor_values):
            arange = torch.arange(n)
            arange = arange.reshape(-1, *((1,) * (self.n_factors - k - 1)))
            arange = arange.expand(*self.n_factor_values)
            factors.append(arange)
        self._factors = torch.stack(factors, dim=-1).reshape(-1, self.n_factors)

        self._strides = (
            torch.as_tensor(self.n_factor_values[1:] + (1,)).flip(0).cumprod(0).flip(0)
        )

        self._runtime_test()

        _logger.info("=== loaded")

    def __len__(self) -> int:
        return self._images.shape[0]

    def __getitem__(self, index: int) -> dataset_with_factors.ImageWithFactors:
        image = self._images[index].permute(2, 0, 1) / 255
        factor = torch.as_tensor(self._factors[index])
        return image, factor

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
        assert (ff[0][1] == torch.as_tensor([0, 0, 5, 0, 0, 0])).all()
        assert (ff[23456][1] == torch.as_tensor([4, 8, 5, 6, 3, 11])).all()
