import functools
import operator

import torch

from . import dataset_with_factors, index_util


class SimpleDataset(dataset_with_factors.DatasetWithFactors):
    """Simple artificial dataset of factors."""

    def __init__(self, n_factor_values: tuple[int, ...]) -> None:
        super().__init__()
        self.n_factors = len(n_factor_values)
        self.n_factor_values = n_factor_values
        self.len = functools.reduce(operator.mul, n_factor_values)
        self.strides = (
            torch.as_tensor(n_factor_values[1:] + (1,)).flip(0).cumprod(0).flip(0)
        )

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> dataset_with_factors.ImageWithFactors:
        indices: list[int] = []
        for n in reversed(self.n_factor_values):
            indices.append(index % n)
            index //= n
        assert index == 0
        indices.reverse()
        factors = torch.as_tensor(indices)
        return factors.to(torch.float32), factors

    def get_index(self, factors: torch.Tensor) -> torch.Tensor:
        return factors @ self.strides

    def fix_factor(
        self, factor: int, value: int
    ) -> torch.utils.data.Subset[dataset_with_factors.ImageWithFactors]:
        strides = self.strides.tolist()
        indices = index_util.StridedIndices(
            index_util.skip_dim(self.n_factor_values, factor),
            index_util.skip_dim(strides, factor),
            value * strides[factor],
        )
        return torch.utils.data.Subset(self, indices)
