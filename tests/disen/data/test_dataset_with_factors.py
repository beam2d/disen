import collections
import pytest
import torch

import disen


class SimpleDatasetWithFactors(disen.data.DatasetWithFactors):
    n_factors = 2
    n_factor_values = (3, 4)

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> disen.data.ImageWithFactors:
        factors = torch.as_tensor(divmod(index, 4))
        return torch.as_tensor(index), factors

    def get_index(self, factors: torch.Tensor) -> torch.Tensor:
        return factors @ torch.as_tensor([4, 1])

    def fix_factor(
        self, factor: int, value: int
    ) -> torch.utils.data.Subset[disen.data.ImageWithFactors]:
        start = value * [4, 1][factor]
        stride = [1, 4][factor]
        indices = list(range(start, len(self), stride))
        return torch.utils.data.Subset(self, indices)


@pytest.fixture
def dataset() -> SimpleDatasetWithFactors:
    return SimpleDatasetWithFactors()


def test_sample_stratified(dataset: SimpleDatasetWithFactors) -> None:
    sample = dataset.sample_stratified(8)
    indices = sample.indices
    indices0, indices1 = zip(*[divmod(i, 4) for i in indices])
    for idxs, n in zip([indices0, indices1], dataset.n_factor_values):
        assert len(idxs) == 8
        counter: collections.Counter[int] = collections.Counter()
        for i in idxs:
            counter[i] += 1
        low = 8 // n
        high = (8 - 1) // n + 1
        for i in counter:
            assert low <= counter[i] <= high
