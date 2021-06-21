import math

import torch


ImageWithFactors = tuple[torch.Tensor, torch.Tensor]


class DatasetWithFactors(torch.utils.data.Dataset[ImageWithFactors]):
    """Dataset of images equipped with discrete generative factors."""

    n_factors: int
    n_factor_values: tuple[int, ...]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> ImageWithFactors:
        raise NotImplementedError

    def fix_factor(
        self, factor: int, value: int
    ) -> torch.utils.data.Subset[ImageWithFactors]:
        raise NotImplementedError

    def factor_entropies(self) -> torch.Tensor:
        """Compute [H(y_j)]_j."""
        return torch.as_tensor([math.log(k) for k in self.n_factor_values])
