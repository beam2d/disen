import random

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

    def get_index(self, factors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fix_factor(
        self, factor: int, value: int
    ) -> torch.utils.data.Subset[ImageWithFactors]:
        raise NotImplementedError

    def factor_entropies(self) -> torch.Tensor:
        """Compute [H(y_j)]_j."""
        return torch.as_tensor(self.n_factor_values).log()

    def sample_stratified(
        self, sample_size: int
    ) -> torch.utils.data.Subset[ImageWithFactors]:
        """Subsample the dataset by balancing each factor.

        For each factor, the subsample contains the (almost) same number of frequencies
        of realizations.
        """
        factors = torch.empty((sample_size, self.n_factors), dtype=torch.int64)
        for j, n in enumerate(self.n_factor_values):
            reps = sample_size // n
            sample = list(range(n)) * reps
            sample += random.sample(range(n), sample_size - len(sample))
            random.shuffle(sample)
            factors[:, j] = torch.as_tensor(sample)
        indices = self.get_index(factors).tolist()
        return torch.utils.data.Subset(self, indices)


class DatasetWithCommonFactor(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    """Sampling a set of data points with a random factor fixed.

    Each example has the form ``(images, j)`` where ``images`` is a set of groups of
    data points. Each group consists of images with the same value of the j-th factor.
    Different groups have distinctly sampled j-th factor values. ``j`` is uniformly
    sampled from the set of generative factors.
    """

    def __init__(
        self,
        base: DatasetWithFactors,
        n_groups: int,
        n_images_in_group: int,
        n_iters: int,
    ) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.n_images_in_group = n_images_in_group
        self.n_iters = n_iters

        self.subsets = [
            [base.fix_factor(j, yj) for yj in range(base.n_factor_values[j])]
            for j in range(base.n_factors)
        ]

    def __len__(self) -> int:
        return self.n_iters

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = random.randrange(len(self.subsets))
        data_j = self.subsets[j]
        images: list[torch.Tensor] = []
        for _ in range(self.n_groups):
            data_yj = data_j[random.randrange(len(data_j))]
            for i in random.sample(range(len(data_yj)), self.n_images_in_group):
                images.append(data_yj[i][0])
        xs = torch.stack(images)
        x = xs.reshape(self.n_groups, self.n_images_in_group, *xs.shape[1:])
        return x, torch.as_tensor(j)
