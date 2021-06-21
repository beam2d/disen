import math
from typing import Any, Callable, Sequence

import torch

from .. import data, distributions
from . import latent_spec


class LatentVariableModel(torch.nn.Module):
    spec: latent_spec.LatentSpec
    loss_keys: Sequence[str]

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def encode(self, x: torch.Tensor) -> list[distributions.Distribution]:
        """Compute the posterior distribution q(z|x) with given x.
        
        It returns multiple distributions when the posterior is heterogeneous
        (e.g. some variables are normal while others are categorical).
        """
        raise NotImplementedError

    def decode(self, zs: Sequence[torch.Tensor]) -> distributions.Distribution:
        """Compute the generator distribution p(x|z) with given z."""
        raise NotImplementedError

    def prior(self, batch_size: int) -> list[distributions.Distribution]:
        """Get the prior distribution p(z) with given batch size."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the loss values at given data points."""
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return super().__call__(x)

    def log_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Compute log density of the posterior."""
        raise NotImplementedError

    def log_aggregated_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        batch_size: int,
        zs: Sequence[torch.Tensor],
        selector: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ) -> torch.Tensor:
        """Compute log density of the aggregated posterior.
        
        It aggregates the posterior over a given dataset of x.

        By passing ``selector``, it can compute aggregated posterior of forms
        other than q(z_i); for example, by passing a function that sums up the
        log density, one can compute log q(z) instead of [log q(z[i]) for all i].
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1)
        lse_batches: list[torch.Tensor] = []
        for batch in loader:
            x = batch[0].to(self.device)
            log_q_zs = self.log_posterior(x, zs)
            lse_batches.append(selector(log_q_zs).logsumexp(-2))
        lse = torch.stack(lse_batches).logsumexp(0)
        return lse - math.log(data_size)

    def aggregated_entropy(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        sample_size: int,
        batch_size: int,
        selector: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ) -> torch.Tensor:
        """Compute the entropy of each latent variable.

        It computes H(z_i) = E[-log E_x[q(z_i|x)]]. The inner expectation is computed
        exactly by sweeping the dataset of x. The outer expectation is approximated by
        Monte-Carlo sampling with the given sample size.
        """
        subset = data.subsample(dataset, data_size, sample_size)
        q_zs = self.infer_dataset(subset, batch_size)
        zs = [q_z.sample()[:, None] for q_z in q_zs]
        log_q_z = self.log_aggregated_posterior(
            dataset, data_size, batch_size, zs, selector
        )
        return -log_q_z.mean(0)

    def infer_dataset(
        self, dataset: torch.utils.data.Dataset[Any], batch_size: int
    ) -> list[distributions.Distribution]:
        """Compute the posterior distribution q(z|x) for a given dataset of x."""
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1)
        q_batches = [self.encode(batch[0].to(self.device)) for batch in loader]
        return [distributions.cat(qs) for qs in zip(*q_batches)]
