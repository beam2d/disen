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
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

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
        q_zs = self.encode(x)
        log_q_zs = [q_z.log_prob(z) for z, q_z in zip(zs, q_zs)]
        return torch.cat(log_q_zs, -1)

    def log_loo_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Compute log density of the posterior at leave-one-out variables.

        It computes [log q(z_{-i}) for all i].
        """
        log_q = self.log_posterior(x, zs)
        return log_q.sum(-1, keepdim=True) - log_q

    def log_aggregated_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        batch_size: int,
        zs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute log density of the aggregated posterior.

        It aggregates the posterior over a given dataset of x to compute log q(z_i) for
        each i.
        """
        return self._log_aggregated_posterior(
            dataset, data_size, batch_size, zs, self.log_posterior
        )

    def log_aggregated_loo_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        batch_size: int,
        zs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute log density of the aggregated posterior for leave-one-out variables.

        It aggregates the posterior over a given dataset of x to compute log q(z_{-i})
        for each i.
        """
        return self._log_aggregated_posterior(
            dataset, data_size, batch_size, zs, self.log_loo_posterior
        )

    def _log_aggregated_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        batch_size: int,
        zs: Sequence[torch.Tensor],
        log_q: Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1)
        lse_batches: list[torch.Tensor] = []
        for batch in loader:
            x = batch[0].to(self.device)
            log_q_zs = log_q(x, zs)
            lse_batches.append(log_q_zs.logsumexp(-2))
        lse = torch.stack(lse_batches).logsumexp(0)
        return lse - math.log(data_size)

    def aggregated_entropy(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        sample_size: int,
        inner_batch_size: int,
        outer_batch_size: int,
    ) -> torch.Tensor:
        """Compute the entropy of each latent variable.

        It computes H(z_i) = E[-log E_x[q(z_i|x)]]. The inner expectation is computed
        exactly by sweeping the dataset of x. The outer expectation is approximated by
        Monte-Carlo sampling with the given sample size.
        """
        return self._aggregated_entropy(
            dataset,
            data_size,
            sample_size,
            inner_batch_size,
            outer_batch_size,
            self.log_posterior,
        )

    def aggregated_loo_entropy(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        sample_size: int,
        inner_batch_size: int,
        outer_batch_size: int,
    ) -> torch.Tensor:
        """Compute the entropy of leave-one-out latent variables.

        It computes H(z_{-i}) = E[-log E_x[q(z_{-i}|x)]].
        """
        return self._aggregated_entropy(
            dataset,
            data_size,
            sample_size,
            inner_batch_size,
            outer_batch_size,
            self.log_loo_posterior,
        )

    def _aggregated_entropy(
        self,
        dataset: torch.utils.data.Dataset[Any],
        data_size: int,
        sample_size: int,
        inner_batch_size: int,
        outer_batch_size: int,
        log_q: Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        subset = data.subsample(dataset, data_size, sample_size)
        zs = self._infer_aggregated(subset, outer_batch_size)
        log_q_z = self._log_aggregated_posterior(
            dataset, data_size, inner_batch_size, zs, log_q
        )
        return -log_q_z.mean(0)

    def _infer_aggregated(
        self, dataset: torch.utils.data.Dataset[Any], batch_size: int
    ) -> list[torch.Tensor]:
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1)
        q_batches = [self.encode(batch[0].to(self.device)) for batch in loader]
        return [distributions.cat(qs).sample()[:, None] for qs in zip(*q_batches)]
