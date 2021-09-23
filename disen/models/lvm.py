import math
from typing import Any, Callable, Sequence

import torch

from .. import data, distributions, nn
from . import latent_spec


class LatentVariableModel(torch.nn.Module):
    spec: latent_spec.LatentSpec
    loss_keys: Sequence[str]

    @property
    def device(self) -> torch.device:
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    @property
    def has_valid_elemwise_posterior(self) -> bool:
        return True

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

    def infer_sample(self, x: torch.Tensor) -> list[torch.Tensor]:
        q_zs = self.encode(x)
        return [q_z.sample() for q_z in q_zs]

    def infer_mean(self, x: torch.Tensor) -> list[torch.Tensor]:
        q_zs = self.encode(x)
        return [q_z.mean for q_z in q_zs]

    def encode_dataset(
        self,
        dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int = 8192,
    ) -> tuple[list[distributions.Distribution], torch.Tensor]:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=1
        )
        qs: list[list[distributions.Distribution]] = []
        ys: list[torch.Tensor] = []
        for x, y in loader:
            qs.append(self.encode(x.to(self.device)))
            ys.append(y.to(self.device))
        q_all = [distributions.cat(q_z_batches) for q_z_batches in zip(*qs)]
        y_all = torch.cat(ys, 0)
        return (q_all, y_all)

    def log_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Compute log density of the posterior.

        It computes [log q(z_i|x)]_i.
        """
        assert self.has_valid_elemwise_posterior
        q_zs = self.encode(x)
        log_q_zs = [q_z.log_prob(z) for z, q_z in zip(zs, q_zs)]
        return torch.cat(log_q_zs, -1)

    def log_loo_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Compute log density of the posterior at leave-one-out variables.

        It computes [log q(z_{-i}|x)]_i.
        """
        assert self.has_valid_elemwise_posterior
        log_q = self.log_posterior(x, zs)
        return nn.enumerate_loo(log_q, -1).sum(-1).movedim(0, -1)

    def log_joint_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Compute log density of the joint posterior.

        It computes log q(z|x). Note that it adds a singleton dimension at the end for
        consistency with other functions.
        """
        log_q = self.log_posterior(x, zs)
        return log_q.sum(-1, keepdim=True)

    def log_aggregated_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        batch_size: int,
        zs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute log density of the aggregated posterior.

        It aggregates the posterior over a given dataset of x to compute log q(z_i) for
        each i.
        """
        return self._log_aggregated_posterior(
            dataset, batch_size, zs, self.log_posterior
        )

    def log_aggregated_loo_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        batch_size: int,
        zs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute log density of the aggregated posterior for leave-one-out variables.

        It aggregates the posterior over a given dataset of x to compute log q(z_{-i})
        for each i.
        """
        return self._log_aggregated_posterior(
            dataset, batch_size, zs, self.log_loo_posterior
        )

    def log_aggregated_joint_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        batch_size: int,
        zs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute log density of the aggregated joint posterior.

        It aggregates the posterior over a given dataset of x to compute log q(z).
        """
        return self._log_aggregated_posterior(
            dataset, batch_size, zs, self.log_joint_posterior
        )

    def _log_aggregated_posterior(
        self,
        dataset: torch.utils.data.Dataset[Any],
        batch_size: int,
        zs: Sequence[torch.Tensor],
        log_q: Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1)
        lse_batches: list[torch.Tensor] = []
        data_size = 0
        for batch in loader:
            x = batch[0].to(self.device)
            log_q_zs = log_q(x, zs)
            lse_batches.append(log_q_zs.logsumexp(-2))
            data_size += log_q_zs.shape[-2]
        lse = torch.stack(lse_batches).logsumexp(0)
        return lse - math.log(data_size)

    def aggregated_entropy(
        self,
        inner_dataset: torch.utils.data.Dataset[Any],
        outer_dataset: torch.utils.data.Dataset[Any],
        inner_batch_size: int,
        outer_batch_size: int,
    ) -> torch.Tensor:
        """Compute the entropy of each latent variable.

        It computes H(z_i) = E[-log E_x[q(z_i|x)]]. The inner expectation is computed
        exactly by sweeping the dataset of x. The outer expectation is approximated by
        Monte-Carlo sampling with the given sample size.
        """
        return self._aggregated_entropy(
            inner_dataset,
            outer_dataset,
            inner_batch_size,
            outer_batch_size,
            self.log_posterior,
        )

    def aggregated_loo_entropy(
        self,
        inner_dataset: torch.utils.data.Dataset[Any],
        outer_dataset: torch.utils.data.Dataset[Any],
        inner_batch_size: int,
        outer_batch_size: int,
    ) -> torch.Tensor:
        """Compute the entropy of leave-one-out latent variables.

        It computes H(z_{-i}) = E[-log E_x[q(z_{-i}|x)]].
        """
        return self._aggregated_entropy(
            inner_dataset,
            outer_dataset,
            inner_batch_size,
            outer_batch_size,
            self.log_loo_posterior,
        )

    def aggregated_joint_entropy(
        self,
        inner_dataset: torch.utils.data.Dataset[Any],
        outer_dataset: torch.utils.data.Dataset[Any],
        inner_batch_size: int,
        outer_batch_size: int,
    ) -> torch.Tensor:
        """Compute the entropy of joint posterior.

        It computes H(z) = E[-log E_x[q(z|x)]].
        """
        return self._aggregated_entropy(
            inner_dataset,
            outer_dataset,
            inner_batch_size,
            outer_batch_size,
            self.log_joint_posterior,
        )

    def _aggregated_entropy(
        self,
        inner_dataset: torch.utils.data.Dataset[Any],
        outer_dataset: torch.utils.data.Dataset[Any],
        inner_batch_size: int,
        outer_batch_size: int,
        log_q: Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        zs = self._infer_aggregated(outer_dataset, outer_batch_size)
        log_q_z = self._log_aggregated_posterior(
            inner_dataset, inner_batch_size, zs, log_q
        )
        return -log_q_z.mean(0)

    def _infer_aggregated(
        self, dataset: torch.utils.data.Dataset[Any], batch_size: int
    ) -> list[torch.Tensor]:
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=1)
        q_batches = [self.encode(batch[0].to(self.device)) for batch in loader]
        return [distributions.cat(qs).sample()[:, None] for qs in zip(*q_batches)]


class EpochInference:
    def __init__(
        self,
        model: LatentVariableModel,
        dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int = 8192,
    ) -> None:
        super().__init__()
        self.model = model
        if model.has_valid_elemwise_posterior:
            self.pre_encoded = model.encode_dataset(dataset, batch_size)
        else:
            self.dataset = dataset
            self.batch_size = batch_size

    def next_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.has_valid_elemwise_posterior:
            q_all, y_all = self.pre_encoded
        if not self.model.has_valid_elemwise_posterior:
            q_all, y_all = self.model.encode_dataset(self.dataset, self.batch_size)

        z_all = torch.cat([q.sample() for q in q_all], 1)
        return (z_all, y_all)
