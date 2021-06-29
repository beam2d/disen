from typing import Iterable, Optional, Sequence

import torch

from .. import distributions, models, nn
from . import base


class RedundancyAttack(base.Attack):
    """z' = (z, alpha * U @ z + eps) where eps ~ N(0, I) and U is orthogonal."""

    def __init__(
        self,
        base: models.LatentVariableModel,
        alpha: float,
        U: torch.Tensor,
    ) -> None:
        super().__init__(base)
        self.au = alpha * U
        self.base_i = -1

        extra_spec: Optional[models.SingleLatentSpec] = None
        for i, latent in enumerate(base.spec):
            if latent.domain == "real":
                assert extra_spec is None, "only one lv is allowed to be attacked"
                extra_spec = models.SingleLatentSpec(
                    latent.name + "_atk",
                    latent.domain,
                    latent.size,
                )
                self.base_i = i
        assert extra_spec is not None
        assert self.base_i >= 0

        assert U.shape == (extra_spec.size, extra_spec.size)
        torch.testing.assert_allclose((U @ U.T).cpu(), torch.eye(extra_spec.size))

        self.spec = models.LatentSpec(base.spec.specs + (extra_spec,))

    def inject_noise(
        self, zs: Sequence[distributions.Distribution]
    ) -> list[distributions.Distribution]:
        new_zs = list(zs)

        for z, spec in zip(zs, self.base.spec):
            if spec.domain != "real":
                continue
            assert isinstance(z, distributions.Normal)
            loc = self._multiply_au(z.base.loc)
            var = z.base.variance
            cov = self.au @ torch.diag_embed(var) @ self.au.transpose(-1, -2)
            cov += torch.eye(cov.shape[-1], device=cov.device)

            value = z.sample()
            eps = torch.randn_like(value)
            new_value = self._multiply_au(value) + eps

            new_zs.append(distributions.MultivariateNormal(loc, cov, new_value))

        return new_zs

    def log_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        q_zs = self.encode(x)
        q_z_extra = q_zs[-1]
        assert isinstance(q_z_extra, distributions.MultivariateNormal)
        q_z_marginal = q_z_extra.marginalize()
        q_zs_marginal = q_zs[:-1] + [q_z_marginal]
        log_q_zs = [q_z.log_prob(z) for z, q_z in zip(zs, q_zs_marginal)]
        return torch.cat(log_q_zs, -1)

    def log_loo_posterior(
        self, x: torch.Tensor, zs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        q_zs = self.encode(x)

        has_other = len(zs) > 2
        if has_other:
            log_q_other = torch.cat([
                q.log_prob(z)
                for i, (z, q) in enumerate(zip(zs[:-1], q_zs[:-1]))
                if i != self.base_i
            ], -1)
            log_q_sum_other = log_q_other.sum(-1, keepdim=True)
        else:
            log_q_sum_other = torch.zeros((), dtype=x.dtype, device=x.device)

        z_base = zs[self.base_i]
        q_base = q_zs[self.base_i]
        z_new = zs[-1]
        q_new = q_zs[-1]
        assert isinstance(q_base, distributions.Normal)
        assert isinstance(q_new, distributions.MultivariateNormal)

        z_cat = torch.cat([z_base, z_new], -1)
        loc_cat = torch.cat([q_base.loc, q_new.loc], -1)
        cov_cat = self._make_cov(q_base, q_new)

        if loc_cat.ndim < z_cat.ndim:
            # extra batch dimension in z; make it explicit
            loc_cat = loc_cat[None]
            cov_cat = cov_cat[None]
        q = distributions.MultivariateNormal(loc_cat, cov_cat)
        log_q_sum_rest = q.log_prob(z_cat)

        loc_loo = nn.enumerate_loo(loc_cat)
        cov_loo = nn.principal_submatrices(cov_cat)
        z_loo = nn.enumerate_loo(z_cat)
        q_loo = distributions.MultivariateNormal(loc_loo, cov_loo)
        del z_cat, loc_cat, cov_cat, loc_loo, cov_loo
        log_q_loo = torch.movedim(q_loo.log_prob(z_loo), 0, -1)

        base_size = z_base.shape[-1]
        log_q_loo_base = log_q_loo[..., :base_size] + log_q_sum_other
        log_q_loo_new = log_q_loo[..., base_size:] + log_q_sum_other

        if has_other:
            log_q_loo_other = log_q_sum_other - log_q_other + log_q_sum_rest
            base_pos = sum(self.base.spec[i].size for i in range(self.base_i))
            return torch.cat([
                log_q_loo_other[..., :base_pos],
                log_q_loo_base,
                log_q_loo_other[..., base_pos:],
                log_q_loo_new,
            ], -1)
        return torch.cat([log_q_loo_base, log_q_loo_new], -1)

    def _multiply_au(self, x: torch.Tensor) -> torch.Tensor:
        return (self.au @ x[..., None])[..., 0]

    def _make_cov(
        self, q_base: distributions.Normal, q_new: distributions.MultivariateNormal
    ) -> torch.Tensor:
        cov00 = q_base.variance.diag_embed()
        cov01 = cov00 @ self.au.T
        cov10 = cov01.swapaxes(-2, -1)
        cov11 = q_new.cov
        return nn.block_matrix([[cov00, cov01], [cov10, cov11]])
