from __future__ import annotations
from typing import Optional

import torch
import torch.distributions as D

from . import distribution


class Normal(distribution.Distribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        value: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(value)
        loc, scale = torch.broadcast_tensors(loc, scale)
        assert loc.ndim >= 2
        self._base = D.Normal(loc, scale)

    @property
    def base(self) -> D.Normal:
        return self._base

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self._base.loc, self._base.scale)

    @classmethod
    def from_params(
        cls,
        params: tuple[torch.Tensor, ...],
        value: Optional[torch.Tensor] = None,
    ) -> Normal:
        loc, scale = params
        return Normal(loc, scale, value)

    @property
    def loc(self) -> torch.Tensor:
        return self._base.loc

    @property
    def scale(self) -> torch.Tensor:
        return self._base.scale

    @property
    def variance(self) -> torch.Tensor:
        return self._base.variance


class MultivariateNormal(distribution.Distribution):
    def __init__(
        self,
        loc: torch.Tensor,
        cov: torch.Tensor,
        value: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(value)
        cov_shape = torch.broadcast_shapes(loc.shape + (loc.shape[-1],), cov.shape)
        loc_shape = cov_shape[:-1]
        loc = torch.broadcast_to(loc, loc_shape)
        cov = torch.broadcast_to(cov, cov_shape)
        self._base = D.MultivariateNormal(loc, covariance_matrix=cov)

    @property
    def base(self) -> D.MultivariateNormal:
        return self._base

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self._base.loc, self._base.covariance_matrix)

    @classmethod
    def from_params(
        cls,
        params: tuple[torch.Tensor, ...],
        value: Optional[torch.Tensor] = None,
    ) -> MultivariateNormal:
        loc, cov = params
        return MultivariateNormal(loc, cov, value)

    @property
    def loc(self) -> torch.Tensor:
        return self._base.loc

    @property
    def cov(self) -> torch.Tensor:
        return self._base.covariance_matrix

    def marginalize(self) -> Normal:
        var = self._base.covariance_matrix.diagonal(0, -1, -2)
        return Normal(self._base.loc, var.sqrt(), self.value)
