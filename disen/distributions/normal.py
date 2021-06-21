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
