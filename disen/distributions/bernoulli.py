from __future__ import annotations
from typing import Optional

import torch
import torch.distributions as D

from . import distribution


class Bernoulli(distribution.Distribution):
    def __init__(
        self, logits: torch.Tensor, value: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(value)
        assert logits.ndim >= 2
        self._base = D.Bernoulli(logits=logits, validate_args=False)

    @property
    def base(self) -> D.Bernoulli:
        return self._base

    @property
    def params(self) -> tuple[torch.Tensor]:
        return (self._base.logits,)

    @classmethod
    def from_params(
        cls, params: tuple[torch.Tensor, ...], value: Optional[torch.Tensor] = None
    ) -> Bernoulli:
        (logits,) = params
        return Bernoulli(logits, value)
