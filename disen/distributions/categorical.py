from __future__ import annotations
from typing import Optional

import torch
import torch.distributions as D

from . import distribution


class OneHotCategorical(distribution.Distribution):
    def __init__(
        self, logits: torch.Tensor, value: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(value)
        assert logits.ndim >= 3
        self._base = D.OneHotCategorical(logits=logits)

    @property
    def base(self) -> D.OneHotCategorical:
        return self._base

    @property
    def params(self) -> tuple[torch.Tensor]:
        return (self._base.logits,)

    @classmethod
    def from_params(
        cls, params: tuple[torch.Tensor, ...], value: Optional[torch.Tensor] = None
    ) -> OneHotCategorical:
        (logits,) = params
        return OneHotCategorical(logits, value)


class RelaxedOneHotCategorical(distribution.Distribution):
    def __init__(
        self,
        temperature: torch.Tensor,
        logits: torch.Tensor,
        value: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(value)
        assert logits.ndim >= 3
        # TODO(beam2d): Relax this constraint
        assert temperature.numel() == 1
        self._base = D.RelaxedOneHotCategorical(temperature, logits=logits)

    @property
    def base(self) -> D.RelaxedOneHotCategorical:
        return self._base

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self._base.temperature, self._base.logits)

    @classmethod
    def from_params(
        cls, params: tuple[torch.Tensor, ...], value: Optional[torch.Tensor] = None
    ) -> RelaxedOneHotCategorical:
        temperature, logits = params
        return RelaxedOneHotCategorical(temperature, logits, value)

    @property
    def strict(self) -> OneHotCategorical:
        return OneHotCategorical(self._base.logits)


class OneHotCategoricalWithProbs(distribution.Distribution):
    def __init__(self, probs: torch.Tensor, value: Optional[torch.Tensor] = None) -> None:
        super().__init__(value)
        assert probs.ndim >= 3
        self._base = D.OneHotCategorical(probs=probs)

    @property
    def base(self) -> D.OneHotCategorical:
        return self._base

    @property
    def params(self) -> tuple[torch.Tensor]:
        return (self._base.probs,)

    @classmethod
    def from_params(
        cls, params: tuple[torch.Tensor, ...], value: Optional[torch.Tensor] = None
    ) -> OneHotCategoricalWithProbs:
        (probs,) = params
        return OneHotCategoricalWithProbs(probs, value)
