from __future__ import annotations
from typing import Optional, Sequence, Type, TypeVar

import torch
import torch.distributions as D


_Distr = TypeVar("_Distr", bound="Distribution")


class Distribution:
    def __init__(self, value: Optional[torch.Tensor] = None) -> None:
        self._value: Optional[torch.Tensor] = value

    @property
    def base(self) -> D.Distribution:
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]

    @property
    def elem_shape(self) -> tuple[int, ...]:
        return self.shape[2:]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @classmethod
    def from_params(
        cls: Type[_Distr],
        params: tuple[torch.Tensor, ...],
        value: Optional[torch.Tensor] = None,
    ) -> _Distr:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        base = self.base
        return tuple(base.batch_shape + base.event_shape)

    @property
    def strict(self) -> Distribution:
        return self

    @property
    def value(self) -> Optional[torch.Tensor]:
        return self._value

    def __getitem__(self: _Distr, index: slice) -> _Distr:
        params = self.params
        new_params = tuple(p[index] for p in params)
        new_value = None if self._value is None else self._value[index]
        return type(self).from_params(new_params, new_value)

    def entropy(self) -> torch.Tensor:
        return self.base.entropy()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.log_prob(x)

    def sample(self) -> torch.Tensor:
        if self._value is None:
            base = self.base
            self._value = base.rsample() if base.has_rsample else base.sample()
        return self._value


def cat(distrs: Sequence[_Distr]) -> _Distr:
    cls = type(distrs[0])
    params = [d.params for d in distrs]
    cat_params = tuple(torch.cat(ps, 0) for ps in zip(*params))

    values: list[torch.Tensor] = []
    for d in distrs:
        if d.value is None:
            return cls.from_params(cat_params)
        values.append(d.value)
    cat_values = torch.cat(values, 0)

    return cls.from_params(cat_params, cat_values)


def kl_divergence(l: Distribution, r: Distribution) -> torch.Tensor:
    return D.kl_divergence(l.base, r.base)
