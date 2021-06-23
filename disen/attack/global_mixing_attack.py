from typing import Sequence

import torch

from .. import distributions, models, nn
from . import base


class GlobalMixingAttack(base.Attack):
    """Add ``alpha * (z.sum() - z)`` to each Gaussian z."""

    def __init__(self, base: models.LatentVariableModel, alpha: float) -> None:
        super().__init__(base)
        self.alpha = alpha

    def inject_noise(
        self, zs: Sequence[distributions.Distribution]
    ) -> list[distributions.Distribution]:
        new_zs: list[distributions.Distribution] = []
        for z in zs:
            if isinstance(z, distributions.Normal):
                z = distributions.Normal(
                    self._mix_additive(z.base.mean),
                    self._mix_additive(z.base.variance).sqrt(),
                    self._mix_additive(z.sample()),
                )
            new_zs.append(z)
        return new_zs

    def _mix_additive(self, value: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * value + self.alpha * value.sum(-1, keepdim=True)
