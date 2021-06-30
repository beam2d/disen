from typing import Generic, Sequence, TypeVar

import torch

from .. import distributions, models


_BaseModel = TypeVar("_BaseModel", bound=models.LatentVariableModel)


class Attack(models.LatentVariableModel, Generic[_BaseModel]):
    loss_keys = ()

    def __init__(self, base: _BaseModel) -> None:
        super().__init__()
        self.base = base
        self.spec = base.spec

    def inject_noise(
        self, zs: Sequence[distributions.Distribution]
    ) -> list[distributions.Distribution]:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> list[distributions.Distribution]:
        zs = self.base.encode(x)
        return self.inject_noise(zs)

    def decode(self, zs: Sequence[torch.Tensor]) -> distributions.Distribution:
        return self.base.decode(zs)

    def prior(self, batch_size: int) -> list[distributions.Distribution]:
        zs = self.base.prior(batch_size)
        return self.inject_noise(zs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("Attack does not define a loss function")
