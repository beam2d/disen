from typing import Sequence

import torch

from .. import distributions, models
from . import base


class NoisedCopyAttack(base.Attack):
    """Add extra variables ``z_{n+i} = z_i + epsilon``.

    Here, epsilon is a common noise following ``N(0, std^2)``.
    """

    def __init__(self, base: models.LatentVariableModel, std: float) -> None:
        super().__init__(base)
        base_spec = self.base.spec.specs
        extra_spec = tuple(
            _generate_noised_spec(latent)
            for latent in base_spec
            if latent.domain == "real"
        )
        self.spec = models.LatentSpec(base_spec + extra_spec)
        self.std = std

    def inject_noise(
        self, zs: Sequence[distributions.Distribution]
    ) -> list[distributions.Distribution]:
        new_zs = list(zs)
        eps = float(torch.randn(()) * self.std)

        for z, spec in zip(zs, self.base.spec):
            if not isinstance(z, distributions.Normal):
                continue
            assert spec.domain == "real"
            new_std = torch.sqrt(z.base.scale ** 2 + self.std ** 2)
            new_zs.append(distributions.Normal(z.base.loc, new_std, z.sample() + eps))

        return new_zs


def _generate_noised_spec(latent: models.SingleLatentSpec) -> models.SingleLatentSpec:
    assert latent.domain == "real"
    return models.SingleLatentSpec(latent.name + "_noised", "real", latent.size)
