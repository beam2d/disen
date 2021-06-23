import dataclasses
from typing import Sequence

import torch
import torch.nn.functional as F

from .. import distributions, nn
from . import latent_spec, lvm


@dataclasses.dataclass
class LinearScheduler:
    start: float
    end: float
    end_at: int

    def get(self, time: int) -> float:
        if time >= self.end_at:
            return self.end
        return self.start + (self.end - self.start) * time / self.end_at


class JointVAE(lvm.LatentVariableModel):
    loss_keys = ("loss", "kl_c", "kl_z", "nll", "elbo")

    def __init__(
        self,
        encoder: nn.EncoderBase,
        decoder: torch.nn.Module,
        n_categories: int,
        n_continuous: int,
        gamma: float,
        temperature: float,
        max_capacity_discrete: float,
        max_capacity_continuous: float,
        max_capacity_iteration: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.logits = torch.nn.Linear(encoder.out_features, n_categories)
        self.loc = torch.nn.Linear(encoder.out_features, n_continuous)
        self.scale = torch.nn.Linear(encoder.out_features, n_continuous)
        self.n_categories = n_categories
        self.n_continuous = n_continuous

        self.spec = latent_spec.LatentSpec(
            [
                latent_spec.SingleLatentSpec("c", "categorical", 1, n_categories),
                latent_spec.SingleLatentSpec("z", "real", n_continuous),
            ]
        )
        self.gamma = gamma
        self.temperature = temperature
        self.sched_discrete = LinearScheduler(
            0.0, max_capacity_discrete, max_capacity_iteration
        )
        self.sched_continuous = LinearScheduler(
            0.0, max_capacity_continuous, max_capacity_iteration
        )

        self.iteration = 0

    def encode(self, x: torch.Tensor) -> list[distributions.Distribution]:
        h = self.encoder(x)
        logits = self.logits(h)[:, None]
        c: distributions.Distribution
        if self.training:
            temperature = torch.as_tensor(self.temperature, dtype=x.dtype, device=x.device)
            c = distributions.RelaxedOneHotCategorical(temperature, logits)
        else:
            c = distributions.OneHotCategorical(logits)
        loc = self.loc(h)
        scale = F.softplus(self.scale(h))
        z = distributions.Normal(loc, scale)
        return [c, z]

    def decode(self, zs: Sequence[torch.Tensor]) -> distributions.Distribution:
        B = zs[0].shape[0]
        z = torch.cat([z.reshape(B, -1) for z in zs], 1)
        logits = self.decoder(z)
        return distributions.Bernoulli(logits)

    def prior(self, batch_size: int) -> list[distributions.Distribution]:
        logits = torch.zeros((batch_size, 1, self.n_categories), device=self.device)
        loc = torch.zeros((batch_size, self.n_continuous), device=self.device)
        scale = torch.ones_like(loc)
        p_c = distributions.OneHotCategorical(logits)
        p_z = distributions.Normal(loc, scale)
        return [p_c, p_z]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        q_c, q_z = self.encode(x)
        p_c, p_z = self.prior(x.shape[0])
        kl_c = distributions.kl_divergence(q_c, p_c).sum(1)
        kl_z = distributions.kl_divergence(q_z, p_z).sum(1)

        C_c = self.sched_discrete.get(self.iteration)
        C_z = self.sched_continuous.get(self.iteration)
        if self.training:
            self.iteration += 1

        c = q_c.sample()
        z = q_z.sample()
        p_x = self.decode([c, z])
        nll = -p_x.log_prob(x).sum(list(range(1, x.ndim)))
        elbo = nll + kl_c + kl_z
        loss = nll + self.gamma * (abs(kl_c - C_c) + abs(kl_z - C_z))

        return {"loss": loss, "kl_c": kl_c, "kl_z": kl_z, "nll": nll, "elbo": elbo}
