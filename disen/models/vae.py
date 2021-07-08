from typing import Sequence

import torch
import torch.nn.functional as F

from .. import distributions
from ..nn import encoder
from . import latent_spec, lvm


class VAE(lvm.LatentVariableModel):
    loss_keys = ("loss", "kl_z", "recon", "elbo")

    def __init__(
        self,
        encoder: encoder.EncoderBase,
        decoder: torch.nn.Module,
        n_latents: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loc = torch.nn.Linear(encoder.out_features, n_latents)
        self.scale = torch.nn.Linear(encoder.out_features, n_latents)
        self.n_latents = n_latents

        self.spec = latent_spec.LatentSpec(
            [latent_spec.SingleLatentSpec("z", "real", n_latents)]
        )
        self.beta = beta

    def encode(self, x: torch.Tensor) -> list[distributions.Distribution]:
        h = self.encoder(x)
        loc = self.loc(h)
        scale = F.softplus(self.scale(h))
        return [distributions.Normal(loc, scale)]

    def decode(self, zs: Sequence[torch.Tensor]) -> distributions.Distribution:
        logits = self.decoder(*zs)
        return distributions.Bernoulli(logits)

    def prior(self, batch_size: int) -> list[distributions.Distribution]:
        loc = torch.zeros((batch_size, self.n_latents), device=self.device)
        scale = torch.ones_like(loc)
        return [distributions.Normal(loc, scale)]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (q_z,) = self.encode(x)
        (p_z,) = self.prior(x.shape[0])
        kl_z = distributions.kl_divergence(q_z, p_z).sum(1)

        z = q_z.sample()
        p_x = self.decode([z])
        recon = -p_x.log_prob(x).sum(list(range(1, x.ndim)))
        elbo = -(recon + kl_z)
        loss = recon + self.beta * kl_z

        return {"loss": loss, "kl_z": kl_z, "recon": recon, "elbo": elbo}
