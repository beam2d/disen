import torch
import torch.nn.functional as F

from .. import distributions
from .. import nn
from . import vae


class FactorVAEDiscriminator(torch.nn.Sequential):
    def __init__(self, n_latents: int) -> None:
        width = 1000
        super().__init__(
            torch.nn.Linear(n_latents, width),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(width, width),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(width, width),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(width, width),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(width, width),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(width, 1),
        )
        self.in_features = n_latents


class FactorVAE(vae.VAE):
    def __init__(
        self,
        encoder: nn.EncoderBase,
        decoder: torch.nn.Module,
        discriminator: FactorVAEDiscriminator,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(encoder, decoder, discriminator.in_features, beta=1.0)
        self.gamma = gamma
        self._D = (discriminator,)

    @property
    def D(self) -> FactorVAEDiscriminator:
        return self._D[0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        (q_z,) = self.encode(x)
        (p_z,) = self.prior(x.shape[0])
        kl_z = distributions.kl_divergence(q_z, p_z).sum(1)

        z = q_z.sample()
        p_x = self.decode([z])
        recon = -p_x.log_prob(x).sum(list(range(1, x.ndim)))
        nll = recon + kl_z
        elbo = -nll
        d_logits = self.D(z)
        loss = nll + self.gamma * d_logits

        return {"loss": loss, "kl_z": kl_z, "recon": recon, "elbo": elbo}

    def discriminator_loss(self, x: torch.Tensor, z_pos: torch.Tensor) -> torch.Tensor:
        (q_z,) = self.encode(x)
        z_neg = q_z.sample()
        z_neg = _permute_dims(z_neg)

        d_pos = F.softplus(-self.D(z_pos).reshape(-1))
        d_neg = F.softplus(self.D(z_neg).reshape(-1))

        return torch.cat([d_pos, d_neg])


def _permute_dims(z: torch.Tensor) -> torch.Tensor:
    perms = [torch.randperm(z.shape[0], device=z.device) for _ in range(z.shape[1])]
    perm = torch.stack(perms, 1)
    return z.gather(0, perm)
