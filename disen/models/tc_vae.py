import math

from typing import Sequence

import torch
import torch.nn.functional as F

from .. import distributions, nn
from . import latent_spec, lvm


class TCVAE(lvm.LatentVariableModel):
    loss_keys = ("loss", "elbo", "recon", "icmi", "tc", "dimkl")

    def __init__(
        self,
        encoder: nn.EncoderBase,
        decoder: torch.nn.Module,
        n_latents: int,
        dataset_size: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
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
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def encode(self, x: torch.Tensor) -> list[distributions.Distribution]:
        h = self.encoder(x)
        loc = self.loc(h)
        scale = F.softplus(self.scale(h))
        return [distributions.Normal(loc, scale)]

    def decode(self, zs: Sequence[torch.Tensor]) -> distributions.Bernoulli:
        logits = self.decoder(*zs)
        return distributions.Bernoulli(logits)

    def prior(self, batch_size: int) -> list[distributions.Distribution]:
        loc = torch.zeros((batch_size, self.n_latents), device=self.device)
        scale = torch.ones_like(loc)
        return [distributions.Normal(loc, scale)]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        N = self.dataset_size
        B = x.shape[0]

        (q_z,) = self.encode(x)
        (p_z,) = self.prior(B)
        z = q_z.sample()
        p_x = self.decode([z])
        recon = -p_x.log_prob(x).sum(list(range(1, x.ndim)))
        kl = distributions.kl_divergence(q_z, p_z).sum(1)
        elbo = -(recon + kl)

        z_expand = z[:, None].expand((B, B, z.shape[1]))

        # [log q(z_i(x_j) | x_k)]_{jki}
        log_qzi_x = q_z.log_prob(z_expand)
        # [log q(z(x_j) | x_j)]_j
        log_qz_x = log_qzi_x.diagonal().sum(0)
        # [log q(z_i(x_j))]_{ji}
        log_qzi = _mss_log_marginal(log_qzi_x, B - 1, N)
        # [log q(z(x_j))]_j
        log_qz = _mss_log_marginal(log_qzi_x.sum(2), B - 1, N)
        # [log p(z_i(x_j))]_{ji}
        log_pzi = p_z.log_prob(z)

        # Note: mean_j is taken by the caller
        # I(z; x) ~= mean_j [log q(z(x_j)|x_j) - log q(z(x_j))]
        icmi = log_qz_x - log_qz
        # TC(q(z) || prod_i q(z_i)) ~= mean_j [log q(z(x_j)) - sum_i log q(z_i(x_j))]
        tc = log_qz - log_qzi.sum(1)
        # sum_i KL(q(z_i) || p(z_i)) ~= sum_i [log q(z_i(x_j)) - log p(z_i(x_j))]
        dimkl = (log_qzi - log_pzi).sum(1)

        print(f"kl={kl.mean().item()} kl'={(log_qz_x - log_pzi.sum(1)).mean().item()} tc={tc.mean().item()}")

        loss = recon + self.alpha * icmi + self.beta * tc + self.gamma * dimkl
        return {"loss": loss, "elbo": elbo, "recon": recon, "icmi": icmi, "tc": tc, "dimkl": dimkl}


def _mss_log_marginal(log_qz_x: torch.Tensor, M: int, N: int) -> torch.Tensor:
    # Minibatch Stratified Sampling
    # We alter the original derivation to simplify the computation.
    # q(z) = 1/N q(z|n^*) + (1/M - 1/MN) sum_i q(z|n_i)
    # We do the computation in log space.
    log_coeff = torch.full_like(log_qz_x, math.log((N - 1) / (M * N)))
    log_coeff.diagonal()[:] = -math.log(N)
    return (log_qz_x + log_coeff).logsumexp(1)
