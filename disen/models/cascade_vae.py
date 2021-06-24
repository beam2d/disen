import math
from typing import Sequence, cast

from ortools.graph import pywrapgraph
import torch
import torch.nn.functional as F

from .. import distributions, nn
from . import latent_spec, lvm


class CascadeVAE(lvm.LatentVariableModel):
    loss_keys = ("loss", "kl_z", "nll", "elbo")

    def __init__(
        self,
        encoder: nn.EncoderBase,
        decoder: torch.nn.Module,
        n_categories: int,
        n_continuous: int,
        beta_h: float,
        beta_l: float,
        beta_dumping_interval: int,
        warmup_iteration: int,
        duplicate_penalty: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loc = torch.nn.Linear(encoder.out_features, n_continuous)
        self.scale = torch.nn.Linear(encoder.out_features, n_continuous)
        self.n_categories = n_categories
        self.n_continuous = n_continuous

        self.spec = latent_spec.LatentSpec(
            [
                latent_spec.SingleLatentSpec("z", "real", n_continuous),
                latent_spec.SingleLatentSpec("c", "categorical", 1, n_categories),
            ]
        )
        self.beta_h = beta_h
        self.beta_l = beta_l
        self.beta_dumping_interval = beta_dumping_interval
        self.duplicate_penalty = duplicate_penalty

        self.warmup_iteration = warmup_iteration
        self.iteration = 0

    def encode(self, x: torch.Tensor) -> list[distributions.Distribution]:
        z = self._infer_z(x)
        probs = self._infer_c(x, z.sample())
        c = _categorical_from_value(probs)
        return [z, c]

    def decode(self, zs: Sequence[torch.Tensor]) -> distributions.Distribution:
        B = zs[0].shape[0]
        z = torch.cat([z.reshape(B, -1) for z in zs], 1)
        logits = self.decoder(z)
        return distributions.Bernoulli(logits)

    def prior(self, batch_size: int) -> list[distributions.Distribution]:
        loc = torch.zeros((batch_size, self.n_continuous), device=self.device)
        scale = torch.ones_like(loc)
        logits = torch.zeros((batch_size, 1, self.n_categories), device=self.device)
        p_z = distributions.Normal(loc, scale)
        p_c = distributions.OneHotCategorical(logits)
        return [p_z, p_c]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            self.iteration += 1

        B = x.shape[0]
        k = self.n_categories

        q_z = self._infer_z(x)
        z = q_z.sample()
        p_z, _ = self.prior(B)
        kl_z_ptw = distributions.kl_divergence(q_z, p_z)
        kl_z = kl_z_ptw.sum(1)
        kl_c = math.log(k)

        if self.iteration <= self.warmup_iteration:
            c = torch.zeros((B, k), dtype=x.dtype, device=x.device)
            p_x = self.decode([z, c])
            nll = -p_x.log_prob(x).sum((1, 2, 3))
        else:
            x_expand = x[:, None].expand(B, k, *_rdim(x)).reshape(B * k, *_rdim(x))
            z_expand = z[:, None].expand(B, k, *_rdim(z)).reshape(B * k, *_rdim(z))
            c_all = torch.eye(k, dtype=x.dtype, device=x.device)
            c_all = c_all[None].expand(B, k, k).reshape(B * k, 1, k)

            p_x_all = self.decode([z_expand, c_all])
            nll_p_all = -p_x_all.log_prob(x_expand).sum((1, 2, 3)).reshape(B, k)
            c_cat = self._solve_mincost_flow(nll_p_all)
            c = F.one_hot(c_cat, k)[:, None]
            nll = torch.gather(nll_p_all, 1, c_cat[:, None])[:, 0]

        elbo = nll + kl_z + kl_c
        loss = nll + kl_z_ptw @ self._beta() + kl_c

        return {"loss": loss, "kl_z": kl_z, "nll": nll, "elbo": elbo}

    def _infer_z(self, x: torch.Tensor) -> distributions.Distribution:
        h = self.encoder(x)
        loc = self.loc(h)
        scale = F.softplus(self.scale(h))
        return distributions.Normal(loc, scale)

    def _infer_c(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        k = self.n_categories
        B = x.shape[0]
        assert z.shape[0] == B

        x_expand = x[:, None].expand(B, k, *_rdim(x)).reshape(B * k, *_rdim(x))
        z_expand = z[:, None].expand(B, k, *_rdim(z)).reshape(B * k, *_rdim(z))
        c_all = torch.eye(k, dtype=x.dtype, device=x.device)
        c_all = c_all[None].expand(B, k, k).reshape(B * k, 1, k)

        p_x_all = self.decode([z_expand, c_all])
        nll_p_all = -p_x_all.log_prob(x_expand).sum((1, 2, 3)).reshape(B, k)
        c_cat = nll_p_all.argmin(1)
        return cast(torch.Tensor, F.one_hot(c_cat, k)[:, None])

    def _beta(self) -> torch.Tensor:
        betas: list[float] = []
        for i in range(self.n_continuous):
            if self.iteration < (i + 1) * self.beta_dumping_interval:
                betas.append(self.beta_h)
            else:
                betas.append(self.beta_l)
        return torch.as_tensor(betas, device=self.device)

    @torch.no_grad()
    def _solve_mincost_flow(self, nll_p_all: torch.Tensor) -> torch.Tensor:
        u = (nll_p_all - nll_p_all.min()).detach().cpu().numpy()
        lmd = self.duplicate_penalty
        n, k = u.shape

        mcf = pywrapgraph.SimpleMinCostFlow()

        # SimpleMinCostFlow does not support non-integral cost.
        # We multiply and round them to approximate the problem with integers.
        # TODO(beam2d): Decide the multiplier adaptively.
        def round(x: float) -> int:
            return int(x * 10000 + 0.5)

        # graph spec:
        # - node 0: source
        # - node 1, ..., n: sample
        # - node n+1, ..., n+k: categories
        # - node n+k+1: sink
        # (i=1,...,n, j=1,...,k)
        # - edge 0 -> i: give 1 to each example
        # - edge i -> n+j: assign category j to i-th example
        # - edge n+j -> n+k+1: duplicate penalty for flow >1
        for i in range(n):
            mcf.AddArcWithCapacityAndUnitCost(0, i + 1, 1, 0)
            for j in range(k):
                u_round = round(u[i, j])
                dup_round = round(i * lmd)
                mcf.AddArcWithCapacityAndUnitCost(i + 1, n + j + 1, 1, u_round)
                mcf.AddArcWithCapacityAndUnitCost(n + j + 1, n + k + 1, 1, dup_round)

        mcf.SetNodeSupply(0, n)
        mcf.SetNodeSupply(n + k + 1, -n)
        mcf.Solve()

        c_cat = torch.full((n,), -1, dtype=torch.int64)
        for edge in range(mcf.NumArcs()):
            if 1 <= mcf.Tail(edge) <= n and mcf.Flow(edge) > 0:
                assert n < mcf.Head(edge) <= n + k
                c_cat[mcf.Tail(edge) - 1] = mcf.Head(edge) - n - 1

        assert (c_cat >= 0).all()
        assert (c_cat < k).all()
        return c_cat.to(nll_p_all.device)


def _rdim(x: torch.Tensor) -> torch.Size:
    return x.shape[1:]


def _categorical_from_value(value: torch.Tensor) -> distributions.OneHotCategorical:
    logits = torch.where(value != 0.0, 0.0, -float("inf"))
    return distributions.OneHotCategorical(logits, value)
