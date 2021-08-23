import dataclasses
import logging
import pathlib
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .. import data, log_util, models, nn


_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VariationalMIMetrics:
    mi_zi_yj: torch.Tensor
    mi_zmi_yj: torch.Tensor
    mi_z_yj: torch.Tensor
    unibound_l: float  # unibound (lower)
    unibound_u: float  # unibound (upper)

    def get_scores(self) -> dict[str, float]:
        return {
            "unibound_lv": self.unibound_l,
            "unibound_uv": self.unibound_u,
        }

    def save(self, path: pathlib.Path) -> None:
        with log_util.torch_sci_mode_disabled():
            with open(path, "w") as f:
                f.write(f"mi_zi_yj=\n{self.mi_zi_yj}\n")
                f.write(f"mi_zmi_yj=\n{self.mi_zmi_yj}\n")
                f.write(f"mi_z_yj=\n{self.mi_z_yj}\n")
                f.write(f"unibound_l={self.unibound_l}\n")
                f.write(f"unibound_u={self.unibound_u}\n")


@log_util.torch_sci_mode_disabled()
@torch.no_grad()
def variational_mi_metrics(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
) -> VariationalMIMetrics:
    m = model.spec.size
    zi = torch.eye(m)
    zmi = torch.ones_like(zi) - zi
    z = torch.ones((1, m))

    def normalize(s: torch.Tensor) -> torch.Tensor:
        return (s / dataset.factor_entropies()).clip(0, 1)

    _logger.info("approximating I(y_j; z_i)...")
    mi_zi_yj = normalize(
        _estimate_mi(
            model,
            dataset,
            zi,
            lambda n_in, n_out: nn.MLP(n_in, n_out, 200, 4),
            n_epochs=10,
            init_lr=3e-3,
            target_lr=5e-4,
        )
    )
    _logger.info(f"I(y_j; z_i)/H(y_j) =\n{mi_zi_yj}")

    _logger.info("approximating I(y_j; z_{-i})...")
    mi_zmi_yj = normalize(
        _estimate_mi(
            model,
            dataset,
            zmi,
            lambda n_in, n_out: nn.DenseNet(n_in, n_out, 200, 15),
            n_epochs=50,
            init_lr=2e-3,
            target_lr=1e-4,
        )
    )
    _logger.info(f"I(y_j; z_{{-i}})/H(y_j) =\n{mi_zmi_yj}")

    _logger.info("approximating I(y_j; z)...")
    mi_z_yj = normalize(
        _estimate_mi(
            model,
            dataset,
            z,
            lambda n_in, n_out: nn.DenseNet(n_in, n_out, 200, 15),
            n_epochs=50,
            init_lr=2e-3,
            target_lr=1e-4,
        )
    )
    _logger.info(f"I(y_j; z)/H(y_j) =\n{mi_z_yj}")

    ub_l = (mi_zi_yj - mi_zmi_yj).relu().amax(0).mean().item()
    ub_u = torch.minimum(mi_z_yj - mi_zmi_yj, mi_zi_yj).amax(0).mean().item()

    _logger.info(f"{ub_l=} {ub_u=}")
    return VariationalMIMetrics(mi_zi_yj, mi_zmi_yj, mi_z_yj, ub_l, ub_u)


class MultitaskClassifier(torch.nn.Module):
    def __init__(
        self,
        spec: models.LatentSpec,
        z_masks: torch.Tensor,
        y_categories: Sequence[int],
        arch: Callable[[int, int], torch.nn.Module],
    ) -> None:
        super().__init__()
        elem_maps = [
            torch.eye(latent.size).repeat_interleave(latent.n_categories, 1)
            for latent in spec
        ]
        i_to_elems = torch.block_diag(*elem_maps)
        elems = z_masks @ i_to_elems
        self.z_indices = nn.BufferList([mask.nonzero()[:, 0] for mask in elems])

        self.y_categories = y_categories
        n_y_elems = sum(y_categories)
        self.clfs = torch.nn.ModuleList(
            [arch(int(elems_i.sum()), n_y_elems) for elems_i in elems]
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = len(self.z_indices)
        n = len(self.y_categories)

        xents: list[torch.Tensor] = []
        for index, clf in zip(self.z_indices, self.clfs):
            z_Si = z[:, index]
            y_pred = clf(z_Si)
            yj_pred = y_pred.split(self.y_categories, 1)
            for j in range(n):
                loss = F.cross_entropy(yj_pred[j], y[:, j])
                xents.append(loss)
        return torch.stack(xents).reshape(m, n)


def _estimate_mi(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    z_masks: torch.Tensor,
    arch: Callable[[int, int], torch.nn.Module],
    n_epochs: int,
    batch_size: int = 1024,
    init_lr: float = 1e-3,
    target_lr: float = 0.0,
    eval_sample_size: int = 5,
) -> torch.Tensor:
    _logger.info(f"Variational bound of MI... (indices[0]={z_masks[0].tolist()})")
    model.eval()

    N = len(dataset)
    m = z_masks.shape[0]
    n = dataset.n_factors
    epoch_inference = models.EpochInference(model, dataset)

    estimator = MultitaskClassifier(model.spec, z_masks, dataset.n_factor_values, arch)
    estimator.to(model.device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=init_lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_epochs, target_lr
    )

    for epoch in range(n_epochs):
        estimator.train()
        z_all, y_all = epoch_inference.next_epoch()
        perm = torch.randperm(N).to(model.device)
        accum = z_all.new_zeros(())

        for i in range(0, N, batch_size):
            B = min(batch_size, N - i)
            z = z_all[perm[i : i + B]]
            y = y_all[perm[i : i + B]]

            with torch.enable_grad():
                estimator.zero_grad(set_to_none=True)
                loss = estimator(z, y).sum()
                loss.backward()
                optimizer.step()

            accum += loss * (B / (m * n))

        loss_avg = float(accum) / N
        last_lr = scheduler.get_last_lr()[0]
        _logger.info(
            f"=== finshed epoch {epoch + 1}/{n_epochs} --- loss={loss_avg} --- lr={last_lr}"
        )

        scheduler.step()

    _logger.info("Estimating...")
    estimator.eval()

    trial_sums: list[torch.Tensor] = []

    for _ in range(eval_sample_size):
        z_all, y_all = epoch_inference.next_epoch()
        xent_batches: list[torch.Tensor] = []

        for i in range(0, N, batch_size):
            B = min(batch_size, N - i)
            z = z_all[i : i + B]
            y = y_all[i : i + B]
            xents = estimator(z, y)
            xent_batches.append(xents * B)

        trial_sums.append(torch.stack(xent_batches).sum(0) / N)

    mean_xents = torch.stack(trial_sums).mean(0)
    H_yj = dataset.factor_entropies()
    return H_yj[None, :] - mean_xents.cpu()
