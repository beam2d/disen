import dataclasses
import logging
import pathlib
from typing import Literal

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
        _estimate_mi(model, dataset, zi, arch="MLP", width=200, depth=4, n_epochs=2)
    )
    _logger.info(f"I(y_j; z_i)/H(y_j) =\n{mi_zi_yj}")

    _logger.info("approximating I(y_j; z_{-i})...")
    mi_zmi_yj = normalize(
        _estimate_mi(
            model,
            dataset,
            zmi,
            arch="DenseNet",
            width=200,
            depth=20,
            n_epochs=20,
        )
    )
    _logger.info(f"I(y_j; z_{{-i}})/H(y_j) =\n{mi_zmi_yj}")

    _logger.info("approximating I(y_j; z)...")
    mi_z_yj = normalize(
        _estimate_mi(
            model,
            dataset,
            z,
            arch="DenseNet",
            width=200,
            depth=30,
            n_epochs=20,
        )
    )
    _logger.info(f"I(y_j; z)/H(y_j) =\n{mi_z_yj}")

    ub_l = (mi_zi_yj - mi_zmi_yj).relu().amax(0).mean().item()
    ub_u = torch.minimum(mi_z_yj - mi_zmi_yj, mi_zi_yj).amax(0).mean().item()

    _logger.info(f"{ub_l=} {ub_u=}")
    return VariationalMIMetrics(mi_zi_yj, mi_zmi_yj, mi_z_yj, ub_l, ub_u)


def _estimate_mi(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    z_indices: torch.Tensor,
    arch: Literal["MLP", "DenseNet"] = "DenseNet",
    width: int = 200,
    depth: int = 6,
    batch_size: int = 1024,
    n_epochs: int = 20,
    lr: float = 1e-3,
) -> torch.Tensor:
    _logger.info(f"Variational bound of MI... (indices[0]={z_indices[0].tolist()})")
    model.eval()

    n = dataset.n_factors

    elem_maps = [
        torch.eye(latent.size).repeat_interleave(latent.n_categories, 1)
        for latent in model.spec
    ]
    i_to_elems = torch.block_diag(*elem_maps)
    elems = z_indices @ i_to_elems

    nn_gen = {"MLP": nn.MLP, "DenseNet": nn.DenseNet}[arch]
    clfs = [
        [
            nn_gen(int(elems_i.sum()), n_yj, width, depth).to(model.device)
            for n_yj in dataset.n_factor_values
        ]
        for elems_i in elems
    ]
    indices = [mask.nonzero()[:, 0].to(model.device) for mask in elems]

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    optimizers = [
        [torch.optim.Adam(clf.parameters(), lr=lr) for clf in clfs_i]
        for clfs_i in clfs
    ]

    for epoch in range(1, n_epochs + 1):
        for clfs_i in clfs:
            for clf in clfs_i:
                clf.train()

        accum = 0.0
        count = 0
        for x, y in train_loader:
            B = x.shape[0]
            x = x.to(model.device)
            y = y.to(model.device)
            z = torch.cat([q_z.sample().reshape(B, -1) for q_z in model.encode(x)], 1)

            total_loss = z.new_zeros(())
            for index, clf_i, optims_i in zip(indices, clfs, optimizers):
                z_Si = z[:, index]
                for yj, clf, optim in zip(y.T, clf_i, optims_i):
                    with torch.enable_grad():
                        clf.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(clf(z_Si), yj)
                        loss.backward()
                        optim.step()
                        total_loss += loss.detach()

            accum += float(total_loss) * B / (len(clfs) * n)
            count += B

        loss_avg = accum / count
        _logger.info(f"=== finshed epoch {epoch}/{n_epochs} --- loss={loss_avg}")

    _logger.info("Estimating...")
    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1
    )
    for clfs_i in clfs:
        for clf in clfs_i:
            clf.eval()

    batch_sums: list[torch.Tensor] = []
    count = 0
    for x, y in eval_loader:
        B = x.shape[0]
        x = x.to(model.device)
        y = y.to(model.device)
        z = torch.cat([q_z.sample().reshape(B, -1) for q_z in model.encode(x)], 1)

        xents: list[torch.Tensor] = []
        for index, clfs_i in zip(indices, clfs):
            z_Si = z[:, index]
            for yj, clf in zip(y.T, clfs_i):
                xent = F.cross_entropy(clf(z_Si), yj, reduction="sum")
                xents.append(xent)

        batch_sums.append(torch.stack(xents).reshape(len(clfs), n))
        count += B

    mean_xents = torch.stack(batch_sums).sum(0) / count
    H_yj = dataset.factor_entropies()
    return H_yj[None, :] - mean_xents.cpu()
