from __future__ import annotations
from disen.models.discriminator import Discriminator
import logging
import pathlib
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from .. import data, log_util, models


_logger = logging.getLogger(__name__)


class SingleClassifier(torch.nn.Module):
    def __init__(
        self,
        left_indices: torch.Tensor,
        right_indices: Optional[torch.Tensor],
        factor_index: int,
        n_factor_categories: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        n_in_L = int(left_indices.sum())
        n_in_R = 0 if right_indices is None else int(right_indices.sum())
        n_in = n_factor_categories + n_in_L + n_in_R

        self.n_in_y = n_factor_categories
        self.n_in_L = n_in_L
        self.n_in_R = n_in_R
        self.mlp = Discriminator(n_in, 1, width, depth)

        self.indices_L: torch.Tensor
        self.register_buffer("indices_L", left_indices.nonzero()[:, 0])

        self.indices_R: Optional[torch.Tensor]
        if right_indices is not None:
            right_indices = right_indices.nonzero()[:, 0]
        self.register_buffer("indices_R", right_indices)

        self.factor_index = factor_index
        self.n_factor_categories = n_factor_categories

    def forward(
        self, y: torch.Tensor, z_L: torch.Tensor, z_R: Optional[torch.Tensor]
    ) -> torch.Tensor:
        y_j = F.one_hot(y[:, self.factor_index], self.n_factor_categories)
        z_L_i = z_L[:, self.indices_L]
        elems = [y_j, z_L_i]

        if self.indices_R is not None:
            assert z_R is not None
            z_R_i = z_R[:, self.indices_R]
            elems.append(z_R_i)

        yz = torch.cat(elems, 1)
        return self.mlp(yz)


class MultiClassifier(torch.nn.Module):
    def __init__(
        self,
        estimation_type: Literal["MI", "MIdiff"],
        left_indices: torch.Tensor,
        right_indices: Optional[torch.Tensor],
        n_factor_values: Sequence[int],
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.m = left_indices.shape[0]
        self.n = len(n_factor_values)

        if right_indices is None:
            inds_R: list[Optional[torch.Tensor]] = [None] * self.m
        else:
            inds_R = list(right_indices)

        self.clfs = torch.nn.ModuleList(
            [
                SingleClassifier(ind_L, ind_R, j, n_yj, width, depth)
                for ind_L, ind_R in zip(left_indices, inds_R)
                for j, n_yj in enumerate(n_factor_values)
            ]
        )
        self.has_R = right_indices is not None
        self.estimation_type = estimation_type

    def forward(
        self,
        y: torch.Tensor,
        zs_L: Sequence[torch.Tensor],
        zs_R: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        B = y.shape[0]
        z_L = torch.cat([z.reshape(B, -1) for z in zs_L], 1)
        z_R = torch.cat([z.reshape(B, -1) for z in zs_R], 1) if self.has_R else None
        logits = torch.stack([clf(y, z_L, z_R) for clf in self.clfs], 1)
        return logits.reshape(B, self.m, self.n)

    def compute_loss(
        self,
        y_L: torch.Tensor,
        y_R: torch.Tensor,
        zs_L: Sequence[torch.Tensor],
        zs_R: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if self.estimation_type == "MI":
            pos_logits = self(y_L, zs_L, zs_L)
            neg_logits = self(y_R, zs_L, zs_R)
        else:
            pos_logits = self(y_L, zs_L, zs_R)
            neg_logits = self(y_R, zs_L, zs_R)
        logits = torch.cat([pos_logits, neg_logits], 0)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        labels = torch.cat([pos_labels, neg_labels], 0)

        return F.binary_cross_entropy_with_logits(logits, labels)


def estimate_mi_difference(
    estimation_type: Literal["MI", "MIdiff"],
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    left_indices: torch.Tensor,
    right_indices: Optional[torch.Tensor] = None,
    width: int = 200,
    depth: int = 4,
    batch_size: int = 1024,
    n_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.,
) -> torch.Tensor:
    """Estimate the difference of two MIs by density ratio estimation."""
    _logger.info("training density ratio estimator for MI...")
    model.eval()

    i_to_elems = torch.block_diag(
        *[
            torch.eye(latent.size).repeat_interleave(latent.n_categories, 1)
            for latent in model.spec
        ]
    )
    elems_L = left_indices @ i_to_elems
    elems_R = None if right_indices is None else right_indices @ i_to_elems
    clf = MultiClassifier(
        estimation_type, elems_L, elems_R, dataset.n_factor_values, width, depth
    ).to(model.device)

    train_loader_L, train_loader_R = [
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        for _ in range(2)
    ]
    optimizer: torch.optim.Optimizer
    if weight_decay > 0.0:
        optimizer = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(n_epochs):
        clf.train()
        accum = 0.0
        for (x_L, y_L), (x_R, y_R) in zip(train_loader_L, train_loader_R):
            with torch.no_grad():
                zs_L = [q_z.sample() for q_z in model.encode(x_L.to(model.device))]
                zs_R: list[torch.Tensor] = []
                if right_indices is not None:
                    zs_R = [q_z.sample() for q_z in model.encode(x_R.to(model.device))]
                y_L = y_L.to(model.device)
                y_R = y_R.to(model.device)

            clf.zero_grad(set_to_none=True)
            loss = clf.compute_loss(y_L, y_R, zs_L, zs_R)
            loss.backward()
            optimizer.step()

            accum += loss.item() * x_L.shape[0]

        loss_avg = accum / len(dataset)
        _logger.info(f"=== finished epoch {epoch + 1}/{n_epochs} --- loss={loss_avg}")

    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1
    )

    with torch.no_grad():
        _logger.info("estimating MI...")
        clf.eval()
        batch_sums: list[torch.Tensor] = []
        count = 0

        for x, y in eval_loader:
            zs = [q_z.sample() for q_z in model.encode(x.to(model.device))]
            dr = clf(y.to(model.device), zs, zs)
            batch_sums.append(dr.sum(0))
            count += dr.shape[0]

        return torch.stack(batch_sums).sum(0) / count


def unibound_lower(
    model: models.LatentVariableModel, dataset: data.DatasetWithFactors
) -> float:
    m = model.spec.size
    zi = torch.eye(m)
    zmi = torch.ones_like(zi) - zi
    ent_j = torch.as_tensor(dataset.n_factor_values).log()
    ub_l = _normalize(
        estimate_mi_difference("MIdiff", model, dataset, zi, zmi), ent_j
    ).amax(0).mean().item()
    _logger.info(f"unibound_l_dre = {ub_l}")
    return ub_l


def unibound_upper(
    model: models.LatentVariableModel, dataset: data.DatasetWithFactors
) -> float:
    m = model.spec.size
    zi = torch.eye(m)
    zmi = torch.ones_like(zi) - zi
    z = torch.ones((1, m))
    ent_j = torch.as_tensor(dataset.n_factor_values).log()
    score_ij = _normalize(
        estimate_mi_difference("MIdiff", model, dataset, z, zmi), ent_j
    )
    mi_ij = _normalize(
        estimate_mi_difference("MI", model, dataset, zi), ent_j
    )
    ub_u = torch.minimum(score_ij, mi_ij).amax(0).mean().item()
    _logger.info(f"unibound_u_dre = {ub_u}")
    return ub_u


def estimate_unibound_in_many_ways(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    out_dir: pathlib.Path,
) -> dict[str, float]:
    m = model.spec.size
    zi = torch.eye(m)
    z = torch.ones((m, m))
    z_short = torch.ones((1, m))
    zmi = z - zi
    ent_j = torch.as_tensor(dataset.n_factor_values).log()

    ub_l_direct_est = _normalize(
        estimate_mi_difference("MIdiff", model, dataset, zi, zmi), ent_j
    )
    ub_l_direct = ub_l_direct_est.relu().amax(0).mean().item()
    _logger.info(f"=== {ub_l_direct=}")

    mi_i_j = _normalize(estimate_mi_difference("MI", model, dataset, zi), ent_j)
    mi_mi_j = _normalize(estimate_mi_difference("MI", model, dataset, zmi), ent_j)
    ub_l_mi = (mi_i_j - mi_mi_j).relu().amax(0).mean().item()
    _logger.info(f"=== {ub_l_mi=}")

    mi_i_jmi = _normalize(estimate_mi_difference("MI", model, dataset, zi, zmi), ent_j)
    mi_mi_ji = _normalize(estimate_mi_difference("MI", model, dataset, zmi, zi), ent_j)
    ub_l_part = (mi_i_jmi - mi_mi_ji).relu().amax(0).mean().item()
    _logger.info(f"=== {ub_l_part=}")

    ub_u_direct_est = _normalize(
        estimate_mi_difference("MIdiff", model, dataset, z, zmi), ent_j
    )
    ub_u_direct = torch.minimum(ub_u_direct_est, mi_i_j).amax(0).mean().item()
    _logger.info(f"=== {ub_u_direct=}")

    mi_z_j = _normalize(estimate_mi_difference("MI", model, dataset, z_short), ent_j)
    ub_u_mi = torch.minimum(mi_z_j - mi_mi_j, mi_i_j).amax(0).mean().item()
    _logger.info(f"=== {ub_u_mi=}")

    with log_util.torch_sci_mode_disabled():
        with open(out_dir / "dre_metrics.txt", "w") as f:
            f.write(f"mi_zi_yj =\n{mi_i_j}\n")
            f.write(f"mi_zmi_yj =\n{mi_mi_j}\n")
            f.write(f"mi_z_yj = {mi_z_j}\n")
            f.write("\n")
            f.write(f"mi_zi_yjzmi =\n{mi_i_jmi}\n")
            f.write(f"mi_zmi_yjzi =\n{mi_mi_ji}\n")
            f.write("\n")
            f.write(f"ub_l_direct = {ub_l_direct}\n")
            f.write(f"ub_l_mi     = {ub_l_mi}\n")
            f.write(f"ub_l_part   = {ub_l_part}\n")
            f.write(f"ub_u_direct = {ub_u_direct}\n")
            f.write(f"ub_u_mi     = {ub_u_mi}\n")

    return {
        "unibound_l_direct": ub_l_direct,
        "unibound_l_mi": ub_l_mi,
        "unibound_l_part": ub_l_part,
        "unibound_u_direct": ub_u_direct,
        "unibound_u_mi": ub_u_mi,
    }


def _normalize(s_ij: torch.Tensor, ent_j: torch.Tensor) -> torch.Tensor:
    return (s_ij.cpu() / ent_j).clip(0, 1)
