import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .. import data, models, nn


_logger = logging.getLogger(__name__)


def estimate_mi_difference(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    left_indices: torch.Tensor,
    right_indices: Optional[torch.Tensor] = None,
    D_width: int = 200,
    D_depth: int = 3,
    batch_size: int = 1024,
    n_epochs: int = 20,
    lr: float = 0.001,
) -> torch.Tensor:
    """Estimate the difference of two MIs by density ratio estimation.

    This function estimates
        I(y_j; z_{L_i}) - I(y_j; z_{R_i})
    where L_i and R_i are subsets of indices specified by ``left_indices`` and
    ``right_indices``, respectively. These are 0/1-valued float tensors that
    map ``i`` to L_i and R_i, respectively. When ``right_indices`` is ``None``,
    it is interpreted as a zero tensor (i.e., it always maps i to the empty
    set, nullifying the second MI term).

    The MI difference is estimated by transforming as
        I(y_j; z_{L_i}) - I(y_j; z_{R_i})
        = E log [p(y_j, z_{L_i})p(z_{R_i}) / p(z_{L_i})p(y_j, z_{R_i})].
    We first train a classifier to approximate
        f(y_j, z, z') = p(y_j, z_{L_i})p(z'_{R_i}) / p(z_{L_i})p(y_j, z'_{R_i})
    where z' is an i.i.d. copy of z, and then Monte-Carlo approximate the MI
    difference by averaging log f(y_j, z, z').
    """
    _logger.info("training density ratio estimator for MI...")
    model.eval()

    m = model.spec.size
    n = dataset.n_factors
    z_numel = model.spec.numel
    y_offsets = torch.as_tensor((0,) + dataset.n_factor_values[:-1]).cumsum(0)
    y_numel = sum(dataset.n_factor_values)

    i_to_elems = torch.block_diag(
        *[
            torch.eye(latent.size).repeat_interleave(latent.n_categories, 1)
            for latent in model.spec
        ]
    )
    L = (left_indices @ i_to_elems).to(model.device)
    R: Optional[torch.Tensor] = None
    if right_indices is not None:
        R = (right_indices @ i_to_elems).to(model.device)

    in_size = y_numel + z_numel + (0 if R is None else z_numel)
    clf = models.Discriminator(in_size, m, D_width, D_depth).to(model.device)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    for epoch in range(n_epochs):
        accum = 0.0
        for x, y in train_loader:
            B = x.shape[0]
            # (n, B, y_numel)
            y_embed = F.one_hot(y + y_offsets, y_numel).movedim(1, 0).to(model.device)
            y_expand = y_embed.expand(2, m, n, B, y_numel)

            with torch.no_grad():
                zs = [q_z.sample() for q_z in model.encode(x.to(model.device))]
            z = torch.cat([z.reshape(B, -1) for z in zs], 1)
            z_shuf_L = nn.shuffle(z, 0)
            # (m, B, z_numel)
            z_L0 = L[:, None, :] * z_shuf_L[None, :, :]
            z_L1 = L[:, None, :] * z[None, :, :]
            # (2, m, n, B, z_numel)
            z_L = torch.stack([z_L0, z_L1])[:, :, None].expand(2, m, n, B, -1)
            del z_shuf_L, z_L0, z_L1

            if R is None:
                yz = torch.cat([y_expand, z_L], -1)
            else:
                z_shuf_R = nn.shuffle(z, 0)
                # (m, B, z_numel)
                z_R0 = R[:, None, :] * z
                z_R1 = R[:, None, :] * z_shuf_R[None, :, :]
                # (2, m, n, B, z_numel)
                z_R = torch.stack([z_R0, z_R1])[:, :, None].expand(2, m, n, B, -1)
                yz = torch.cat([y_expand, z_L, z_R], -1)
                del z_shuf_R, z_R0, z_R1, z_R
            del z_L

            clf.zero_grad(set_to_none=True)
            # (2 * m * n * B, m)
            logits = clf(yz.reshape(2 * m * n * B, -1)).reshape(2, m, n, B, m)
            # (2, n, B, m)
            logits = logits.movedim(1, -2).diagonal(0, -2, -1)
            t = logits.new_tensor([0.0, 1.0])[:, None, None, None].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits.ravel(), t.ravel())
            loss.backward()
            optimizer.step()

            accum += loss.item() * B

        loss_avg = accum / len(dataset)
        _logger.info(f"=== finished epoch {epoch + 1}/{n_epochs} --- loss={loss_avg}")

    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1
    )

    with torch.no_grad():
        _logger.info("estimating MI...")
        batch_sums: list[torch.Tensor] = []
        count = 0

        for x, y in eval_loader:
            B = x.shape[0]
            # (n, B, y_numel)
            y_embed = F.one_hot(y + y_offsets, y_numel).movedim(1, 0).to(model.device)
            y_expand = y_embed.expand(m, n, B, y_numel)

            zs = [q_z.sample() for q_z in model.encode(x.to(model.device))]
            z = torch.cat([z.reshape(B, -1) for z in zs], 1)
            # (m, B, z_numel)
            z_L1 = L[:, None, :] * z[None, :, :]
            z_L = z_L1[:, None].expand(m, n, B, z_numel)

            if R is None:
                yz = torch.cat([y_expand, z_L], -1)
            else:
                z_R0 = R[:, None, :] * z[None, :, :]
                z_R = z_R0[:, None].expand(m, n, B, z_numel)
                yz = torch.cat([y_expand, z_L, z_R], -1)
                del z_R0, z_R
            del z_L1, z_L

            logits = clf(yz.reshape(m * n * B, -1)).reshape(m, n, B, m)
            # (B, m, n)
            logits = logits.movedim(0, -2).diagonal(0, -2, -1).movedim(0, -1)
            batch_sums.append(logits.sum(0))
            count += B

        return torch.stack(batch_sums).sum(0) / count


def unibound_lower(
    model: models.LatentVariableModel, dataset: data.DatasetWithFactors
) -> float:
    m = model.spec.size
    score_ij = estimate_mi_difference(
        model, dataset, torch.eye(m), torch.ones((m, m)) - torch.eye(m)
    )
    ent_j = torch.as_tensor(dataset.n_factor_values).log()
    ulbo = _normalize(score_ij, ent_j).amax(0).mean().item()
    _logger.info(f"ulbo = {ulbo}")
    return ulbo


def unibound_upper(
    model: models.LatentVariableModel, dataset: data.DatasetWithFactors
) -> float:
    m = model.spec.size
    ones = torch.ones((m, m))
    nondiags = ones - torch.eye(m)
    ent_j = torch.as_tensor(dataset.n_factor_values).log()
    score_ij = _normalize(estimate_mi_difference(model, dataset, ones, nondiags), ent_j)
    mi_ij = _normalize(estimate_mi_difference(model, dataset, torch.eye(m)), ent_j)
    uubo = torch.minimum(score_ij, mi_ij).amax(0).mean().item()
    _logger.info(f"uubo = {uubo}")
    return uubo


def _normalize(s_ij: torch.Tensor, ent_j: torch.Tensor) -> torch.Tensor:
    return (s_ij.cpu() / ent_j).clip(0, 1)
