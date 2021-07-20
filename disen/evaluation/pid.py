import json
import logging
import math
import pathlib
from typing import Sequence

import torch
import torch.nn.functional as F

from .. import data, models


_logger = logging.getLogger(__name__)


@torch.no_grad()
def ri_zi_zmi_yj(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    out_dir: pathlib.Path,
) -> torch.Tensor:
    """Estimate [R(y_j; z_i, z_{-i})]_{ij}."""

    DZ = _train_ratio_estimator(model, dataset, out_dir)
    L0 = 10_000
    K = 32
    B0 = 16
    m = model.spec.size
    n = dataset.n_factors
    numel_z = model.spec.numel

    ri_yj_zi: list[torch.Tensor] = []

    for j in range(n):
        n_yj = dataset.n_factor_values[j]
        L_sub = ((L0 - 1) // (n_yj * B0) + 1) * B0

        summands1: list[torch.Tensor] = []
        summands2: list[torch.Tensor] = []

        for yj in range(n_yj):
            shuffle_subset = dataset.sample_stratified(L_sub)
            shuffle_loader = torch.utils.data.DataLoader(
                shuffle_subset, batch_size=B0, shuffle=True, num_workers=1
            )
            subset = data.subsample(dataset.fix_factor(j, yj), L_sub * K)
            loader = torch.utils.data.DataLoader(
                subset, batch_size=B0 * K, num_workers=1
            )

            for (x0, _), (xk, _) in zip(shuffle_loader, loader):
                B = x0.shape[0]
                assert B * K == xk.shape[0]

                z0 = [z.sample() for z in model.encode(x0.to(model.device))]
                zk = [z.sample() for z in model.encode(xk.to(model.device))]
                z_mixed1, z_mixed2 = _mix_repr(z0, zk)

                def compute_summand(z_mixed: torch.Tensor) -> torch.Tensor:
                    assert z_mixed.shape == (m, B, K, numel_z)
                    logit_flat = DZ(z_mixed.reshape(m * B * K, numel_z))
                    logit = logit_flat.reshape(m, B, K, m + 1)
                    logit_0 = logit[..., -1]
                    logit_i = logit.movedim(-1, 1).diagonal().movedim(-1, 0)
                    log_rho = logit_0 - logit_i
                    log_rho_mean = log_rho.logsumexp(2) - math.log(K)
                    return log_rho_mean * log_rho_mean.exp()

                summands1.append(compute_summand(z_mixed1))
                summands2.append(compute_summand(z_mixed2))

        mi_path1 = torch.cat(summands1, 1).mean(1)
        mi_path2 = torch.cat(summands2, 1).mean(1)
        ri_yj_zi.append(torch.minimum(mi_path1, mi_path2))

    return torch.stack(ri_yj_zi, 1)


def _train_ratio_estimator(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    out_dir: pathlib.Path,
) -> models.Discriminator:
    _logger.info("start train density ratio estimator for PUI...")
    m = model.spec.size
    B_each = 32
    B = B_each * (m + 1)
    n_epochs = 10

    in_size = model.spec.numel
    clf = models.Discriminator(in_size, m + 1, width=200, depth=4)
    clf.to(model.device)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=B, shuffle=True, num_workers=1, drop_last=True
    )
    optimizer = torch.optim.Adam(clf.parameters(), lr=5e-4)

    def evaluate() -> dict[str, float]:
        clf.eval()
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=B, num_workers=1, drop_last=True
        )
        n_pos = 0
        accum_loss = torch.zeros((), dtype=torch.float32, device=model.device)
        for x, _ in test_loader:
            zs = [q_z.sample() for q_z in model.encode(x.to(model.device))]
            z, t = _prepare_perm(zs)
            y = clf(z)
            n_pos += (y.argmax(1) == t).sum().item()
            accum_loss += F.cross_entropy(y, t) * x.shape[0]
        N = len(dataset)
        return {"acc": n_pos / N, "loss": accum_loss.item() / N}

    epoch_accs: list[float] = []
    epoch_losses: list[float] = []
    model.eval()

    for epoch in range(n_epochs):
        clf.train()
        for x, _ in train_loader:
            zs = [q_z.sample() for q_z in model.encode(x.to(model.device))]
            z, t = _prepare_perm(zs)
            with torch.enable_grad():
                clf.zero_grad(set_to_none=True)
                y = clf(z)
                loss = F.cross_entropy(y, t)
                loss.backward()
                optimizer.step()
        d = evaluate()
        epoch_accs.append(d["acc"])
        epoch_losses.append(d["loss"])
        _logger.info(f"[epoch={epoch + 1}/{n_epochs}] {d}")

    with open(out_dir / "ratio_estimator_accuracies.json", "w") as f:
        json.dump({"acc": epoch_accs, "loss": epoch_losses}, f, indent=4)

    return clf


# This function modifies zs in place.
def _prepare_perm(zs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    m = sum(z.shape[1] for z in zs)
    B = zs[0].shape[0]
    assert B % (m + 1) == 0
    B_each = B // (m + 1)

    for i in range(m):
        start = B_each * i
        end = start + B_each
        z = zs[0]
        for z1 in zs:
            z = z1
            if i < z1.shape[1]:
                break
            i -= z1.shape[1]
        perm = torch.randperm(B_each, device=z.device)
        z[start:end, i] = z[start:end, i][perm]

    z_perm = torch.cat([z.reshape(B, -1) for z in zs], 1)
    t = torch.arange(m + 1, device=z_perm.device)
    return (z_perm, t.repeat_interleave(B_each))


def _mix_repr(
    zs0: Sequence[torch.Tensor], zsk: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(zs0) == len(zsk)
    assert all(z0.shape[1] == zk.shape[1] for z0, zk in zip(zs0, zsk))
    m = sum(z0.shape[1] for z0 in zs0)
    B = zs0[0].shape[0]
    BK = zsk[0].shape[0]
    assert BK % B == 0
    K = BK // B

    mixed1: list[torch.Tensor] = []
    mixed2: list[torch.Tensor] = []

    for i in range(m):
        mixed1_i: list[torch.Tensor] = []
        mixed2_i: list[torch.Tensor] = []

        start = 0
        for z0, zk in zip(zs0, zsk):
            end = start + z0.shape[1]
            if start <= i < end:
                di = i - start
                zk1 = zk.reshape(z0.shape[0], -1, *zk.shape[1:]).clone()
                zk2 = z0[:, None].expand_as(zk1).clone()
                zk2[:, :, di] = zk1[:, :, di]
                zk1[:, :, di] = z0[:, None, di]
                zk1 = zk1.reshape(B, K, -1)
                zk2 = zk2.reshape(B, K, -1)
            else:
                zk1 = zk.reshape(B, K, -1)
                zk2 = z0.reshape(B, 1, -1).expand_as(zk1)
            mixed1_i.append(zk1)
            mixed2_i.append(zk2)
            start = end
        mixed1.append(torch.cat(mixed1_i, -1))
        mixed2.append(torch.cat(mixed2_i, -1))
    # shape: (m, B, K, latents)
    # note: latents may include one-hot encoding of discrete variables
    return torch.stack(mixed1), torch.stack(mixed2)
