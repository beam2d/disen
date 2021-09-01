import json
import logging
import pathlib
from typing import Optional

import torch
import torch.nn.functional as F

from .. import data, models


_logger = logging.getLogger(__name__)


@torch.no_grad()
def beta_vae_score(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    sample_size: int = 200,
    eval_size: int = 800,
    batch_size: int = 10,
    lr: float = 1.,
    n_iters: int = 3_000,
) -> float:
    _logger.info("computing BetaVAE score...")
    model.eval()

    device = model.device
    classifier = torch.nn.Linear(model.spec.size, dataset.n_factors, device=device)
    optimizer = torch.optim.Adagrad(classifier.parameters(), lr=lr)

    n_train = n_iters * batch_size
    train_set = data.DatasetWithCommonFactor(dataset, sample_size, 2, n_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=1
    )

    def embed_sample(xs: torch.Tensor) -> torch.Tensor:
        zs = model.infer_mean(xs.reshape(-1, *xs.shape[3:]))
        zs = [z.reshape(*xs.shape[:3], *z.shape[1:]) for z in zs]
        z_diffs = torch.cat([_l1_diff(z[:, :, 0], z[:, :, 1]) for z in zs], -1)
        return z_diffs.mean(1)

    epoch_size = n_iters // 10
    epoch = 0
    loss_accum = 0.0
    count = 0

    for i, (x, t) in enumerate(train_loader):
        z_diff_mean = embed_sample(x.to(device))
        with torch.enable_grad():
            classifier.zero_grad(set_to_none=True)
            y = classifier(z_diff_mean)
            loss = F.cross_entropy(y, t.to(device))
            loss.backward()
            optimizer.step()

            loss_accum += loss.item() * x.shape[0]
            count += x.shape[0]

        if (i + 1) % epoch_size == 0:
            epoch += 1
            _logger.info(f"=== {epoch}/10  loss={loss_accum / count}")
            loss_accum = 0.0
            count = 0

    test_set = data.DatasetWithCommonFactor(dataset, sample_size, 2, eval_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, num_workers=1
    )
    n_pos = 0
    for x, t in test_loader:
        z_diff_mean = embed_sample(x.to(device))
        y = classifier(z_diff_mean)
        n_pos += (y.argmax(1).cpu() == t).sum().item()

    score = n_pos / eval_size
    _logger.info(f"beta vae score = {score}")
    return score


def _l1_diff(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    assert z1.shape == z2.shape
    z_diff = abs(z1 - z2)
    if z1.ndim >= 4:
        z_diff = z_diff.sum(tuple(range(3, z1.ndim)))
    return z_diff
