import logging
import pathlib
from typing import Optional

import torch

from .. import evaluation, models
from . import history


_logger = logging.getLogger(__name__)


def train_model(
    model: models.LatentVariableModel,
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    eval_batch_size: int,
    out_dir: pathlib.Path,
    n_epochs: Optional[int] = None,
    n_iters: Optional[int] = None,
    num_workers: int = 1,
) -> history.History:
    h = history.History()
    iteration = 0
    epoch = 0
    assert n_epochs or n_iters

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset, eval_batch_size, num_workers=num_workers
    )

    finished = False

    while not finished:
        epoch += 1
        _logger.info(f"epoch: {epoch}...")

        model.train()
        for x, _ in loader:
            model.zero_grad(set_to_none=True)
            d = model(x.to(model.device))
            d["loss"].mean().backward()
            optimizer.step()
            iteration += 1
            if n_iters and iteration >= n_iters:
                finished = True
                break

        model.eval()
        with torch.no_grad():
            accum = evaluation.BatchAccumulator()
            for x, _ in eval_loader:
                d = model(x.to(model.device))
                accum.accumulate(d)

            entry = accum.mean()
            entry["iteration"] = iteration
            h.add_epoch(entry)

        h.plot(out_dir)

        if n_epochs and epoch >= n_epochs:
            break

    return h


def train_factor_vae(
    model: models.FactorVAE,
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    betas: tuple[float, float],
    d_lr: float,
    d_betas: tuple[float, float],
    out_dir: pathlib.Path,
    n_epochs: Optional[int] = None,
    n_iters: Optional[int] = None,
    num_workers: int = 1,
) -> history.History:
    h = history.History()
    iteration = 0
    epoch = 0
    assert n_epochs or n_iters

    model.D.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr, betas)
    d_optimizer = torch.optim.Adam(model.D.parameters(), d_lr, d_betas)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    d_loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset, eval_batch_size, num_workers=num_workers
    )

    finished = False

    while not finished:
        epoch += 1
        _logger.info(f"epoch: {epoch}...")

        model.train()
        model.D.train()
        for (x0, _), (x1, _) in zip(loader, d_loader):
            x0 = x0.to(model.device)
            x1 = x1.to(model.device)

            model.zero_grad(set_to_none=True)
            d = model(x0)
            d["loss"].mean().backward()
            optimizer.step()

            model.D.zero_grad(set_to_none=True)
            (q_z0,) = model.encode(x0)
            z0 = q_z0.sample()
            d_loss = model.discriminator_loss(x1.to(model.device), z0)
            d_loss.mean().backward()
            d_optimizer.step()

            iteration += 1
            if n_iters and iteration >= n_iters:
                finished = True
                break

        model.eval()
        model.D.eval()
        with torch.no_grad():
            accum = evaluation.BatchAccumulator()
            for x, _ in eval_loader:
                d = model(x.to(model.device))
                assert "z" not in d
                accum.accumulate(d)

            entry = accum.mean()
            entry["iteration"] = iteration
            h.add_epoch(entry)

        h.plot(out_dir)

        if n_epochs and epoch >= n_epochs:
            break

    return h
