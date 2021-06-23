import pathlib
from typing import Optional

import torch

from .. import evaluation, models


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
) -> evaluation.Result:
    result = evaluation.Result([], {})
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
        print(f"epoch: {epoch}...")

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
            result.history.append(entry)

        result.plot_history(out_dir)

        if n_epochs and epoch >= n_epochs:
            break

    return result
