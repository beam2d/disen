import pathlib

import torch

from .. import evaluation, models


def train_model(
    model: models.LatentVariableModel,
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    eval_batch_size: int,
    n_epochs: int,
    out_dir: pathlib.Path,
    num_workers: int = 1,
) -> evaluation.Result:
    result = evaluation.Result([], {})
    iteration = 0

    loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset, eval_batch_size, num_workers=num_workers
    )

    for epoch in range(n_epochs):
        print(f"epoc: {epoch + 1}...")
        model.train()
        for x, _ in loader:
            model.zero_grad(set_to_none=True)
            d = model(x.to(model.device))
            d["loss"].mean().backward()
            optimizer.step()
            iteration += 1

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

    return result
