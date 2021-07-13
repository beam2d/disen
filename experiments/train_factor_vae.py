import argparse
import logging
import pathlib
from typing import Optional

import torch

import disen


_logger = logging.getLogger(__name__)


def _train_model(
    model: disen.models.FactorVAE,
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
) -> disen.evaluation.Result:
    result = disen.evaluation.Result.new()
    iteration = 0
    epoch = 0
    assert n_epochs or n_iters

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
            accum = disen.evaluation.BatchAccumulator()
            for x, _ in eval_loader:
                d = model(x.to(model.device))
                assert "z" not in d
                accum.accumulate(d)

            entry = accum.mean()
            entry["iteration"] = iteration
            result.add_epoch(entry)

        result.plot_history(out_dir)

        if n_epochs and epoch >= n_epochs:
            break

    return result


def train_factor_vae(
    dataset_path: pathlib.Path, device: str, out_dir: pathlib.Path
) -> None:
    disen.setup_logger(out_dir)

    n_latents = 6

    dataset = disen.data.DSprites(dataset_path)
    image_size = dataset[0][0].shape[-1]
    encoder = disen.nn.SimpleConvNet(image_size, 1, 256)
    decoder = disen.nn.SimpleTransposedConvNet(image_size, n_latents, 1)
    discr = disen.models.FactorVAEDiscriminator(n_latents)
    model = disen.models.FactorVAE(encoder, decoder, discr, gamma=35.0)
    model.to(device)
    model.D.to(device)

    result = _train_model(
        model,
        dataset,
        batch_size=64,
        eval_batch_size=1024,
        lr=1e-4,
        betas=(0.9, 0.999),
        d_lr=1e-4,
        d_betas=(0.5, 0.9),
        out_dir=out_dir,
        n_iters=int(3e5),
    )
    disen.evaluation.evaluate_model(model, dataset, result, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    train_factor_vae(
        pathlib.Path(args.dataset), args.device, pathlib.Path(args.out_dir)
    )


if __name__ == "__main__":
    main()
