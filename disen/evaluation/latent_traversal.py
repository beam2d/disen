import pathlib
from typing import Any

import torch
import torchvision

from .. import models


def render_latent_traversal(
    dataset: torch.utils.data.Dataset[Any],
    model: models.LatentVariableModel,
    sample_size: int,
    out_dir: pathlib.Path,
) -> None:
    out_dir.mkdir(exist_ok=True)
    loader = torch.utils.data.DataLoader(
        dataset, sample_size, shuffle=True, num_workers=1
    )
    x = next(iter(loader))[0].to(model.device)
    q_zs = model.encode(x)
    zs = [q_z.sample() for q_z in q_zs]
    for i, latent in enumerate(model.spec):
        for k in range(latent.size):
            zs_trav = model.spec.generate_traversal(zs, i, k)
            x_trav = model.decode(zs_trav).mean
            nrow = x_trav.shape[0] // sample_size
            image = torchvision.utils.make_grid(
                x_trav, nrow, padding=1, pad_value=0.5
            )
            image = (image * 255).to(torch.uint8)
            torchvision.io.write_png(
                image.cpu(), str(out_dir / f"traversal_{latent.name}_{k}.png")
            )
