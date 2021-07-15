import pathlib
from typing import Sequence

import torch

from .. import attack, data, evaluation, models


def evaluate_model(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    result: evaluation.Result,
    out_dir: pathlib.Path,
    n_traversal: int = 12,
    alphas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> None:
    evaluation.render_latent_traversal(
        dataset, model, n_traversal, out_dir / "traversal"
    )

    D = model.spec.real_components_size
    U = (torch.eye(D) - 2 / D).to(model.device)

    for alpha in alphas:
        target = model if alpha == 0.0 else attack.RedundancyAttack(model, alpha, U)
        param = ("alpha", alpha)
        evaluation.evaluate_factor_vae_score(target, dataset, result, param=param)
        evaluation.evaluate_beta_vae_score(
            target, dataset, result, out_dir, param=param
        )
        evaluation.evaluate_mi_metrics(target, dataset, result, out_dir, param=param)

    result.save(out_dir / "result.json")
