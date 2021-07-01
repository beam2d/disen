import itertools
from typing import Iterable

import torch

from .. import data, evaluation, models


def evaluate_factor_vae_score(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    result: evaluation.Result,
) -> None:
    score = factor_vae_score(model, dataset)
    result.add_metric("factor_vae_score", score)


@torch.no_grad()
def factor_vae_score(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    n_train: int = 800,
    n_test: int = 800,
    sample_size: int = 100,
    n_normalizer_data: int = 65536,
) -> float:
    model.eval()
    n_factors = dataset.n_factors
    n_latents = model.spec.size
    batch_size = 10

    normalizer = _compute_normalizer(model, dataset, n_normalizer_data)

    def embed_sample(x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(batch_size * sample_size, *x.shape[3:])
        zs_flat = model.infer_mean(x_flat.to(model.device))
        zs = [z.reshape(batch_size, sample_size, n_latents) for z in zs_flat]
        zs_var = _compute_variance(zs, model.spec, 1) / normalizer
        return zs_var.argmin(1).cpu()

    # Train the majority vote classifier
    train_set = data.DatasetWithCommonFactor(dataset, 1, sample_size, n_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=1
    )
    votes = torch.zeros((n_latents, n_factors), dtype=torch.int32)
    for x, j in train_loader:
        i = embed_sample(x)
        for i0, j0 in zip(i.tolist(), j.tolist()):
            votes[i0, j0] += 1
    classifier = votes.argmax(1)

    # Evaluate
    test_set = data.DatasetWithCommonFactor(dataset, 1, sample_size, n_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, num_workers=1
    )
    n_pos = 0
    for x, j in test_loader:
        i = embed_sample(x)
        n_pos += (classifier[i] == j).sum()
    return n_pos / n_test


def _compute_normalizer(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    n_data: int,
) -> torch.Tensor:
    batch_size = 128
    assert n_data % batch_size == 0
    n_iters = n_data // batch_size
    loader_to_normalize = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    
    zs_batches = [
        model.infer_mean(batch[0].to(model.device))
        for batch in itertools.islice(loader_to_normalize, n_iters)
    ]
    zs_cat = [torch.cat(z_batches, 0) for z_batches in zip(*zs_batches)]
    return _compute_variance(zs_cat, model.spec, 0)


def _compute_variance(
    zs: Iterable[torch.Tensor], spec: models.LatentSpec, batch_dims: int
) -> torch.Tensor:
    zs_var: list[torch.Tensor] = []
    for z, z_spec in zip(zs, spec):
        if z_spec.domain == "real":
            assert z.ndim == batch_dims + 2
            zs_var.append(z.var(batch_dims))
        elif z_spec.domain == "categorical":
            # Use Gini's definition of empirical variance
            assert z.ndim == batch_dims + 3
            N = z.shape[batch_dims]
            c = z.argmax(batch_dims + 2)
            c_var = (c[..., :, None] == c[..., None, :]).sum() / (2 * N * (N - 1))
            zs_var.append(c_var)
        else:
            raise ValueError("invalid latent domain")
    return torch.cat(zs_var, -1)