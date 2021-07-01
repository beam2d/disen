import json
import pathlib

import torch
import torch.nn.functional as F

from .. import data, evaluation, models


def evaluate_beta_vae_score(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    result: evaluation.Result,
    out_dir: pathlib.Path,
) -> None:
    score = beta_vae_score(model, dataset, out_dir)
    result.add_metric("beta_vae_score", score)


@torch.no_grad()
def beta_vae_score(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    out_dir: pathlib.Path,
    sample_size: int = 200,
    eval_size: int = 800,
    batch_size: int = 10,
    lr: float = 0.01,
    n_iters: int = 10_000,
) -> float:
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
        zs = [z.reshape(*xs.shape[:3], model.spec.size) for z in zs]
        z_diffs = torch.cat([_l1_diff(z[:, :, 0], z[:, :, 1]) for z in zs], -1)
        return z_diffs.mean(1)

    def evaluate() -> float:
        test_set = data.DatasetWithCommonFactor(dataset, sample_size, 2, eval_size)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, num_workers=1
        )
        n_pos = 0
        for x, t in test_loader:
            z_diff_mean = embed_sample(x.to(device))
            y = classifier(z_diff_mean)
            n_pos += (y.argmax(1).cpu() == t).sum().item()
        return n_pos / eval_size

    epoch_accs: list[float] = []
    epoch_size = (n_iters - 1) // 10 + 1

    for i, (x, t) in enumerate(train_loader):
        if i % epoch_size == 0:
            epoch_accs.append(evaluate())

        z_diff_mean = embed_sample(x.to(device))
        with torch.enable_grad():
            classifier.zero_grad(set_to_none=True)
            y = classifier(z_diff_mean)
            loss = F.cross_entropy(y, t.to(device))
            loss.backward()
            optimizer.step()

    with open(out_dir / "beta_vae_accuracies.json", "w") as f:
        json.dump(epoch_accs, f, indent=4)

    return evaluate()


def _l1_diff(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    assert z1.shape == z2.shape
    z_diff = abs(z1 - z2)
    if z1.ndim >= 4:
        z_diff = z_diff.sum(tuple(range(3, z1.ndim)))
    return z_diff
