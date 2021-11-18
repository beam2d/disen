#!/usr/bin/env python3
import argparse
import pathlib
from typing import Sequence, cast

import torch
import torchvision

import disen


def _find_y_z_mapping(experiments: Sequence[disen.Experiment]) -> torch.Tensor:
    ubs: list[torch.Tensor] = []
    for experiment in experiments:
        mi = experiment.load_mi_metrics()
        ubs.append(mi["mi_zi_yj"] - mi["mi_zmi_yj"])
    return disen.nn.tensor_sum(ubs).argmax(0)


def _learn_coeffs(
    dataset: disen.data.DatasetWithFactors,
    model: disen.models.LatentVariableModel,
    mapping: torch.Tensor,
) -> torch.Tensor:
    """Trains linear regression that predicts yk from zmi.

    For K factors and L latents, it returns (K, L + 1) matrix. For each k, the element
    at (k, mapping[k]) is zero. The rightmost column is the bias.
    """
    q_all, y_all = model.encode_dataset(dataset)
    # sample z; attach 1 corresponding to bias
    z_all = torch.cat([q.sample() for q in q_all], 1)
    z_all = torch.cat([z_all, z_all.new_ones((len(z_all), 1))], 1)
    # normalize
    y_all = y_all / (torch.as_tensor(dataset.n_factor_values) - 1).to(y_all.device)
    y_all = y_all * 2 - 1

    weights: list[torch.Tensor] = []
    for k in range(dataset.n_factors):
        ell = int(mapping[k])
        z = z_all.clone()
        z[ell] = 0.0
        wk = torch.linalg.lstsq(z, y_all[:, k]).solution
        wk[ell] = 0.0
        err = (z @ wk - y_all[:, k]).abs().mean().item()
        print(f"factor {k}: ell = {ell}, err = {err}")
        weights.append(wk.to("cpu"))
    return torch.stack(weights)


@torch.no_grad()
def plot_latent_traversal(
    experiments: Sequence[disen.Experiment],
    device: torch.device,
    load_coeffs: bool = False,
) -> None:
    experiment = experiments[0]
    dir_path = experiment.get_model_dir() / "traversal"
    dir_path.mkdir(exist_ok=True)

    dataset = experiment.load_dataset()
    model = experiment.load_model()
    model.to(device)
    model.eval()

    mapping_path = dir_path / "mapping.pt"
    coeff_path = dir_path / "coeff.pt"
    if load_coeffs:
        y_to_z = torch.load(mapping_path)
        coeff = torch.load(coeff_path)
    else:
        y_to_z = _find_y_z_mapping(experiments)
        coeff = _learn_coeffs(dataset, model, y_to_z)
        torch.save(y_to_z, mapping_path)
        torch.save(coeff, coeff_path)

    N = 8
    prior = model.prior(N)
    z_orig = torch.cat([p.sample() for p in prior], 1)

    # axis-aligned traversal
    L = z_orig.shape[1]
    K = dataset.n_factors

    n_traversal = 10
    trajectory = torch.linspace(-2.0, 2.0, n_traversal, device=device)
    z = z_orig[:, None, None].expand((N, n_traversal, L, L)).clone()
    z.diagonal(dim1=-2, dim2=-1)[:] = trajectory[:, None]
    z = z.permute(0, 2, 1, 3).reshape(-1, L)
    x = model.decode([z]).mean.to("cpu")
    img_shape = x.shape[1:]
    x_axis = x.reshape(N, L * n_traversal, *img_shape)
    x_facs = x.reshape(N, L, n_traversal, *img_shape)[:, y_to_z]
    x_facs = x_facs.reshape(N, K * n_traversal, *img_shape)
    for i, xi in enumerate(x_axis):
        grid = torchvision.utils.make_grid(xi, n_traversal, pad_value=0.5)
        torchvision.utils.save_image(grid, dir_path / f"traversal_z_x{i}.png")

    # traversal along direction defined by least square coefficients
    coeff = coeff.to(device)
    weight = coeff[:, :-1]  # (K, L)
    w = weight * (1.0 / abs(weight).amax(1, keepdim=True))
    z_bias = torch.cat([z_orig, z_orig.new_ones((N, 1))], 1)
    y_pred = z_bias @ coeff.T  # (N, K)
    z_centered = z_orig[:, None] - y_pred[:, :, None] * w  # (N, K, L)
    z = z_centered[:, :, None] + trajectory[:, None] * w[:, None]
    z = z.reshape(-1, L)
    x = model.decode([z]).mean.to("cpu")
    x_regr = x.reshape(N, K * n_traversal, *img_shape)
    for i, (xi_f, xi_r) in enumerate(zip(x_facs, x_regr)):
        xi_f = xi_f.reshape(K, n_traversal, *img_shape)
        xi_r = xi_r.reshape(K, n_traversal, *img_shape)
        rows: list[torch.Tensor] = []
        for xi_f_k, xi_r_k in zip(xi_f, xi_r):
            xi_k = torch.cat([xi_f_k, xi_r_k], 0)
            grid_k = torchvision.utils.make_grid(
                xi_k, n_traversal, padding=1, pad_value=0.5
            )
            rows.append(grid_k)
        grid = torchvision.utils.make_grid(rows, 1, padding=4, pad_value=1.0)
        torchvision.utils.save_image(grid, dir_path / f"traversal_y_x{i}.png")


def main() -> None:
    parser = argparse.ArgumentParser("Plot latent traversal")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_seed", required=True, type=int)
    parser.add_argument("--eval_seed", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beta", type=disen.parse_optional(float))
    parser.add_argument("--load_coeffs", action="store_true")
    args = parser.parse_args()

    experiments = [
        disen.Experiment(
            pathlib.Path(args.out_dir),
            pathlib.Path(args.dataset_path),
            cast(disen.TaskType, args.task),
            cast(disen.ModelType, args.model),
            args.train_seed,
            "eval",
            int(eval_seed),
            beta=args.beta,
        )
        for eval_seed in args.eval_seed.split(",")
    ]
    plot_latent_traversal(experiments, torch.device(args.device), args.load_coeffs)


if __name__ == "__main__":
    main()
