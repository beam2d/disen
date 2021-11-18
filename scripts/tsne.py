#!/usr/bin/env python3
import argparse
import pathlib
from typing import Sequence, cast

from matplotlib import pyplot
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy
import seaborn
import torch
import torchvision

import disen


def _find_y_z_mapping(experiments: Sequence[disen.Experiment]) -> torch.Tensor:
    ubs: list[torch.Tensor] = []
    for experiment in experiments:
        mi = experiment.load_mi_metrics()
        ubs.append(mi["mi_zi_yj"] - mi["mi_zmi_yj"])
    return disen.nn.tensor_sum(ubs).argmax(0)


def _make_dataset(
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    model: disen.models.LatentVariableModel,
    k: int,
    ell: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a dataset of z_{\ ell}."""
    q_all, y_all = model.encode_dataset(dataset)
    # sample z
    z_all = torch.cat([q.sample() for q in q_all], 1)
    # exclude ell
    z_all = torch.cat([z_all[:, :ell], z_all[:, ell + 1 :]], 1)
    return (z_all, y_all[:, k])


@torch.no_grad()
def plot_tsne(experiments: Sequence[disen.Experiment], device: torch.device) -> None:
    seaborn.set_theme("paper", "darkgrid", "muted")

    experiment = experiments[0]
    dir_path = experiment.get_model_dir() / "tsne"
    dir_path.mkdir(exist_ok=True)

    dataset = experiment.load_dataset()
    sample = dataset.sample_stratified(len(dataset) // 10)
    model = experiment.load_model()
    model.to(device)
    model.eval()

    q_all, y_all = model.encode_dataset(sample)
    z_all = torch.cat([q.sample() for q in q_all], 1)

    y_to_z = _find_y_z_mapping(experiments)
    for k in range(dataset.n_factors):
        print(f"factor {k}...")
        ell = int(y_to_z[k])
        yk_max = dataset.n_factor_values[k] - 1
        zk = torch.cat([z_all[:, :ell], z_all[:, ell + 1:]], 1).to("cpu")
        yk = y_all[:, k].to("cpu")

        tsne = TSNE(n_jobs=8)
        embed = tsne.fit_transform(zk)
        subset = numpy.random.choice(len(embed), size=500, replace=False)
        xy = embed[subset]
        hue = yk[subset]
        fg = seaborn.relplot(x=xy[:, 0], y=xy[:, 1], hue=subset)
        fg.figure.savefig(dir_path / f"tsne_y{k}.png", bbox_inches="tight")
        pyplot.close(fg.fig)
        del fg


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
    plot_tsne(experiments, torch.device(args.device))


if __name__ == "__main__":
    main()
