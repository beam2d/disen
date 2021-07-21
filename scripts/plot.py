import argparse
import pathlib

from matplotlib import pyplot
import pandas
import seaborn

import disen


def plot(root: pathlib.Path, task: disen.TaskType) -> None:
    entries = [
        exp.load_entry_with_attrs() for exp in disen.collect_experiments(root, task)
    ]
    df = pandas.DataFrame(entries)
    df = df.melt(
        ["task", "model", "train_seed", "eval_seed", "alpha"],
        ["factor_vae_score", "beta_vae_score", "mig", "ub", "lti"],
        "metric",
        "score",
    ).replace({
        "betaVAE": "βVAE",
        "beta_vae_score": "BetaVAE",
        "factor_vae_score": "FactorVAE",
        "mig": "MIG",
        "ub": "UB(lower)",
        "lti": "UB(upper)",
    })
    model_order = ["βVAE", "FactorVAE", "TCVAE", "JointVAE", "CascadeVAE"]

    seaborn.set_style("darkgrid")

    task_dir = root / f"task-{task}"

    fg = seaborn.catplot(
        kind="box", x="model", y="score", hue="metric", order=model_order, data=df
    )
    _render_and_close(fg, task_dir / "model_metric.png")

    fg = seaborn.catplot(
        kind="box",
        x="train_seed",
        y="score",
        hue="metric",
        col="model",
        col_order=["βVAE", "CascadeVAE"],
        data=df,
    )
    _render_and_close(fg, task_dir / "eval_deviation.png")

    fg = seaborn.catplot(
        kind="box",
        x="train_seed",
        y="score",
        hue="metric",
        col="model",
        col_order=model_order,
        col_wrap=3,
        data=df,
    )
    _render_and_close(fg, task_dir / "eval_deviation_all.png")


def _render_and_close(fg: seaborn.FacetGrid, path: pathlib.Path) -> None:
    fg.fig.savefig(path)
    pyplot.close(fg.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    plot(pathlib.Path(args.root), args.task)


if __name__ == "__main__":
    main()
