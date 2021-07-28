import argparse
import pathlib

from matplotlib import pyplot
import pandas
import seaborn

import disen


def plot(root: pathlib.Path, task: disen.TaskType) -> None:
    entries = [
        exp.load_entry_with_attrs()
        for exp in disen.collect_experiments(root, task)
        if exp.has_entry()
    ]
    entries = [e for e in entries if e]
    df = pandas.DataFrame(entries)
    df = df.melt(
        ["task", "model", "train_seed", "eval_seed", "alpha"],
        [
            "beta_vae_score",
            "factor_vae_score",
            "mig",
            "ulbo_exact",
            "uubo_exact",
            "ulbo",
            "uubo",
        ],
        "metric",
        "score",
    ).replace(
        {
            "betaVAE": "βVAE",
            "beta_vae_score": "BetaVAE",
            "factor_vae_score": "FactorVAE",
            "mig": "MIG",
            "ulbo_exact": "UniBound-L",
            "uubo_exact": "UniBound-U",
            "ulbo": "UniBound-L(DRE)",
            "uubo": "UniBound-U(DRE)",
        }
    )
    df_clean = df.loc[df["alpha"] == 0.0]
    df_attack = df.loc[
        ((df["train_seed"] == 1) & (df["model"] == "CascadeVAE"))
        | ((df["train_seed"] == 6) & (df["model"] == "TCVAE"))
    ]

    metric_order = [
        "BetaVAE",
        "FactorVAE",
        "MIG",
        "UniBound-L",
        "UniBound-L(DRE)",
        "UniBound-U",
        "UniBound-U(DRE)",
    ]
    model_order = ["βVAE", "FactorVAE", "TCVAE", "JointVAE", "CascadeVAE"]

    seaborn.set_style("darkgrid")

    task_dir = root / f"task-{task}"

    fg = seaborn.catplot(
        kind="box",
        x="metric",
        order=metric_order,
        y="score",
        hue="model",
        hue_order=model_order,
        aspect=2.0,
        data=df_clean,
    )
    _render_and_close(fg, task_dir / "model_metric.png")

    fg = seaborn.catplot(
        kind="box",
        x="metric",
        order=metric_order,
        y="score",
        hue="train_seed",
        legend=False,
        col="model",
        col_order=["βVAE", "CascadeVAE"],
        aspect=2.0,
        data=df_clean,
    )
    _render_and_close(fg, task_dir / "eval_deviation.png")

    fg = seaborn.catplot(
        kind="box",
        x="metric",
        order=metric_order,
        y="score",
        hue="train_seed",
        legend=False,
        col="model",
        col_order=model_order,
        col_wrap=3,
        aspect=2.0,
        data=df_clean,
    )
    _render_and_close(fg, task_dir / "eval_deviation_all.png")

    fg = seaborn.relplot(
        kind="line",
        x="alpha",
        y="score",
        hue="metric",
        hue_order=metric_order,
        col="model",
        col_order=["TCVAE", "CascadeVAE"],
        data=df_attack,
    )
    _render_and_close(fg, task_dir / "attacked.png")


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
