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
            "unibound_l",
            "unibound_u",
        ],
        "metric",
        "score",
    ).replace(
        {
            "betaVAE": "βVAE",
            "beta_vae_score": "BetaVAE",
            "factor_vae_score": "FactorVAE",
            "mig": "MIG",
            "unibound_l": "UniBound-L",
            "unibound_u": "UniBound-U",
        }
    )
    df_clean = df.loc[df["alpha"] == 0.0]
    df_attack = df.loc[((df["train_seed"] == 6) & (df["model"] == "TCVAE"))]

    metric_order = [
        "BetaVAE",
        "FactorVAE",
        "MIG",
        "UniBound-L",
        "UniBound-U",
    ]
    model_order = ["βVAE", "FactorVAE", "TCVAE", "JointVAE"]

    seaborn.set("paper", "darkgrid", "muted")

    task_dir = root / f"task-{task}"

    fg = seaborn.catplot(
        kind="box",
        x="metric",
        order=metric_order,
        y="score",
        hue="model",
        hue_order=model_order,
        legend_out=False,
        data=df_clean,
    )
    _render_and_close(fg, task_dir / "model_metric.png")

    for model in model_order:
        fg = seaborn.catplot(
            kind="box",
            x="metric",
            order=metric_order,
            y="score",
            hue="train_seed",
            legend=False,
            data=df_clean.loc[df_clean["model"] == model],
        )
        _render_and_close(fg, task_dir / f"eval_deviation-{model}.png")

    for model in ["TCVAE"]:
        fg = seaborn.relplot(
            kind="line",
            x="alpha",
            y="score",
            hue="metric",
            hue_order=metric_order,
            data=df_attack.loc[df_attack["model"] == model],
        )
        fg._legend.set_bbox_to_anchor([0.8, 0.55])
        for ax in fg.axes.flatten():
            ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        _render_and_close(fg, task_dir / f"attacked-{model}.png")


def _render_and_close(fg: seaborn.FacetGrid, path: pathlib.Path) -> None:
    fg.set_axis_labels("", "")
    fg.figure.savefig(path)
    pyplot.close(fg.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    plot(pathlib.Path(args.root), args.task)


if __name__ == "__main__":
    main()
