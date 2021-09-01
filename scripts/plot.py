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
            "redundancy_l",
            "redundancy_u",
            "synergy_l",
            "synergy_u",
        ],
        "metric",
        "score",
    ).replace(
        {
            "betaVAE": "βVAE",
            "beta_vae_score": "BetaVAE",
            "factor_vae_score": "FactorVAE",
            "mig": "MIG",
            "unibound_l": "UniBound",
            "unibound_u": "UniBound_u",
            "redundancy_l": "redundancy",
            "redundancy_u": "redundancy_u",
            "synergy_l": "synergy",
            "synergy_u": "synergy_u",
        }
    )
    # Remove unsuccessful trials of JointVAE
    df = df.loc[(df["model"] != "JointVAE") | df["train_seed"].isin([3, 5, 6])]

    df_clean = df.loc[df["alpha"] == 0.0]
    df_attack = df.loc[
        ((df["train_seed"] == 6) & (df["model"] == "TCVAE"))
        | ((df["train_seed"] == 5) & (df["model"] == "βVAE"))
    ]

    metric_order = [
        "BetaVAE",
        "FactorVAE",
        "MIG",
        "UniBound",
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
    fg.set_axis_labels("metric", "disentanglenet score")
    fg.set(ylim=(0, 1))
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
        fg.set_axis_labels("", "disentanglement score")
        fg.set(ylim=(0, 1))
        _render_and_close(fg, task_dir / f"eval_deviation-{model}.png")

    pid_keys = ["UniBound", "redundancy", "synergy"]
    ub_keys = [f"{k}_u" for k in pid_keys]
    fg = seaborn.catplot(
        kind="bar",
        x="metric",
        order=ub_keys,
        y="score",
        col="model",
        col_order=model_order,
        color="orange",
        legend=False,
        aspect=0.3,
        facet_kws={"gridspec_kws": {"wspace": 0.1}},
        data=df_clean,
    )
    for model, ax in fg.axes_dict.items():
        seaborn.barplot(
            x="metric",
            order=pid_keys,
            y="score",
            color="#EAEAF2",
            data=df_clean.loc[df_clean["model"] == model],
            ax=ax,
        )
    fg.set_xticklabels(["U", "R", "C"])
    fg.set_titles("{col_name}")
    fg.set_axis_labels("", "normalized value")
    fg.set(ylim=(0, 0.7))
    _render_and_close(fg, task_dir / "range.png")

    fg = seaborn.relplot(
        kind="line",
        x="alpha",
        y="score",
        hue="metric",
        hue_order=metric_order,
        col="model",
        col_order=["βVAE", "TCVAE"],
        aspect=0.5,
        data=df_attack,
    )
    fg.set_titles("{col_name}")
    fg.set_axis_labels("α", "disentanglement score")
    for ax in fg.axes.flatten():
        ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    _render_and_close(fg, task_dir / f"attacked.png")


def _render_and_close(fg: seaborn.FacetGrid, path: pathlib.Path) -> None:
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
