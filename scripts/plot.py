import argparse
import pathlib
from typing import Sequence

from matplotlib import pyplot
import pandas
import seaborn

import disen


def load_df(root: pathlib.Path, task: disen.TaskType) -> pandas.DataFrame:
    entries = [
        exp.load_entry_with_attrs()
        for exp in disen.collect_experiments(root, task)
        if exp.has_entry()
    ]
    entries = [e for e in entries if e]
    return pandas.DataFrame(entries)


def select_best_half(
    df_raw: pandas.DataFrame,
    metric: str,
    columns: Sequence[str],
) -> pandas.DataFrame:
    df_raw = df_raw[df_raw["alpha"].isna()]
    eval_mean = df_raw.groupby(["model", "train_seed"]).mean()
    counts = (eval_mean.count(level="model")[metric] - 1) // 2 + 1
    count = counts[0]
    if not (counts == count).all():
        raise ValueError("evaluation counts differ between models (not supported)")
    sort = eval_mean.sort_values(metric, ascending=False)
    index = sort.groupby(level="model").head(count).index
    df = df_raw[df_raw.set_index(index.names).index.isin(index)]
    return df.melt(
        ["task", "model", "train_seed", "eval_seed"],
        columns,
        "metric",
        "score",
    )


def melt_for_train(df_raw: pandas.DataFrame) -> pandas.DataFrame:
    return df_raw.melt(
        ["task", "model", "train_seed"],
        ["elbo", "loss", "recon"],
        "objective",
        "value",
    ).replace({"betaVAE": "βVAE"})


def melt_for_eval(df_raw: pandas.DataFrame, choose_best_half: bool) -> pandas.DataFrame:
    pid_cols = [
        "unibound_l",
        "unibound_u",
        "redundancy_l",
        "redundancy_u",
        "synergy_l",
        "synergy_u",
        "mi",
    ]

    if choose_best_half:
        df_bv = select_best_half(df_raw, "beta_vae_score", ["beta_vae_score"])
        df_fv = select_best_half(df_raw, "factor_vae_score", ["factor_vae_score"])
        df_mig = select_best_half(df_raw, "mig", ["mig"])
        df_ub = select_best_half(df_raw, "unibound_l", pid_cols)
        df = pandas.concat([df_bv, df_fv, df_mig, df_ub])
    else:
        df = df_raw.melt(
            ["task", "model", "train_seed", "eval_seed", "alpha"],
            ["beta_vae_score", "factor_vae_score", "mig", *pid_cols],
            "metric",
            "score",
        )
    return df.replace(
        {
            "betaVAE": "βVAE",
            "beta_vae_score": "BetaVAE",
            "factor_vae_score": "FactorVAE",
            "mi": "MI",
            "mig": "MIG",
            "unibound_l": "UniBound",
            "unibound_u": "UniBound_u",
            "redundancy_l": "redundancy",
            "redundancy_u": "redundancy_u",
            "synergy_l": "synergy",
            "synergy_u": "synergy_u",
        }
    )


def plot(root: pathlib.Path, task: disen.TaskType) -> None:
    df_raw = load_df(root, task)
    df_train = melt_for_train(df_raw)
    df = melt_for_eval(df_raw, choose_best_half=False)
    df_clean = melt_for_eval(df_raw, choose_best_half=True)
    df_attack = df[df["alpha"].notna()]

    metric_order = ["BetaVAE", "FactorVAE", "MIG", "UniBound"]
    model_order = ["βVAE", "FactorVAE", "TCVAE", "JointVAE"]
    _set_style(font_scale=2)

    task_dir = root / f"task-{task}"
    legend = task == "dSprites"

    for objective in ["elbo", "loss", "recon"]:
        fg = seaborn.catplot(
            kind="box",
            x="objective",
            y="value",
            col="model",
            col_order=model_order,
            col_wrap=2,
            hue="train_seed",
            legend=False,
            sharey=False,
            data=df_train.loc[df_train["objective"] == objective],
        )
        fg.set_axis_labels("", objective)
        _render_and_close(fg, task_dir / f"train_{objective}.png")

    _set_style(font_scale=2.5)
    fg = seaborn.catplot(
        kind="box",
        x="task",
        hue="model",
        hue_order=model_order,
        y="score",
        col="metric",
        col_order=metric_order,
        col_wrap=2,
        legend_out=False,
        data=df_clean,
    )
    fg.set_axis_labels("", "metric score")
    fg.set_xticklabels(labels=[])
    fg.set(ylim=(0, 1))
    if legend:
        fg.add_legend(loc="best", title="model")
    _render_and_close(fg, task_dir / "model_metric.png")

    _set_style(font_scale=2)
    for model in model_order:
        name = "betaVAE" if model == "βVAE" else model
        fg = seaborn.catplot(
            kind="box",
            x="train_seed",
            y="score",
            col="metric",
            col_order=metric_order,
            col_wrap=2,
            legend=False,
            data=df[(df["model"] == model) & df["alpha"].isna()],
        )
        fg.set_axis_labels("", "disentanglement score")
        fg.set_xticklabels(labels=[])
        fg.set(ylim=(0, 1))
        _render_and_close(fg, task_dir / f"eval_deviation-{name}.png")

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
        df_model = df_clean.loc[df_clean["model"] == model]
        seaborn.barplot(
            x="metric",
            order=pid_keys,
            y="score",
            color="#EAEAF2",
            data=df_model,
            ax=ax,
        )
    fg.set_xticklabels(["U", "R", "C"])
    fg.set_titles("{col_name}")
    fg.set_axis_labels("", "normalized information")
    fg.set(ylim=(0, 0.8))
    _render_and_close(fg, task_dir / "range.png")

    _set_style(font_scale=2.5)
    score_range = 0.0
    score_med: list[float] = []
    for metric in metric_order:
        df_metric = df_attack[df_attack["metric"] == metric]
        score_min = df_metric["score"].min()
        score_max = df_metric["score"].max()
        score_range = max(score_range, score_max - score_min)
        score_med.append((score_min + score_max) / 2)
    score_range += 0.1

    attacked_models = ["βVAE", "TCVAE"]
    fg = seaborn.relplot(
        kind="line",
        x="alpha",
        y="score",
        hue="model",
        hue_order=attacked_models,
        col="metric",
        col_order=metric_order,
        col_wrap=2,
        facet_kws={"sharex": True, "sharey": False},
        legend=False,
        data=df_attack,
    )
    fg.set_titles("{col_name}")
    fg.set_axis_labels("α", "")
    for ax, med, metric in zip(fg.axes.flatten(), score_med, metric_order):
        ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0])
        ymin = med - score_range / 2
        ymax = med + score_range / 2
        if ymin < 0:
            ymin = 0.0
            ymax = score_range
        elif ymax > 1:
            ymax = 1.0
            ymin = 1.0 - score_range
        ax.set_ylim(ymin, ymax)
        if legend and metric == "BetaVAE":
            ax.legend(attacked_models, loc="best")
    _render_and_close(fg, task_dir / f"attacked.png")


def _set_style(
    context: str = "paper",
    style: str = "darkgrid",
    palette: str = "muted",
    font_scale: float = 2,
) -> None:
    seaborn.set_theme(context, style, palette, font_scale=font_scale)


def _render_and_close(fg: seaborn.FacetGrid, path: pathlib.Path) -> None:
    fg.figure.savefig(path, bbox_inches="tight")
    pyplot.close(fg.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    plot(pathlib.Path(args.root), args.task)


if __name__ == "__main__":
    main()
