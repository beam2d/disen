import argparse
import math
import pathlib
import re
from typing import Any, Optional, Sequence

from matplotlib import pyplot
import pandas
import seaborn
import torch

import disen


_FACTOR_NAMES = {
    "dSprites": ("shape", "scale", "orientation", "position x", "position y"),
    "3dshapes": (
        "floor hue",
        "wall hue",
        "object hue",
        "scale",
        "shape",
        "orientation",
    ),
}


def load_df(root: pathlib.Path, task: disen.TaskType) -> pandas.DataFrame:
    entries: list[dict[str, Any]] = []
    for exp in disen.collect_experiments(root, task):
        if not exp.has_entry():
            continue
        entry = exp.load_entry_with_attrs()
        mi = exp.load_mi_metrics()
        mi_zi_yj = mi["mi_zi_yj"]
        mi_zmi_yj = mi["mi_zmi_yj"]
        mi_z_yj = mi["mi_z_yj"]

        ii = mi_zi_yj + mi_zmi_yj - mi_z_yj
        uni_l = (mi_zi_yj - mi_zmi_yj).relu().amax(0)
        uni_u = (mi_zi_yj - ii.relu()).amax(0)
        red_l = ii.relu().amax(0)
        red_u = torch.minimum(mi_zi_yj, mi_zmi_yj).amax(0)
        syn_l = (-ii).relu().amax(0)
        syn_u = (torch.minimum(mi_zi_yj, mi_zmi_yj) - ii).amax(0)

        for k in range(ii.shape[1]):
            entry[f"factor_unibound_l_{k}"] = float(uni_l[k])
            entry[f"factor_unibound_u_{k}"] = float(uni_u[k])
            entry[f"factor_redundancy_l_{k}"] = float(red_l[k])
            entry[f"factor_redundancy_u_{k}"] = float(red_u[k])
            entry[f"factor_synergy_l_{k}"] = float(syn_l[k])
            entry[f"factor_synergy_u_{k}"] = float(syn_u[k])

        entries.append(entry)

    return pandas.DataFrame(entries)


def select_best(
    df_raw: pandas.DataFrame,
    metric: str,
    columns: Sequence[str],
    ratio: float,
) -> pandas.DataFrame:
    df_raw = df_raw[df_raw["alpha"].isna()]
    eval_mean = df_raw.groupby(["model", "train_seed"]).mean()
    counts = (eval_mean.count(level="model")[metric] * ratio).apply(math.ceil)
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


def melt_for_eval(
    task: str,
    df_raw: pandas.DataFrame,
    choose_best: float = 1.0,
) -> pandas.DataFrame:
    pid_cols = ["mi"]
    pat_factor = re.compile(r"factor_(unibound|redundancy|synergy)_(l|u)_([0-9]+)")
    pat_pid = re.compile(r"(unibound|redundancy|synergy)(_u)?")
    for key in df_raw.keys():
        if pat_factor.match(key) or pat_pid.match(key):
            pid_cols.append(key)

    if choose_best < 1.0:
        df_bv = select_best(df_raw, "beta_vae_score", ["beta_vae_score"], choose_best)
        df_fv = select_best(df_raw, "factor_vae_score", ["factor_vae_score"], choose_best)
        df_mig = select_best(df_raw, "mig", ["mig"], choose_best)
        df_ub = select_best(df_raw, "unibound_l", pid_cols, choose_best)
        df = pandas.concat([df_bv, df_fv, df_mig, df_ub])
    else:
        df = df_raw.melt(
            ["task", "model", "train_seed", "eval_seed", "alpha"],
            ["beta_vae_score", "factor_vae_score", "mig", *pid_cols],
            "metric",
            "score",
        )

    def metric_to_factor(s: str) -> Optional[str]:
        match = pat_factor.match(s)
        if match is None:
            return None
        factor_index = int(match[3])
        return _FACTOR_NAMES[task][factor_index]

    def rename_metric(s: str) -> str:
        match = pat_factor.match(s)
        if match is None:
            return s
        return f"{match[1]}_{match[2]}"

    factor_name = df["metric"].apply(metric_to_factor)
    df["factor"] = factor_name
    df["metric"] = df["metric"].apply(rename_metric)

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
    df = melt_for_eval(task, df_raw)
    df = df[df["factor"].isna()]
    df_clean = melt_for_eval(task, df_raw, choose_best=0.5)
    df_factor = df_clean[df_clean["factor"].notna()]
    df_clean = df_clean[df_clean["factor"].isna()]
    df_attack = df[df["alpha"].notna()]

    metric_order = ["BetaVAE", "FactorVAE", "MIG", "UniBound"]
    model_order = ["βVAE", "FactorVAE", "TCVAE", "JointVAE"]
    model_to_name = {
        "βVAE": "betaVAE",
        "FactorVAE": "FactorVAE",
        "TCVAE": "TCVAE",
        "JointVAE": "JointVAE",
    }
    _set_style(font_scale=2)

    task_dir = root / f"task-{task}"

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
    fg.add_legend(loc="best", title="model")
    _render_and_close(fg, task_dir / "model_metric.png")

    _set_style(font_scale=2)
    for model in model_order:
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
        _render_and_close(fg, task_dir / f"eval_deviation-{model_to_name[model]}.png")

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
        facet_kws={"gridspec_kws": {"wspace": 0.2}},
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

    _set_style(font_scale=1.5)
    for model in model_order:
        df_factor_model = df_factor[df_factor["model"] == model]
        fg = seaborn.catplot(
            kind="bar",
            x="metric",
            order=ub_keys,
            y="score",
            col="factor",
            col_order=_FACTOR_NAMES[task],
            color="orange",
            legend=False,
            aspect=0.3,
            facet_kws={"gridspec_kws": {"wspace": 0.2}},
            data=df_factor_model,
        )
        for factor, ax in fg.axes_dict.items():
            seaborn.barplot(
                x="metric",
                order=pid_keys,
                y="score",
                color="#EAEAF2",
                data=df_factor_model[df_factor_model["factor"] == factor],
                ax=ax,
            )

        fg.set_xticklabels(["U", "R", "C"])
        fg.set_titles("{col_name}")
        fg.set_axis_labels("", "normalized information")
        fg.set(ylim=(0, 1))
        _render_and_close(fg, task_dir / f"factor_range_{model_to_name[model]}.png")

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
