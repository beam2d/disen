from __future__ import annotations
import dataclasses
import json
import pathlib
from typing import Sequence, Union

from matplotlib import pyplot
import numpy
import pandas
import seaborn


@dataclasses.dataclass(frozen=True)
class NamedValue:
    name: str
    value: float


@dataclasses.dataclass
class Result:
    history: list[dict[str, float]]
    metrics: list[tuple[NamedValue, NamedValue]]

    @staticmethod
    def new() -> Result:
        return Result([], [])

    def save(self, path: Union[str, pathlib.Path]) -> None:
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=4)

    @staticmethod
    def load(path: Union[str, pathlib.Path]) -> Result:
        with open(path, "r") as f:
            j = json.load(f)
        ret = Result.new()
        ret.history = j["history"]
        for m, p in j["metrics"]:
            ret.add_parameterized_metric(m["name"], m["value"], p["name"], p["value"])
        return ret

    def add_metric(self, name: str, value: float) -> None:
        self.add_parameterized_metric("", 0.0, name, value)

    def add_parameterized_metric(
        self,
        param_name: str,
        param_value: float,
        metric_name: str,
        metric_value: float,
    ) -> None:
        self.metrics.append((
            NamedValue(param_name, param_value), NamedValue(metric_name, metric_value)
        ))

    def plot_history(self, out_dir: Union[str, pathlib.Path]) -> None:
        _plot_history(self.history, pathlib.Path(out_dir))


def summarize_multiple_trials(
    results: Sequence[Result], out_dir: Union[str, pathlib.Path]
) -> None:
    out_dir = pathlib.Path(out_dir)
    _plot_history(sum([r.history for r in results], []), out_dir)

    metrics: dict[tuple[str, NamedValue], list[float]] = {}
    for result in results:
        for param, metric in result.metrics:
            metrics.setdefault((metric.name, param), []).append(metric.value)

    _plot_metrics(metrics, out_dir)

    metrics_summary: dict[str, float] = {}
    for metric_name, param in sorted(metrics):
        values = numpy.asarray(metrics[metric_name, param])
        name = f"{metric_name}-{param.name}={param.value}"
        metrics_summary[f"{name}.mean"] = float(values.mean())
        metrics_summary[f"{name}.std"] = float(values.std())

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)


def _plot_history(history: list[dict[str, float]], out_dir: pathlib.Path) -> None:
    for key in history[0]:
        if key in ("epoch", "iteration"):
            continue
        data: dict[str, list[float]] = {"iteration": [], key: []}
        for d in history:
            data["iteration"].append(d["iteration"])
            data[key].append(d[key])

        seaborn.set_style("darkgrid")
        df = pandas.DataFrame(data)
        fg = seaborn.relplot(x="iteration", y=key, kind="line", data=df)
        fg.savefig(out_dir / f"{key}.png")
        pyplot.close(fg.fig)


def _plot_metrics(
    metrics: dict[tuple[str, NamedValue], list[float]],
    out_dir: pathlib.Path,
) -> None:
    metric_param_set = {(metric_name, param.name) for metric_name, param in metrics}
    for metric_name, param_name in metric_param_set:
        data: dict[str, list[float]] = {metric_name: [], param_name: []}
        for (m, p), values in metrics.items():
            if m != metric_name or p.name != param_name:
                continue
            data[metric_name] += values
            data[param_name] += [p.value] * len(values)

        seaborn.set_style("darkgrid")
        df = pandas.DataFrame(data)
        fg = seaborn.relplot(x=param_name, y=metric_name, kind="line", data=df)
        fg.savefig(out_dir / f"{metric_name}_vs_{param_name}.png")
        pyplot.close(fg.fig)
        