from __future__ import annotations
import dataclasses
import json
import pathlib
from typing import Sequence, Union

from matplotlib import pyplot
import numpy
import pandas
import seaborn


@dataclasses.dataclass
class Result:
    history: list[dict[str, float]]
    final_metrics: dict[str, float]

    def save(self, path: Union[str, pathlib.Path]) -> None:
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=4)

    @staticmethod
    def load(path: Union[str, pathlib.Path]) -> Result:
        with open(path, "r") as f:
            j = json.load(f)
        return Result(**j)

    def plot_history(self, out_dir: Union[str, pathlib.Path]) -> None:
        _plot_history(self.history, pathlib.Path(out_dir))


def summarize_multiple_trials(
    results: Sequence[Result], out_dir: Union[str, pathlib.Path]
) -> None:
    out_dir = pathlib.Path(out_dir)
    _plot_history(sum([r.history for r in results], []), out_dir)

    metrics_summary: dict[str, float] = {}
    for key in results[0].final_metrics:
        values = numpy.asarray([r.final_metrics[key] for r in results])
        metrics_summary[key + ".mean"] = values.mean()
        metrics_summary[key + ".std"] = values.std()
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
