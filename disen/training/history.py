import json
import logging
from matplotlib import pyplot
import pandas
import pathlib
import seaborn


_logger = logging.getLogger(__name__)


class History:
    def __init__(self) -> None:
        self._history: list[dict[str, float]] = []

    def add_epoch(self, entry: dict[str, float]) -> None:
        _logger.info(" ".join([f"{k}={v}" for k, v in entry.items()]))
        self._history.append(entry)

    def plot(self, out_dir: pathlib.Path) -> None:
        for key in self._history[0]:
            if key in ("epoch", "iteration"):
                continue
            data = {
                "iteration": [d["iteration"] for d in self._history],
                key: [d[key] for d in self._history],
            }
            df = pandas.DataFrame(data)
            seaborn.set_style("darkgrid")
            fg = seaborn.relplot(x="iteration", y=key, kind="line", data=df)
            fg.savefig(out_dir / f"{key}.png")
            pyplot.close(fg.fig)

    def save(self, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(self._history, f, indent=4)
