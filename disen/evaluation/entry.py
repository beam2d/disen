import json
import pathlib


class Entry:
    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    @property
    def scores(self) -> dict[str, float]:
        return self._scores.copy()

    def add_score(self, name: str, value: float) -> None:
        assert name not in self._scores
        self._scores[name] = value

    def load(self, path: pathlib.Path) -> None:
        with open(path) as f:
            self._scores = json.load(f)

    def save(self, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(self._scores, f)
