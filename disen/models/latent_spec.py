import dataclasses
from typing import Iterable, Literal


_Domain = Literal["real", "categorical"]


@dataclasses.dataclass
class SingleLatentSpec:
    name: str
    domain: _Domain
    size: int
    n_categories: int = 1

    @property
    def numel(self) -> int:
        return self.size * self.n_categories

    def _repr_props(self) -> str:
        dom = f",{self.n_categories}" if self.domain == "categorical" else ""
        return f"{self.name}({self.domain},{self.size}{dom})"

    def __repr__(self) -> str:
        return f"Latent({self._repr_props()})"


class LatentSpec:
    def __init__(self, specs: Iterable[SingleLatentSpec]) -> None:
        self.specs = tuple(specs)

        for spec in self.specs:
            if spec.domain == "real":
                assert spec.n_categories == 1

    @property
    def size(self) -> int:
        return sum(spec.size for spec in self.specs)

    @property
    def numel(self) -> int:
        return sum(spec.numel for spec in self.specs)

    def __repr__(self) -> str:
        return f"LatentSpec[{', '.join(spec._repr_props() for spec in self.specs)}]"
