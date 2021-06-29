import dataclasses
from typing import Iterable, Iterator, Literal, Sequence

import torch


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


    def generate_traversal(self, z: torch.Tensor, index: int) -> torch.Tensor:
        assert 0 <= index < self.size
        if self.domain == "real":
            # sample ten points from [-2, 2]
            N = 10
            z_new = torch.stack([z] * N, -1)
            z_new[:, index] = torch.linspace(-2.0, 2.0, 10, device=z.device)
            return z_new.movedim(-1, 1)
        elif self.domain == "categorical":
            # sample all categories
            N = self.n_categories
            assert z.shape[-1] == N
            z_new = torch.stack([z] * N, -1)
            z_new[:, index] = torch.eye(N, device=z.device)
            return z_new.movedim(-1, 1)
        raise ValueError("invalid domain")


class LatentSpec:
    def __init__(self, specs: Iterable[SingleLatentSpec]) -> None:
        self.specs = tuple(specs)

        for spec in self.specs:
            if spec.domain == "real":
                assert spec.n_categories == 1

    def __iter__(self) -> Iterator[SingleLatentSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> SingleLatentSpec:
        return self.specs[index]

    @property
    def size(self) -> int:
        return sum(spec.size for spec in self.specs)

    @property
    def real_components_size(self) -> int:
        return sum(spec.size for spec in self.specs if spec.domain == "real")

    @property
    def numel(self) -> int:
        return sum(spec.numel for spec in self.specs)

    def __repr__(self) -> str:
        return f"LatentSpec[{', '.join(spec._repr_props() for spec in self.specs)}]"

    def generate_traversal(
        self,
        zs: Sequence[torch.Tensor],
        index: int,
        sub_index: int,
    ) -> list[torch.Tensor]:
        assert len(zs) == len(self)
        zi_new = self[index].generate_traversal(zs[index], sub_index)
        n_copies = zi_new.shape[1]

        ret: list[torch.Tensor] = []
        for i, z in enumerate(zs):
            if i == index:
                z = zi_new
            else:
                z = z[:, None].expand(z.shape[0], n_copies, *z.shape[1:])
            ret.append(z.reshape(-1, *z.shape[2:]))

        return ret
