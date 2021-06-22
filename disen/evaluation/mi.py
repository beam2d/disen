import dataclasses
import pathlib
from typing import Callable

import torch

from .. import data
from ..models import lvm


@dataclasses.dataclass
class MIMetrics:
    mi_zi_yj: torch.Tensor
    mi_zmi_yj: torch.Tensor
    mig: float  # mutual information gap
    uig: float  # unique information gap
    ltig: float  # latent traversal information gap

    def save(self, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            f.write(f"mi_zi_yj=\n{self.mi_zi_yj}\n")
            f.write(f"mi_zmi_yj=\n{self.mi_zmi_yj}\n")
            f.write(f"mig={self.mig}\n")
            f.write(f"uig={self.uig}\n")
            f.write(f"ltig={self.ltig}\n")

    def set_final_metrics(self, final_metrics: dict[str, float]) -> dict[str, float]:
        final_metrics["mig"] = self.mig
        final_metrics["uig"] = self.uig
        final_metrics["ltig"] = self.ltig
        return final_metrics


@torch.no_grad()
def evaluate_mi_metrics(
    dataset: data.DatasetWithFactors,
    model: lvm.LatentVariableModel,
) -> MIMetrics:
    mi_zi_yj = _compute_mi_zi_yj(dataset, model, lambda log_q: log_q).cpu()
    mi_zmi_yj = _compute_mi_zi_yj(
        dataset, model, lambda log_q: log_q.sum(-1, keepdim=True) - log_q
    ).cpu()
    ent_yj = dataset.factor_entropies()
    mig = (_gap(mi_zi_yj, 0) / ent_yj).mean().item()
    uig = (((mi_zi_yj - mi_zmi_yj) / ent_yj).amax(0).mean().item() + 1.) / 2.
    ltig = (_gap(mi_zmi_yj, 0, largest=False) / ent_yj).mean().item()
    return MIMetrics(mi_zi_yj, mi_zmi_yj, mig, uig, ltig)


def _compute_mi_zi_yj(
    dataset: data.DatasetWithFactors,
    model: lvm.LatentVariableModel,
    selector: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    N = len(dataset)
    batch_size = 1000

    # Compute H(z_i).
    ent_zi = model.aggregated_entropy(dataset, N, N // 50, batch_size, selector)

    # Compute H(z_i|y_j).
    # Note: we assume p(y_j) is a uniform distribution.
    ent_zi_yj_list: list[torch.Tensor] = []
    for j in range(dataset.n_factors):
        ent_zi_yj_points: list[torch.Tensor] = []
        for yj in range(dataset.n_factor_values[j]):
            subset = dataset.fix_factor(j, yj)
            N_sub = len(subset)
            ent_zi_yj_point = model.aggregated_entropy(
                subset, N_sub, N_sub // 20, batch_size, selector
            )
            ent_zi_yj_points.append(ent_zi_yj_point)
        ent_zi_yj_list.append(torch.stack(ent_zi_yj_points).mean(0))
    ent_zi_yj = torch.stack(ent_zi_yj_list, 1)

    return ent_zi[:, None] - ent_zi_yj


def _gap(x: torch.Tensor, dim: int, largest: bool = True) -> torch.Tensor:
    (top2, _) = torch.topk(x, 2, dim, largest)
    top2 = torch.movedim(top2, dim, 0)
    return top2[0] - top2[1] if largest else top2[1] - top2[0]
