import dataclasses
import pathlib
from typing import Any, Callable, Sequence

import torch

from .. import attack, data, evaluation
from ..models import lvm


EntropyFn = Callable[[torch.utils.data.Dataset[Any], int, int, int, int], torch.Tensor]


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

    def set_final_metrics(
        self, final_metrics: dict[str, float], prefix: str
    ) -> dict[str, float]:
        final_metrics[prefix + ".mig"] = self.mig
        final_metrics[prefix + ".uig"] = self.uig
        final_metrics[prefix + ".ltig"] = self.ltig
        return final_metrics


def evaluate_mi_metrics_with_attacks(
    name: str,
    dataset: data.DatasetWithFactors,
    model: lvm.LatentVariableModel,
    result: evaluation.Result,
    out_dir: pathlib.Path,
    alpha: Sequence[float] = (),
) -> None:
    attacks = {name: model}
    if alpha:
        D = model.spec.real_components_size
        U = (torch.eye(D) - 2 / D).to(model.device)
        for a in alpha:
            attacks[f"{name}_redundancy_{a}"] = attack.RedundancyAttack(model, a, U)

    for model_name, target in attacks.items():
        print(f"evaluating {model_name}...")
        mi_metrics = evaluate_mi_metrics(dataset, target)
        mi_metrics.save(out_dir / f"mi_metrics-{model_name}.txt")
        mi_metrics.set_final_metrics(result.final_metrics, model_name)


@torch.no_grad()
def evaluate_mi_metrics(
    dataset: data.DatasetWithFactors,
    model: lvm.LatentVariableModel,
) -> MIMetrics:
    ent_yj = dataset.factor_entropies()
    mi_zi_yj = _compute_mi_zi_yj(dataset, model.aggregated_entropy).cpu() / ent_yj
    mi_zmi_yj = _compute_mi_zi_yj(dataset, model.aggregated_loo_entropy).cpu() / ent_yj
    mig = _gap(mi_zi_yj, 0).mean().item()
    uig = (mi_zi_yj - mi_zmi_yj).amax(0).mean().item()
    ltig = _gap(mi_zmi_yj, 0, largest=False).mean().item()
    return MIMetrics(mi_zi_yj, mi_zmi_yj, mig, uig, ltig)


def _compute_mi_zi_yj(
    dataset: data.DatasetWithFactors, entropy_fn: EntropyFn
) -> torch.Tensor:
    N = len(dataset)
    N_sub = N // 50
    inner_bsize = 256
    outer_bsize = 1024

    def ent(dataset: torch.utils.data.Dataset[Any], n: int) -> torch.Tensor:
        return entropy_fn(dataset, n, min(n, N_sub), inner_bsize, outer_bsize)

    # Compute H(z_i).
    ent_zi = ent(dataset, N)

    # Compute H(z_i|y_j).
    # Note: we assume p(y_j) is a uniform distribution.
    ent_zi_yj_list: list[torch.Tensor] = []
    for j in range(dataset.n_factors):
        ent_zi_yj_points: list[torch.Tensor] = []
        for yj in range(dataset.n_factor_values[j]):
            subset = dataset.fix_factor(j, yj)
            ent_zi_yj_point = ent(subset, len(subset))
            ent_zi_yj_points.append(ent_zi_yj_point)
        ent_zi_yj_list.append(torch.stack(ent_zi_yj_points).mean(0))
    ent_zi_yj = torch.stack(ent_zi_yj_list, 1)

    return ent_zi[:, None] - ent_zi_yj


def _gap(x: torch.Tensor, dim: int, largest: bool = True) -> torch.Tensor:
    (top2, _) = torch.topk(x, 2, dim, largest)
    top2 = torch.movedim(top2, dim, 0)
    return top2[0] - top2[1] if largest else top2[1] - top2[0]
