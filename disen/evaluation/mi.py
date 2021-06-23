import dataclasses
import pathlib
from typing import Callable, Optional

import torch

from .. import attack, data, evaluation
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
    noise: Optional[float] = None,
    mix_rate: Optional[float] = None,
) -> None:
    attacks = {name: model}
    if noise is not None:
        attacks[name + "_noised"] = attack.NoisedCopyAttack(model, noise)
    if mix_rate is not None:
        attacks[name + "_mixed"] = attack.GlobalMixingAttack(model, mix_rate)

    for model_name, target in attacks.items():
        mi_metrics = evaluate_mi_metrics(dataset, target)
        mi_metrics.save(out_dir / f"mi_metrics-{model_name}.txt")
        mi_metrics.set_final_metrics(result.final_metrics, model_name)


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
    uig = ((mi_zi_yj - mi_zmi_yj) / ent_yj).amax(0).mean().item()
    ltig = (_gap(mi_zmi_yj, 0, largest=False) / ent_yj).mean().item()
    return MIMetrics(mi_zi_yj, mi_zmi_yj, mig, uig, ltig)


def _compute_mi_zi_yj(
    dataset: data.DatasetWithFactors,
    model: lvm.LatentVariableModel,
    selector: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    N = len(dataset)
    N_sub = N // 50
    batch_size = 1024

    # Compute H(z_i).
    ent_zi = model.aggregated_entropy(dataset, N, N_sub, batch_size, selector)

    # Compute H(z_i|y_j).
    # Note: we assume p(y_j) is a uniform distribution.
    ent_zi_yj_list: list[torch.Tensor] = []
    for j in range(dataset.n_factors):
        ent_zi_yj_points: list[torch.Tensor] = []
        for yj in range(dataset.n_factor_values[j]):
            subset = dataset.fix_factor(j, yj)
            n = len(subset)
            ent_zi_yj_point = model.aggregated_entropy(
                subset, n, min(n, N_sub), batch_size, selector
            )
            ent_zi_yj_points.append(ent_zi_yj_point)
        ent_zi_yj_list.append(torch.stack(ent_zi_yj_points).mean(0))
    ent_zi_yj = torch.stack(ent_zi_yj_list, 1)

    return ent_zi[:, None] - ent_zi_yj


def _gap(x: torch.Tensor, dim: int, largest: bool = True) -> torch.Tensor:
    (top2, _) = torch.topk(x, 2, dim, largest)
    top2 = torch.movedim(top2, dim, 0)
    return top2[0] - top2[1] if largest else top2[1] - top2[0]
