import dataclasses
import itertools
import logging
import pathlib
from typing import Any, Callable, Sequence

import torch

from .. import attack, data, evaluation
from ..models import lvm


EntropyFn = Callable[[torch.utils.data.Dataset[Any], int, int, int], torch.Tensor]
_logger = logging.getLogger(__name__)


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

    def set_metrics(
        self, result: evaluation.Result, param_name: str, param_value: float
    ) -> None:
        result.add_parameterized_metric(param_name, param_value, "mig", self.mig)
        result.add_parameterized_metric(param_name, param_value, "uig", self.uig)
        result.add_parameterized_metric(param_name, param_value, "ltig", self.ltig)


def evaluate_mi_metrics_with_attacks(
    name: str,
    dataset: data.DatasetWithFactors,
    model: lvm.LatentVariableModel,
    result: evaluation.Result,
    out_dir: pathlib.Path,
    alpha: Sequence[float] = (),
) -> None:
    D = model.spec.real_components_size
    U = (torch.eye(D) - 2 / D).to(model.device)

    for a in itertools.chain([0.0], alpha):
        _logger.info(f"evaluating {name} [alpha={a}]...")
        target = model if a == 0.0 else attack.RedundancyAttack(model, a, U)
        mi_metrics = evaluate_mi_metrics(dataset, target)
        mi_metrics.save(out_dir / f"mi_metrics-{name}-alpha={a}.txt")
        mi_metrics.set_metrics(result, "alpha", a)


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
    inner_bsize = 256
    outer_bsize = 1024

    def ent(dataset: torch.utils.data.Dataset[Any], sample_size: int) -> torch.Tensor:
        sample_size = min(sample_size, data.dataset_size(dataset))
        return entropy_fn(dataset, sample_size, inner_bsize, outer_bsize)

    # Compute H(z_i).
    # TODO(beam2d): Do stratified sampling over each y_j.
    ent_zi = ent(dataset, 20_000)

    # Compute H(z_i|y_j).
    # Note: we assume p(y_j) is a uniform distribution.
    ent_zi_yj_list: list[torch.Tensor] = []
    for j in range(dataset.n_factors):
        ent_zi_yj_points: list[torch.Tensor] = []
        for yj in range(dataset.n_factor_values[j]):
            subset = dataset.fix_factor(j, yj)
            ent_zi_yj_point = ent(subset, 10_000)
            ent_zi_yj_points.append(ent_zi_yj_point)
        ent_zi_yj_list.append(torch.stack(ent_zi_yj_points).mean(0))
    ent_zi_yj = torch.stack(ent_zi_yj_list, 1)

    return ent_zi[:, None] - ent_zi_yj


def _gap(x: torch.Tensor, dim: int, largest: bool = True) -> torch.Tensor:
    (top2, _) = torch.topk(x, 2, dim, largest)
    top2 = torch.movedim(top2, dim, 0)
    return top2[0] - top2[1] if largest else top2[1] - top2[0]
