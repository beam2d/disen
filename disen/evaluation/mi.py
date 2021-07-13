import dataclasses
import logging
import pathlib
from typing import Any, Callable

import torch

from .. import data, evaluation
from ..models import lvm


EntropyFn = Callable[[torch.utils.data.Dataset[Any], int, int, int], torch.Tensor]
_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MIMetrics:
    mi_zi_yj: torch.Tensor
    mi_zmi_yj: torch.Tensor
    mi_z_yj: torch.Tensor
    mig: float  # mutual information gap
    ub: float  # unibound
    lti: float  # latent traversal information

    def save(self, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            f.write(f"mi_zi_yj=\n{self.mi_zi_yj}\n")
            f.write(f"mi_zmi_yj=\n{self.mi_zmi_yj}\n")
            f.write(f"mi_z_yj=\n{self.mi_z_yj}\n")
            f.write(f"mig={self.mig}\n")
            f.write(f"ub={self.ub}\n")
            f.write(f"lti={self.lti}\n")

    def set_metrics(
        self, result: evaluation.Result, param_name: str, param_value: float
    ) -> None:
        result.add_parameterized_metric(param_name, param_value, "mig", self.mig)
        result.add_parameterized_metric(param_name, param_value, "ub", self.ub)
        result.add_parameterized_metric(param_name, param_value, "lti", self.lti)


def evaluate_mi_metrics(
    model: lvm.LatentVariableModel,
    dataset: data.DatasetWithFactors,
    result: evaluation.Result,
    out_dir: pathlib.Path,
    param: tuple[str, float] = ("", 0.0),
) -> None:
    _logger.info(f"evaluating MI metrics [{param[0]}={param[1]}]...")
    mi = mi_metrics(model, dataset)
    mi.save(out_dir / f"mi_metrics-{param[0]}={param[1]}.txt")
    mi.set_metrics(result, *param)


@torch.no_grad()
def mi_metrics(
    model: lvm.LatentVariableModel,
    dataset: data.DatasetWithFactors,
) -> MIMetrics:
    ent_yj = dataset.factor_entropies()
    mi_zi_yj = _compute_mi_zi_yj(dataset, model.aggregated_entropy).cpu() / ent_yj
    mi_zmi_yj = _compute_mi_zi_yj(dataset, model.aggregated_loo_entropy).cpu() / ent_yj
    mi_z_yj = _compute_mi_zi_yj(dataset, model.aggregated_joint_entropy).cpu() / ent_yj
    mig = _gap(mi_zi_yj, 0).mean().item()
    ub = (mi_zi_yj - mi_zmi_yj).amax(0).mean().item()
    lti = (mi_z_yj - mi_zmi_yj).amax(0).mean().item()
    return MIMetrics(mi_zi_yj, mi_zmi_yj, mi_z_yj, mig, ub, lti)


def _compute_mi_zi_yj(
    dataset: data.DatasetWithFactors, entropy_fn: EntropyFn
) -> torch.Tensor:
    inner_bsize = 256
    outer_bsize = 1024

    def ent(dataset: torch.utils.data.Dataset[Any], sample_size: int) -> torch.Tensor:
        sample_size = min(sample_size, data.dataset_size(dataset))
        return entropy_fn(dataset, sample_size, inner_bsize, outer_bsize)

    # Compute H(z_i).
    ent_zi = ent(dataset, 10_000)

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
