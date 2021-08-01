import dataclasses
import logging
import pathlib
from typing import Any, Callable

import torch

from .. import data, log_util, models


EntropyFn = Callable[
    [torch.utils.data.Dataset[Any], torch.utils.data.Dataset[Any], int, int],
    torch.Tensor,
]
_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MIMetrics:
    mi_zi_yj: torch.Tensor
    mi_zmi_yj: torch.Tensor
    mi_z_yj: torch.Tensor
    mi: float  # mutual information
    mig: float  # mutual information gap
    unibound_l: float  # unibound (lower)
    unibound_u: float  # unibound (upper)

    def get_scores(self) -> dict[str, float]:
        return {
            "mi": self.mi,
            "mig": self.mig,
            "unibound_l": self.unibound_l,
            "unibound_u": self.unibound_u,
        }

    def save(self, path: pathlib.Path) -> None:
        with log_util.torch_sci_mode_disabled():
            with open(path, "w") as f:
                f.write(f"mi_zi_yj=\n{self.mi_zi_yj}\n")
                f.write(f"mi_zmi_yj=\n{self.mi_zmi_yj}\n")
                f.write(f"mi_z_yj=\n{self.mi_z_yj}\n")
                # f.write(f"ri_zi_zmi_yj=\n{self.ri_zi_zmi_yj}\n")
                f.write(f"mi={self.mi}\n")
                f.write(f"mig={self.mig}\n")
                f.write(f"unibound_l={self.unibound_l}\n")
                f.write(f"unibound_u={self.unibound_u}\n")
                # f.write(f"pui={self.pui}\n")


@torch.no_grad()
def mi_metrics(
    model: models.LatentVariableModel,
    dataset: data.DatasetWithFactors,
) -> MIMetrics:
    ent_yj = dataset.factor_entropies()

    def normalize(s: torch.Tensor) -> torch.Tensor:
        return (s.cpu() / ent_yj).clip(0, 1)

    _logger.info("computing mi_zi_yj...")
    mi_zi_yj = normalize(_compute_mi_zi_yj(dataset, model.aggregated_entropy))
    _logger.info("computing mi_zmi_yj...")
    mi_zmi_yj = normalize(_compute_mi_zi_yj(dataset, model.aggregated_loo_entropy))
    _logger.info("computing mi_z_yj...")
    mi_z_yj = normalize(_compute_mi_zi_yj(dataset, model.aggregated_joint_entropy))
    mi = mi_zi_yj.amax(0).mean().item()
    mig = _gap(mi_zi_yj, 0).mean().item()
    ub_l = (mi_zi_yj - mi_zmi_yj).relu().amax(0).mean().item()
    ub_u = torch.minimum((mi_z_yj - mi_zmi_yj).relu(), mi_zi_yj).amax(0).mean().item()
    return MIMetrics(mi_zi_yj, mi_zmi_yj, mi_z_yj, mi, mig, ub_l, ub_u)


def _compute_mi_zi_yj(
    dataset: data.DatasetWithFactors, entropy_fn: EntropyFn, sample_size: int = 10_000
) -> torch.Tensor:
    inner_bsize = 256
    outer_bsize = 1024

    def ent(dataset: torch.utils.data.Dataset[Any], sample_size: int) -> torch.Tensor:
        sample_size = min(sample_size, data.dataset_size(dataset))
        sample = data.subsample(dataset, sample_size)
        if sample_size <= 10_000:
            return entropy_fn(dataset, sample, inner_bsize, outer_bsize)

        n_samples = (sample_size - 1) // 10_000 + 1
        each_size = (sample_size - 1) // n_samples + 1
        subsets = data.split(sample, each_size)
        ents = [
            entropy_fn(dataset, subset, inner_bsize, outer_bsize) * len(subset)
            for subset in subsets
        ]
        return torch.stack(ents).sum(0) / sample_size

    # Compute H(z_i).
    ent_zi = ent(dataset, sample_size)

    # Compute H(z_i|y_j).
    # Note: we assume p(y_j) is a uniform distribution.
    ent_zi_yj_list: list[torch.Tensor] = []
    for j in range(dataset.n_factors):
        ent_zi_yj_points: list[torch.Tensor] = []
        for yj in range(dataset.n_factor_values[j]):
            subset = dataset.fix_factor(j, yj)
            ent_zi_yj_point = ent(subset, sample_size)
            ent_zi_yj_points.append(ent_zi_yj_point)
        ent_zi_yj_list.append(torch.stack(ent_zi_yj_points).mean(0))
    ent_zi_yj = torch.stack(ent_zi_yj_list, 1)

    return ent_zi[:, None] - ent_zi_yj


def _gap(x: torch.Tensor, dim: int, largest: bool = True) -> torch.Tensor:
    (top2, _) = torch.topk(x, 2, dim, largest)
    top2 = torch.movedim(top2, dim, 0)
    return top2[0] - top2[1] if largest else top2[1] - top2[0]
