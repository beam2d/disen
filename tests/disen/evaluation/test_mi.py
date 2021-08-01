from disen.data.dataset_with_factors import ImageWithFactors
from typing import Sequence

import pytest
import torch
import torch.nn.functional as F

import disen


@pytest.fixture
def n_categories() -> list[int]:
    return [2, 3, 4]


class SimpleModel(disen.models.LatentVariableModel):
    def __init__(self, n_categories: Sequence[int]) -> None:
        super().__init__()
        specs = [
            disen.models.SingleLatentSpec(f"y{i + 1}", "categorical", 1, k)
            for i, k in enumerate(n_categories)
        ]
        self.spec = disen.models.LatentSpec(specs)

    def encode(self, y: torch.Tensor) -> list[disen.distributions.Distribution]:
        m = self.spec.size
        assert m <= y.shape[1]

        q_zs: list[disen.distributions.Distribution] = []
        for i in range(m):
            yi = y[:, i]
            k = self.spec[i].n_categories
            # yi w.p. 2/3, other labels uniformly
            p = torch.where(F.one_hot(yi, k) == 1.0, 2 / 3, 1 / (3 * (k - 1)))
            q_zs.append(disen.distributions.OneHotCategoricalWithProbs(p[:, None, :]))
        return q_zs


def test_model(n_categories: list[int]) -> None:
    model = SimpleModel(n_categories)
    y = torch.as_tensor([[0, 0, 1], [1, 2, 3]])
    q_zs = model.encode(y)
    assert len(q_zs) == len(n_categories)

    assert isinstance(q_zs[0], disen.distributions.OneHotCategoricalWithProbs)
    assert isinstance(q_zs[1], disen.distributions.OneHotCategoricalWithProbs)
    assert isinstance(q_zs[2], disen.distributions.OneHotCategoricalWithProbs)
    torch.testing.assert_allclose(
        q_zs[0].base.probs,
        torch.as_tensor([[[2 / 3, 1 / 3]], [[1 / 3, 2 / 3]]]),
    )
    torch.testing.assert_allclose(
        q_zs[1].base.probs,
        torch.as_tensor([[[2 / 3, 1 / 6, 1 / 6]], [[1 / 6, 1 / 6, 2 / 3]]]),
    )
    torch.testing.assert_allclose(
        q_zs[2].base.probs,
        torch.as_tensor(
            [[[1 / 9, 2 / 3, 1 / 9, 1 / 9]], [[1 / 9, 1 / 9, 1 / 9, 2 / 3]]]
        ),
    )


class SimpleDataset(disen.data.DatasetWithFactors):
    def __init__(self, n_categories: Sequence[int]) -> None:
        super().__init__()
        self.n_factors = len(n_categories)
        self.n_factor_values = tuple(n_categories)

        self.strides = (
            torch.as_tensor([*n_categories[1:], 1]).flip(0).cumprod(0).flip(0)
        )
        print(self.strides)
        self.size = int(self.strides[0]) * n_categories[0]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> ImageWithFactors:
        labels: list[int] = []
        for stride in self.strides.tolist():
            yj, index = divmod(index, stride)
            labels.append(yj)
        y = torch.as_tensor(labels, dtype=torch.int64)
        return (y, y)

    def get_index(self, factors: torch.Tensor) -> torch.Tensor:
        return factors @ self.strides

    def fix_factor(
        self, factor: int, value: int
    ) -> torch.utils.data.Subset[ImageWithFactors]:
        strides = self.strides.tolist()
        indices = disen.data.StridedIndices(
            disen.data.skip_dim(self.n_factor_values, factor),
            disen.data.skip_dim(strides, factor),
            value * strides[factor],
        )
        return torch.utils.data.Subset(self, indices)


def test_dataset(n_categories: list[int]) -> None:
    dataset = SimpleDataset(n_categories + [10])
    assert dataset[24][0].tolist() == [0, 0, 2, 4]
    assert dataset[86][0].tolist() == [0, 2, 0, 6]
    assert dataset[172][1].tolist() == [1, 1, 1, 2]


def test_mi(n_categories: list[int]) -> None:
    model = SimpleModel(n_categories)
    # 1,000 points for each combination of factors
    dataset = SimpleDataset(n_categories + [100])

    mi_metrics = disen.evaluation.mi_metrics(model, dataset)
    mi_metrics = disen.evaluation.MIMetrics(
        mi_zi_yj=mi_metrics.mi_zi_yj[:, :3],
        mi_zmi_yj=mi_metrics.mi_zmi_yj[:, :3],
        mi_z_yj=mi_metrics.mi_z_yj[:, :3],
        mi=mi_metrics.mi * 4 / 3,
        mig=mi_metrics.mig * 4 / 3,
        unibound_l=mi_metrics.unibound_l * 4 / 3,
        unibound_u=mi_metrics.unibound_u * 4 / 3,
    )

    i = torch.arange(1, 4)[:, None]
    j = torch.arange(1, 4)[None, :]
    eye = torch.eye(3)
    ones = torch.ones((3, 3))
    nondiag = ones - eye

    ent_j = (j + 1).log()
    expect_mi_zi_yj = eye * ((4 * (i + 1) ** 3 / (27 * i)).log() / 3) / ent_j
    expect_mi_zmi_yj = nondiag * (((4 * (j + 1) ** 3) / (27 * j)).log() / 3) / ent_j
    expect_mi_z_yj = (4 * (j + 1) ** 3 / (27 * j)).log() / 3 / ent_j
    expect_mi = expect_mi_zi_yj.diag().mean().item()
    expect_mig = expect_mi
    expect_ub_l = expect_mi
    expect_ub_u = expect_mi_z_yj.mean()

    def printt(name: str, t: torch.Tensor) -> None:
        print(f"{name}=\n{t}")

    with disen.torch_sci_mode_disabled():
        printt("expect_mi_zi_yj", expect_mi_zi_yj)
        printt("expect_mi_zmi_yj", expect_mi_zmi_yj)
        printt("expect_mi_z_yj", expect_mi_z_yj)
        printt("actual_mi_zi_yj", mi_metrics.mi_zi_yj)
        printt("actual_mi_zmi_yj", mi_metrics.mi_zmi_yj)
        printt("actual_mi_z_yj", mi_metrics.mi_z_yj)

    torch.testing.assert_allclose(
        mi_metrics.mi_zi_yj, expect_mi_zi_yj, rtol=0.1, atol=0.01
    )
    torch.testing.assert_allclose(
        mi_metrics.mi_zmi_yj, expect_mi_zmi_yj, rtol=0.1, atol=0.01
    )
    torch.testing.assert_allclose(
        mi_metrics.mi_z_yj, expect_mi_z_yj, rtol=0.3, atol=0.03
    )
    assert abs(mi_metrics.mi - expect_mi) < 0.05
    assert abs(mi_metrics.mig - expect_mig) < 0.05
    assert abs(mi_metrics.unibound_l - expect_ub_l) < 0.05
    assert abs(mi_metrics.unibound_u - expect_ub_u) < 0.05


def _tensor_mean(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors).mean(0)


def _value_mean(values: Sequence[float]) -> float:
    return torch.as_tensor(values).mean(0).item()
