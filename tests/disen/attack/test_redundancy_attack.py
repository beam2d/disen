import math

import pytest
import torch

import disen


class NoiseModel(disen.models.LatentVariableModel):
    """A test model that just adds a noise to the input."""

    loss_keys = ()

    def __init__(self, std: torch.Tensor) -> None:
        super().__init__()
        self.spec = disen.models.LatentSpec(
            [disen.models.SingleLatentSpec("z", "real", std.shape[1])]
        )
        self.std = std

    def encode(self, x: torch.Tensor) -> list[disen.distributions.Distribution]:
        return [disen.distributions.Normal(x, self.std)]


@pytest.fixture
def D() -> int:
    return 5


@pytest.fixture
def alpha() -> float:
    return 0.75


@pytest.fixture
def U(D: int) -> torch.Tensor:
    return torch.eye(D) - torch.full((D, D), 2 / D)


@pytest.fixture
def std(D: int) -> torch.Tensor:
    return torch.linspace(1, D, D).sqrt()


@pytest.fixture
def model(
    D: int, alpha: float, U: torch.Tensor, std: torch.Tensor
) -> disen.attack.RedundancyAttack:
    base = NoiseModel(std[None])
    return disen.attack.RedundancyAttack(base, alpha=alpha, U=U)


@pytest.fixture
def cov(D: int, alpha: float, U: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    aU = alpha * U
    cov00 = (std ** 2 + 1.0).diag()
    cov01 = cov00 @ aU.T
    cov10 = aU @ cov00
    cov11 = aU @ cov00 @ aU.T + torch.eye(D)
    cov0x = torch.cat([cov00, cov01], 1)
    cov1x = torch.cat([cov10, cov11], 1)
    return torch.cat([cov0x, cov1x], 0)


def test_redundancy_attack_aggregated_entropy(
    model: disen.attack.RedundancyAttack,
    D: int,
    cov: torch.Tensor,
) -> None:
    N = 1000
    dataset = torch.utils.data.TensorDataset(torch.randn(N, D))
    H = model.aggregated_entropy(dataset, N, N, N)

    expect = _normal_entropy(cov.diag())

    torch.testing.assert_allclose(H, expect, atol=0.05, rtol=0.01)


def test_redundancy_attack_aggregated_loo_entropy(
    model: disen.attack.RedundancyAttack,
    D: int,
    cov: torch.Tensor,
) -> None:
    N = 1000
    dataset = torch.utils.data.TensorDataset(torch.randn(N, D))
    H = model.aggregated_loo_entropy(dataset, N, N, N)

    logdetcov = disen.nn.principal_submatrices(cov).logdet()
    expect = ((2 * D - 1) * math.log(2 * math.pi * math.e) + logdetcov) / 2

    torch.testing.assert_allclose(H, expect, atol=0.05, rtol=0.01)


def _normal_entropy(var: torch.Tensor) -> torch.Tensor:
    return (2 * math.pi * math.e * var).log() / 2
