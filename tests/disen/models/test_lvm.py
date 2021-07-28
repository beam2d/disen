import math

import pytest
import torch

import disen


class NoiseModel(disen.models.LatentVariableModel):
    """A test model that just adds a noise to the input."""

    loss_keys = ["loss"]

    def __init__(self, std1: torch.Tensor, std2: torch.Tensor) -> None:
        super().__init__()
        self.spec = disen.models.LatentSpec(
            [
                disen.models.SingleLatentSpec("z1", "real", std1.shape[1]),
                disen.models.SingleLatentSpec("z2", "real", std2.shape[1]),
            ]
        )
        self.std1 = std1
        self.std2 = std2

    def encode(self, x: torch.Tensor) -> list[disen.distributions.Distribution]:
        return [
            disen.distributions.Normal(x[..., : self.std1.shape[1]], self.std1),
            disen.distributions.Normal(x[..., self.std1.shape[1] :], self.std2),
        ]


@pytest.fixture
def model() -> NoiseModel:
    std1 = torch.as_tensor([[1.0, 2.0, 3.0]])
    std2 = torch.as_tensor([[1.0, 3.0, 5.0]])
    return NoiseModel(std1, std2)


@pytest.fixture
def x() -> torch.Tensor:
    return torch.as_tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]])


@pytest.fixture
def dataset() -> torch.utils.data.TensorDataset:
    return torch.utils.data.TensorDataset(torch.randn((100000, 6)))


@pytest.fixture
def z1() -> torch.Tensor:
    return torch.as_tensor([[0.0, 2.0, 4.0]])


@pytest.fixture
def z2() -> torch.Tensor:
    return torch.as_tensor([[1.0, -1.0, 3.0]])


def test_log_posterior(
    model: NoiseModel, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
) -> None:
    log_q = model.log_posterior(x, [z1, z2])

    d1 = torch.distributions.Normal(x[:, :3], model.std1)
    d2 = torch.distributions.Normal(x[:, 3:], model.std2)
    expect = torch.empty((1, 6), dtype=x.dtype)
    expect[:, :3] = d1.log_prob(z1)
    expect[:, 3:] = d2.log_prob(z2)

    torch.testing.assert_allclose(log_q, expect)


def test_log_loo_posterior(
    model: NoiseModel, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
) -> None:
    log_q_loo = model.log_loo_posterior(x, [z1, z2])

    d1 = torch.distributions.Normal(x[:, :3], model.std1)
    d2 = torch.distributions.Normal(x[:, 3:], model.std2)
    log_q = torch.empty((1, 6), dtype=x.dtype)
    log_q[:, :3] = d1.log_prob(z1)
    log_q[:, 3:] = d2.log_prob(z2)
    expect = torch.empty_like(log_q)
    for i in range(6):
        expect[:, i] = log_q[:, :i].sum(1) + log_q[:, i + 1 :].sum(1)

    torch.testing.assert_allclose(log_q_loo, expect)


def test_log_joint_posterior(
    model: NoiseModel, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
) -> None:
    log_q_joint = model.log_joint_posterior(x, [z1, z2])

    d1 = torch.distributions.Normal(x[:, :3], model.std1)
    d2 = torch.distributions.Normal(x[:, 3:], model.std2)
    log_q1 = d1.log_prob(z1).sum(1, keepdim=True)
    log_q2 = d2.log_prob(z2).sum(1, keepdim=True)
    expect = log_q1 + log_q2

    torch.testing.assert_allclose(log_q_joint, expect)


def test_aggregated_entropy(model: NoiseModel) -> None:
    N = 1000
    dataset = torch.utils.data.TensorDataset(torch.randn(N, 6))
    H = model.aggregated_entropy(dataset, dataset, N, N)

    var1 = model.std1[0].square() + 1.0
    var2 = model.std2[0].square() + 1.0
    var = torch.cat([var1, var2])
    entropy = (2 * math.pi * math.e * var).log() / 2

    torch.testing.assert_allclose(H, entropy, atol=0.05, rtol=0.01)


def test_aggregated_loo_entropy(model: NoiseModel) -> None:
    N = 1000
    dataset = torch.utils.data.TensorDataset(torch.randn(N, 6))
    H = model.aggregated_loo_entropy(dataset, dataset, N, N)

    var1 = model.std1[0].square() + 1.0
    var2 = model.std2[0].square() + 1.0
    var = torch.cat([var1, var2])
    entropy = (2 * math.pi * math.e * var).log() / 2
    loo_entropy = entropy.sum() - entropy

    torch.testing.assert_allclose(H, loo_entropy, atol=0.05, rtol=0.01)


def test_aggregated_joint_entropy(model: NoiseModel) -> None:
    N = 1000
    N = 1000
    dataset = torch.utils.data.TensorDataset(torch.randn(N, 6))
    H = model.aggregated_joint_entropy(dataset, dataset, N, N)

    var1 = model.std1[0].square() + 1.0
    var2 = model.std2[0].square() + 1.0
    var = torch.cat([var1, var2])
    entropy = (2 * math.pi * math.e * var).log().sum(0, keepdim=True) / 2

    torch.testing.assert_allclose(H, entropy, atol=0.05, rtol=0.01)
