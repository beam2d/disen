import disen.distributions as DD

import torch


def test_relaxed_one_hot_categorical() -> None:
    tau = torch.as_tensor([[0.1], [0.5], [1.0]])
    logits = torch.randn(3, 2, 4)
    distr = DD.RelaxedOneHotCategorical(tau, logits)

    temper = distr.base.temperature
    assert temper.shape == (3, 2)
    assert torch.allclose(tau, temper)
