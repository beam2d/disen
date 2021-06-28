import disen.distributions as DD

import torch


def test_relaxed_one_hot_categorical() -> None:
    tau = torch.as_tensor(0.1)
    logits = torch.randn(3, 2, 4)
    distr = DD.RelaxedOneHotCategorical(tau, logits)

    temper = distr.base.temperature
    assert (tau == temper).all()
