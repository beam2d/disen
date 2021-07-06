import torch
import torch.nn.functional as F

import disen


class Model(disen.models.LatentVariableModel):
    loss_keys = ()

    def __init__(self) -> None:
        super().__init__()
        self.spec = disen.models.LatentSpec(
            [
                disen.models.SingleLatentSpec("c", "categorical", 2, 8),
                disen.models.SingleLatentSpec("z", "real", 2),
            ]
        )

    def encode(self, x: torch.Tensor) -> list[disen.distributions.Distribution]:
        # x = [a, b, c, d]
        # logits: concentrate to category (a % 2, a // 2 * b)
        t = x.long()
        logits0 = F.one_hot(t[:, 0] % 2, 4)
        logits1 = F.one_hot(t[:, 0] // 2 * t[:, 1], 4)
        logits = torch.stack([logits0, logits1], -2) * 100.0
        # normal: concentrate to (c % 2, c // 2 * d)
        loc = torch.stack([x[:, 2] % 2, x[:, 2] // 2 * x[:, 3]], -1)
        scale = torch.full_like(loc, 0.01)
        return [
            disen.distributions.OneHotCategorical(logits),
            disen.distributions.Normal(loc, scale),
        ]


def test_factor_vae() -> None:
    model = Model()
    dataset = disen.data.SimpleDataset((4, 4, 4, 4))

    def compute_score() -> float:
        return disen.evaluation.factor_vae_score(
            model,
            dataset,
            n_train=128,
            n_test=128,
            sample_size=32,
            n_normalizer_data=128,
        )

    # The model output concentrates to the following values
    # (y0, y1) => (z0, z1)
    # (0, x)   => (0, 0)
    # (1, x)   => (1, 0)
    # (2, x)   => (0, x)
    # (3, x)   => (1, x)
    # and ((y2, y3), (z2, z3)) behaves similarly (with noise).
    # Therefore,
    # - fixing y0=0,1: variances of z0,z1 are both zero
    # - fixing y0=2,3: variance of z0 is zero; that of z1 is positive
    # - fixing y1: variance of z1 is zero; that of z0 is positive
    # so the desirable accuracy is (1.0 + 0.75) / 2 = 0.875

    N = 10
    score = sum(compute_score() for _ in range(N)) / N
    assert 0.85 < score < 0.9, score
