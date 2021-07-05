import torch

import disen


def test_prod() -> None:
    assert 1 == disen.math_util.prod([1])
    assert 6 == disen.math_util.prod(range(1, 4))
    assert 4.5 == disen.math_util.prod((3.0, 6.0, 0.25))
    assert [4, 6] == disen.math_util.prod(
        [torch.as_tensor([2, 3]), torch.as_tensor([2])]
    ).tolist()
