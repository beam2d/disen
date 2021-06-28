import torch

import disen


def test_enumerate_loo() -> None:
    a = torch.as_tensor([1, 2, 3, 4, 5])
    expect = torch.as_tensor(
        [
            [2, 3, 4, 5],
            [1, 3, 4, 5],
            [1, 2, 4, 5],
            [1, 2, 3, 5],
            [1, 2, 3, 4],
        ]
    )
    actual = disen.nn.enumerate_loo(a)
    assert (expect == actual).all()


def test_offdiagonal() -> None:
    k, n = 3, 5
    x = torch.arange(k * n * n).reshape(k, n, n)
    actual = disen.nn.offdiagonal(x)

    expect = torch.empty((k, n, n - 1), dtype=x.dtype)
    for i in range(n):
        expect[:, i, :i] = x[:, i, :i]
        expect[:, i, i:] = x[:, i, i + 1 :]

    assert (expect == actual).all()


def test_principal_submatrices() -> None:
    k, n = 3, 5
    x = torch.arange(k * n * n).reshape(k, n, n)
    actual = disen.nn.principal_submatrices(x)

    expect = torch.empty((n, k, n - 1, n - 1), dtype=x.dtype)
    for i in range(n):
        expect[i, :, :i, :i] = x[:, :i, :i]
        expect[i, :, i:, :i] = x[:, i + 1 :, :i]
        expect[i, :, :i, i:] = x[:, :i, i + 1 :]
        expect[i, :, i:, i:] = x[:, i + 1 :, i + 1 :]

    assert (expect == actual).all()
