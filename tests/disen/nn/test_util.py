import torch

import disen


def test_block_matrix() -> None:
    a = torch.arange(6).reshape(2, 3)
    actual = disen.nn.block_matrix([[a, a * 10], [a * 100, a * 1000]])
    expect = torch.as_tensor([
        [0, 1, 2, 0, 10, 20],
        [3, 4, 5, 30, 40, 50],
        [0, 100, 200, 0, 1000, 2000],
        [300, 400, 500, 3000, 4000, 5000],
    ])
    assert (expect == actual).all()


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


def test_gini_variance() -> None:
    K = 5
    N = 4
    x = torch.randint(0, K, (N, 5, 6))

    actual = disen.nn.gini_variance(x, K, 0)
    expect = (x[None, :] != x[:, None]).sum((0, 1)) / (2 * N * (N - 1))

    torch.testing.assert_allclose(actual, expect)


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
