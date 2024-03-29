from typing import Iterable, Optional

import torch


def block_matrix(blocks: list[list[torch.Tensor]]) -> torch.Tensor:
    rows = [torch.cat(row, -1) for row in blocks]
    return torch.cat(rows, -2)


def enumerate_loo(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Enumerate all leave-one-out subvectors.

    For n-vector, it returns (n, n - 1)-matrix. The original axis is shrinked to size
    (n - 1) and a new axis of size n is inserted to the leftmost position.
    """
    n = x.shape[dim]
    dim %= x.ndim
    return offdiagonal(x.expand(n, *x.shape), 0, dim + 1)


def gini_variance(x: torch.Tensor, n_categories: int, dim: int = -1) -> torch.Tensor:
    """Gini's empirical variance for categorical data.

    For a vector (x_1, ..., x_N) of values in range(n_categories), Gini's empirical
    variance is defined by the mean of 1 * (x_i != x_j) for all index pairs with
    i != j.
    """
    x = x.movedim(dim, -1)
    N = x.shape[-1]
    count = torch.zeros(x.shape[:-1] + (n_categories,), dtype=x.dtype, device=x.device)
    ones = torch.ones((), dtype=x.dtype, device=x.device).expand_as(x)
    count.scatter_add_(-1, x, ones)
    return (N ** 2 - (count ** 2).sum(-1)) / (2 * N * (N - 1))


def offdiagonal(x: torch.Tensor, dim1: int = -2, dim2: int = -1) -> torch.Tensor:
    """Extract non-diagonal elements from matrices.

    For (n, n)-matrix, it returns (n, n - 1)-matrix.
    """
    x = torch.movedim(x, (dim1, dim2), (-2, -1))
    *s, n, _ = x.shape
    assert x.shape[-1] == n, "non-square matrix is not supported"
    x = x.reshape(*s, n * n)[..., :-1]
    x = x.reshape(*s, n - 1, n + 1)[..., 1:]
    x = x.reshape(*s, n, n - 1)
    return torch.movedim(x, (-2, -1), (dim1, dim2))


def principal_submatrices(
    x: torch.Tensor, dim1: int = -2, dim2: int = -1
) -> torch.Tensor:
    """Extract the principal submatrices from a matrix.

    Here, the principal submatrices are the submatrices obtained by removing the row
    and column at the same position from the original matrix.

    For (n, n)-matrix, it returns (n, n - 1, n - 1)-tensor. It inserts a new axis of
    size n to the left.
    """
    n = x.shape[dim1]
    assert x.shape[dim2] == n, "non-square matrix is not supported"
    x = torch.stack([x] * n)
    x = offdiagonal(x, 0, dim1)
    x = offdiagonal(x, 0, dim2)
    return x


def shuffle(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.movedim(dim, -1)
    perm = torch.randperm(x.shape[-1], device=x.device)
    x = x[..., perm]
    return x.movedim(-1, dim)


def tensor_sum(xs: Iterable[torch.Tensor]) -> torch.Tensor:
    ret: Optional[torch.Tensor] = None
    for x in xs:
        if ret is None:
            ret = x.clone()
        else:
            ret += x
    assert ret is not None
    return ret
