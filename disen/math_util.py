import functools
import operator
from typing import Iterable, Protocol, TypeVar


_MulT = TypeVar("_MulT", bound="Multipliable")


class Multipliable(Protocol):
    def __mul__(self: _MulT, other: _MulT) -> _MulT: ...


def prod(xs: Iterable[_MulT]) -> _MulT:
    return functools.reduce(operator.mul, xs)
