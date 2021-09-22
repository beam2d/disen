from typing import Callable, Optional, TypeVar


_T = TypeVar("_T")


def parse_optional(f: Callable[[str], _T]) -> Callable[[str], Optional[_T]]:
    def wrapped(s: str) -> Optional[_T]:
        if s == "None":
            return None
        return f(s)

    return wrapped
