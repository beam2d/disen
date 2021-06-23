from typing import Iterable, Optional

import torch


def tensor_sum(xs: Iterable[torch.Tensor]) -> torch.Tensor:
    ret: Optional[torch.Tensor] = None
    for x in xs:
        if ret is None:
            ret = x.clone()
        else:
            ret += x
    assert ret is not None
    return ret
