import random
from typing import Any, TypeVar, cast

import torch


_ItemType = TypeVar("_ItemType")

def subsample(
    data: torch.utils.data.Dataset[_ItemType], sample_size: int
) -> torch.utils.data.Dataset[_ItemType]:
    total_size = dataset_size(data)
    if total_size == sample_size:
        return data
    indices = random.sample(range(total_size), sample_size)
    return torch.utils.data.Subset(data, indices)


def dataset_size(data: torch.utils.data.Dataset[_ItemType]) -> int:
    # Dataset does not provide __len__, but we assume it here.
    return len(cast(Any, data))
