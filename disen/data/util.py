from typing import TypeVar

import torch


_ItemType = TypeVar("_ItemType")

def subsample(
    data: torch.utils.data.Dataset[_ItemType], total_size: int, sample_size: int
) -> torch.utils.data.Dataset[_ItemType]:
    weight = torch.ones(1).broadcast_to((total_size,))
    indices = weight.multinomial(sample_size)
    return torch.utils.data.Subset(data, indices.tolist())
