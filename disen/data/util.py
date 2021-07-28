import random
from typing import Any, TypeVar, cast

import torch


_ItemType = TypeVar("_ItemType")


def dataset_size(data: torch.utils.data.Dataset[_ItemType]) -> int:
    # Dataset does not provide __len__, but we assume it here.
    return len(cast(Any, data))


def split(
    dataset: torch.utils.data.Dataset[_ItemType], each_size: int
) -> list[torch.utils.data.Subset[_ItemType]]:
    total_size = dataset_size(dataset)
    splits: list[torch.utils.data.Subset[_ItemType]] = []
    for start in range(0, total_size, each_size):
        end = min(start + each_size, total_size)
        splits.append(torch.utils.data.Subset(dataset, range(start, end)))
    return splits


def subsample(
    dataset: torch.utils.data.Dataset[_ItemType], sample_size: int
) -> torch.utils.data.Dataset[_ItemType]:
    total_size = dataset_size(dataset)
    if total_size == sample_size:
        return dataset
    # isinstance does not work with mypy here... we resort to Any.
    if hasattr(dataset, "sample_stratified"):
        return cast(Any, dataset).sample_stratified(sample_size)
    indices = random.sample(range(total_size), sample_size)
    return torch.utils.data.Subset(dataset, indices)
