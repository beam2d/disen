import random
from typing import Any, TypeVar, cast

import torch


_ItemType = TypeVar("_ItemType")


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


def dataset_size(data: torch.utils.data.Dataset[_ItemType]) -> int:
    # Dataset does not provide __len__, but we assume it here.
    return len(cast(Any, data))
