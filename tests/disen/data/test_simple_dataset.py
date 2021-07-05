import torch

import disen


def test_simple_dataset() -> None:
    dataset = disen.data.SimpleDataset((2, 3, 4, 5))
    assert len(dataset) == 2 * 3 * 4 * 5

    assert dataset[0][1].tolist() == [0, 0, 0, 0]
    assert dataset[1][1].tolist() == [0, 0, 0, 1]
    assert dataset[13][1].tolist() == [0, 0, 2, 3]
    assert dataset[103][1].tolist() == [1, 2, 0, 3]

    assert dataset.get_index(torch.as_tensor([0, 2, 1, 3])).item() == 48

    subset = dataset.fix_factor(2, 3)
    assert subset[0][1].tolist() == [0, 0, 3, 0]
    assert subset[1][1].tolist() == [0, 0, 3, 1]
    assert subset[13][1].tolist() == [0, 2, 3, 3]
    assert subset[27][1].tolist() == [1, 2, 3, 2]
