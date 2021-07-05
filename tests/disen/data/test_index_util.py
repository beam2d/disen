import disen


def test_strided_indices() -> None:
    indices = disen.data.StridedIndices(
        shape=[2, 3, 4], strides=[36, 12, 1], offset=4
    )
    assert indices[0] == 4
    assert indices[1] == 5
    assert indices[10] == 30
    assert indices[20] == 64


def test_skip_dim() -> None:
    assert disen.data.skip_dim([2, 3, 4, 5], 2) == [2, 3, 5]
