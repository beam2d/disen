import torch

import disen


def test_buffer_list() -> None:
    N = 5
    bl = disen.nn.BufferList([torch.full((1,), i) for i in range(N)])
    assert len(bl) == N
    torch.testing.assert_equal(bl[2], torch.full((1,), 2))
    for i, b in enumerate(bl):
        torch.testing.assert_equal(b, torch.full((1,), i))
