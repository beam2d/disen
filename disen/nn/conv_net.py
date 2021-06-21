from typing import Callable

import torch

from . import encoder


class SimpleConvNet(torch.nn.Sequential, encoder.EncoderBase):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        out_features: int,
        act_fn: Callable[[], torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        assert image_size in (32, 64)
        self.out_features = out_features

        def nop_if_small(l: list[torch.nn.Module]) -> list[torch.nn.Module]:
            return [] if image_size == 32 else l

        super().__init__(
            torch.nn.Conv2d(in_channels, 32, 4, 2, 1),
            act_fn(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            act_fn(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),
            act_fn(),
            *nop_if_small(
                [
                    torch.nn.Conv2d(64, 64, 4, 2, 1),
                    act_fn(),
                ]
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, out_features),
            act_fn(),
        )


class SimpleTransposedConvNet(torch.nn.Sequential):
    def __init__(
        self,
        image_size: int,
        in_features: int,
        out_channels: int,
        act_fn: Callable[[], torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        assert image_size in (32, 64)
        self.out_channels = out_channels

        def nop_if_small(l: list[torch.nn.Module]) -> list[torch.nn.Module]:
            return [] if image_size == 32 else l

        super().__init__(
            torch.nn.Linear(in_features, 256),
            act_fn(),
            torch.nn.Linear(256, 64 * 4 * 4),
            act_fn(),
            torch.nn.Unflatten(1, (64, 4, 4)),
            *nop_if_small(
                [
                    torch.nn.ConvTranspose2d(64, 64, 4, 2, 1),
                    act_fn(),
                ]
            ),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            act_fn(),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1),
            act_fn(),
            torch.nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
        )
