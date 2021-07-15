import torch


class Discriminator(torch.nn.Sequential):
    def __init__(
        self, n_in: int, n_out: int, width: int = 1000, depth: int = 6
    ) -> None:
        assert depth >= 2
        layers: list[torch.nn.Module] = [torch.nn.Linear(n_in, width)]
        for _ in range(depth - 2):
            layers += [torch.nn.LeakyReLU(0.2, True), torch.nn.Linear(width, width)]
        layers.append(torch.nn.Linear(width, n_out))
        super().__init__(*layers)

        self.in_features = n_in
        self.out_features = n_out
