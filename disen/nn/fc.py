import torch
import torch.nn.functional as F


class FCBase(torch.nn.Module):
    in_features: int
    out_features: int


class MLP(FCBase):
    def __init__(
        self, n_in: int, n_out: int, width: int = 1000, depth: int = 6
    ) -> None:
        super().__init__()
        assert depth >= 2

        layers: list[torch.nn.Module] = []
        w_in = n_in
        for _ in range(depth - 1):
            fc = torch.nn.Linear(w_in, width, bias=False)
            act = torch.nn.SiLU()
            layers += [fc, act]
            w_in = width
        layers.append(torch.nn.Linear(w_in, n_out))
        self.layers = torch.nn.Sequential(*layers)

        self.in_features = n_in
        self.out_features = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseNet(FCBase):
    def __init__(self, n_in: int, n_out: int, width: int = 200, depth: int = 6) -> None:
        super().__init__()
        assert depth >= 2
        layers: list[torch.nn.Module] = []
        for i in range(depth - 1):
            fc = torch.nn.Linear(n_in + i * width, width)
            act = torch.nn.SiLU()
            layers.append(torch.nn.Sequential(fc, act))
        self.mid_layers = torch.nn.ModuleList(layers)
        self.out_layer = torch.nn.Linear(n_in + (depth - 1) * width, n_out)

        self.in_features = n_in
        self.out_features = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mid_layers:
            x = torch.cat([x, layer(x)], 1)
        return self.out_layer(x)
