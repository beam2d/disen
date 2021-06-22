import torch


class BatchAccumulator:
    def __init__(self) -> None:
        self.accum: dict[str, torch.Tensor] = {}
        self.total_size = 0

    def accumulate(self, losses: dict[str, torch.Tensor]) -> None:
        batch_size = 0
        for k, v in losses.items():
            if batch_size == 0:
                batch_size = v.shape[0]
            else:
                assert batch_size == v.shape[0]
            if k in self.accum:
                self.accum[k] += v.sum()
            else:
                self.accum[k] = v.sum()
        self.total_size += batch_size

    def mean(self) -> dict[str, float]:
        return {k: float(v) / self.total_size for k, v in self.accum.items()}
