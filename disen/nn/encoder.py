import torch


class EncoderBase(torch.nn.Module):
    out_features: int
