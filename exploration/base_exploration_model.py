import torch
import torch.nn as nn

class BaseExplorationModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, heads: torch.LongTensor):
        raise NotImplementedError()

    def update(self, heads: torch.LongTensor):
        raise NotImplementedError()
