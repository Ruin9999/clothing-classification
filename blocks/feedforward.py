import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, ch_mult: int = 4, dropout: int = 0.2) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(in_channels, out_channels * ch_mult)
    self.linear_2 = nn.Linear(out_channels * ch_mult, out_channels)
    self.act = nn.SiLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear_1(x)
    x = self.act(x)
    x = self.dropout(x)
    x = self.linear_2(x)

    return x