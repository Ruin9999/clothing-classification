import torch
import torch.nn as nn
from torch.nn import functional as F

# class FeedForward(nn.Module):
#   def __init__(self, in_channels: int, out_channels: int, ch_mult: int = 4, dropout: float = 0.2) -> None:
#     super().__init__()
#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.ch_mult = ch_mult
#     self.dropout = dropout

#     self.linear_in = nn.Linear(in_channels, in_channels * ch_mult)
#     self.linear_out = nn.Linear(in_channels * ch_mult, out_channels)
#     self.dropout = nn.Dropout(dropout)
#     self.act_fn = nn.ReLU()

#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     x = self.linear_in(x)
#     x = self.act_fn(x)
#     x = self.dropout(x)
#     x = self.linear_out(x)
    
#     return x
  
class FeedForward(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, ch_mult: int = 4, dropout: float = 0.2) -> None:
    super().__init__()

    self.conv_in = nn.Conv2d(in_channels, in_channels * ch_mult, kernel_size=1, stride=1, padding=0)
    self.conv_out = nn.Conv2d(in_channels * ch_mult, out_channels, kernel_size=1, stride=1, padding=0)
    self.dropout = nn.Dropout(dropout)
    self.act_fn = nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_in(x)
    x = self.act_fn(x)
    x = self.dropout(x)
    x = self.conv_out(x)
    
    return x