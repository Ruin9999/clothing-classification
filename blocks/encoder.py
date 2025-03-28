import torch
import torch.nn as nn
from torch.nn import functional as F

from blocks import SelfAttention, FeedForward

class Encoder(nn.Module):
  def __init__(self, in_channels: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.2) -> None:
    super().__init__()
    self.in_channels = in_channels
    self.num_heads = num_heads
    self.mlp_ratio = mlp_ratio
    self.dropout = dropout

    self.norm_1 = nn.BatchNorm2d(in_channels)
    self.attention = SelfAttention(in_channels, num_heads=num_heads)
    self.norm_2 = nn.BatchNorm2d(in_channels)
    self.feedforward = FeedForward(in_channels, in_channels, ch_mult=mlp_ratio, dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, hidden_channels, height, width)

    residual = x
    x = self.norm_1(x) # (batch_size, hidden_channels, height, width)
    x = self.attention(x) #(batch_size, hidden_channels, height, width) -> 
    x = x + residual

    residual = x
    x = self.norm_2(x)
    x = self.feedforward(x)
    x = x + residual

    return x