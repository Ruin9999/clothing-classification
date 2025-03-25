import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from blocks import SelfAttention, FeedForward

# Architecture referenced from https://arxiv.org/abs/2010.11929
class TransformerEncoder(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, num_heads: int, dropout: Optional[int] = None, ch_mult: Optional[int] = None) -> None:
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.num_heads = num_heads
    self.dropout = dropout
    self.ch_mult = ch_mult

    self.norm_1 = nn.LayerNorm(in_channels)
    self.norm_2 = nn.LayerNorm(in_channels)

    self.attention = SelfAttention(in_channels, in_channels, num_heads, dropout)
    self.feedforward = FeedForward(in_channels, out_channels, ch_mult=ch_mult, dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = self.norm_1(x)
    x = self.attention(x)
    x = x + residual

    residual = x
    x = self.norm_2(x)
    x = self.feedforward(x)
    x = x + residual

    return x