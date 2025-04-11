import torch
import torch.nn as nn
from torch.nn import functional as F

from blocks import SelfAttention, FeedForward

# Takes in (batch_size, hidden_channels, height, width)
# and outputs (batch_size, hidden_channels, height, width)
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

class OptimizedEncoder(nn.Module):
  def __init__(self, in_channels: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.2) -> None:
    super().__init__()
    self.in_channels = in_channels
    self.num_heads = num_heads
    self.mlp_ratio = mlp_ratio
    self.dropout = dropout

    self.norm_1 = nn.LayerNorm(in_channels)
    self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
    self.norm_2 = nn.LayerNorm(in_channels)
    self.feedforward = FeedForward(in_channels, in_channels, ch_mult=mlp_ratio, dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [batch_size, num_channels, height, width]
    # Reshape to [batch_size, height * width, num_channels]
    residual = x
    batch_size, num_channels, height, width = x.size()
    x = x.view(batch_size, num_channels, height * width)
    x = x.permute(0, 2, 1)  # [batch_size, height * width, num_channels]
    x = self.norm_1(x)
    x, _ = self.attention(x, x, x)  # [batch_size, height * width, num_channels]
    x = x.permute(0, 2, 1)  # [batch_size, num_channels, height * width]
    x = x.view(batch_size, num_channels, height, width)  # [batch_size, num_channels, height, width]
    x = x + residual

    residual = x
    x = x.view(batch_size, num_channels, height * width)
    x = x.permute(0, 2, 1)  # [batch_size, height * width, num_channels]
    x = self.norm_2(x)
    x = x.permute(0, 2, 1)  # [batch_size, num_channels, height * width]
    x = x.view(batch_size, num_channels, height, width)  # [batch_size, num_channels, height, width]
    x = self.feedforward(x)  # [batch_size, num_channels, height, width]
    x = x + residual
    return x