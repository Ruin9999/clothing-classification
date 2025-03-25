import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8, dropout: float = 0.2) -> None:
    super().__init__()
    assert (in_channels % num_heads == 0), "in_channels must be divisible by num_heads"
    assert (in_channels == out_channels), "in_channels must be equal to out_channels" # We have out_channels just to make interfacing more standardized lmao

    self.in_channels = in_channels
    self.num_heads = num_heads
    self.head_dim = in_channels // num_heads

    self.norm1 = nn.GroupNorm(32, in_channels)
    self.norm2 = nn.GroupNorm(32, in_channels)

    self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = self.norm1(x)

    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    batch_size, _, height, width = q.shape
    q = q.reshape(batch_size, self.num_heads, self.head_dim, height * width)
    k = k.reshape(batch_size, self.num_heads, self.head_dim, height * width)
    v = v.reshape(batch_size, self.num_heads, self.head_dim, height * width)

    q = q.permute(0, 3, 1, 2) # (batch_size, num_heads, head_dim, height * width) -> (batch_size, height * width, num_heads, head_dim)
    k = k.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)

    q = q.transpose(1, 2) # (batch_size, height * width, num_heads, head_dim) -> (batch_size, num_heads, height * width, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scale = int(self.head_dim) ** (-0.5)
    q.mul_(scale)

    # Perform attention
    k = k.transpose(2, 3) # (batch_size, num_heads, height * width, head_dim) -> (batch_size, num_heads, head_dim, height * width)
    output = torch.matmul(q, k)
    output = F.softmax(output, dim=-1)
    output = torch.matmul(output, v)

    output = output.transpose(1, 2) # (batch_size, num_heads, height * width, head_dim) -> (batch_size, height * width, num_heads, head_dim)
    output = output.contiguous()
    output = output.view(batch_size, height, width, -1)
    output = output.permute(0, 3, 1, 2)

    output = self.out(output)

    return output + residual
