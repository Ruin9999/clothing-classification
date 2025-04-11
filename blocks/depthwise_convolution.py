import torch
import torch.nn as nn

class DepthwiseConv(nn.Module):
  def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    super().__init__()
    self.act_fn = nn.GELU()
    self.norm = nn.BatchNorm2d(in_channels) 
    self.depthwise_conv = nn.Conv2d(
      in_channels,
      in_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      groups=in_channels,
      bias=False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.norm(x)
    x = self.act_fn(x)
    x = self.depthwise_conv(x)
    return x