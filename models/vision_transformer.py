import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from blocks import Encoder

# Architecture referenced from https://arxiv.org/abs/2010.11929
class SimplifiedVisionTransformer(nn.Module):
  def __init__(
    self,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_transformer_layers: int,
    num_heads: int = 7,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    image_height: int = 28,
    image_width: int = 28,
  ) -> None:
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.num_transformer_layers = num_transformer_layers
    self.num_heads = num_heads
    self.mlp_ratio = mlp_ratio
    self.dropout = dropout

    # Image Embedding
    self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)

    # Positional Embedding
    self.position_embedding = nn.Parameter(torch.randn(1, hidden_channels, image_height, image_width))

    # Transformer
    self.transformers = nn.Sequential(*[
        Encoder(hidden_channels, num_heads, mlp_ratio, dropout)
        for _ in range(num_transformer_layers)
    ])

    # Classification Head
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.linear_out = nn.Linear(hidden_channels, out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, in_channels, height, width)
    batch_size = x.shape[0]

    # Embed Image
    x = self.conv_in(x)

    # Add Positional Embedding
    x = x + self.position_embedding

    # Transformer
    x = self.transformers(x)

    # Global average pooling suggested by https://arxiv.org/abs/2205.01580
    x = self.avg_pool(x)
    x = x.view(batch_size, -1)
    x = self.linear_out(x)

    return x