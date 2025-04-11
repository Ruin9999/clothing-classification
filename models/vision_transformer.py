import torch
import torch.nn as nn

from blocks import Encoder, OptimizedEncoder
from utils import PositionalEmbedding2D

# Architecture referenced from https://arxiv.org/abs/2010.11929
# Code referenced from https://github.com/google-research/vision_transformer
class VisionTransformer(nn.Module):
  def __init__(
    self,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_transformer_layers: int,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    patch_size: int = 7,
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
    self.patch_size = patch_size
    self.image_height = image_height
    self.image_width = image_width

    assert image_height == image_width, f"image_height and width have to be the same but are image_height: {image_height} and image_width: {image_width}."
    assert image_height % patch_size == 0, f"image_height and width have to be divisible by patch_size but are image_height: {image_height} and patch_size: {patch_size}."

    self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=patch_size, stride=patch_size, padding=0)
    self.positional_embedding = nn.Parameter(torch.randn(1, hidden_channels, image_height // patch_size, image_width // patch_size)).to(device = next(self.parameters()).device)
    self.positional_embedding_dropout = nn.Dropout(dropout)
    self.transformer = nn.Sequential(*[
      Encoder(hidden_channels, num_heads, mlp_ratio, dropout)
      for _ in range(num_transformer_layers)
    ])

    # Classification Head
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.linear_out = nn.Linear(hidden_channels, out_channels)    

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, in_channels, height, width)
    x = self.conv_in(x)
    x = x + self.positional_embedding
    x = self.positional_embedding_dropout(x)
    x = self.transformer(x)
    
    # Global average pooling suggested by https://arxiv.org/abs/2205.01580
    x = self.avg_pool(x)
    x = torch.flatten(x, 1)
    x = self.linear_out(x)

    return x