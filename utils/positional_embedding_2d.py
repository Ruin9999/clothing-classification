import torch
import torch.nn as nn

class PositionalEmbedding2D(nn.Module):
  def __init__(self, max_period: int = 10000):
    super().__init__()
    self.max_period = max_period

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = x.shape

    y_positions = torch.arange(height).float()
    x_positions = torch.arange(width).float()

    freq_bands = torch.arange(num_channels // 4).float()
    freq_bands = self.max_period ** (-freq_bands / (num_channels // 4))

    x_embedding = torch.outer(freq_bands, x_positions)
    x_sin = torch.sin(x_embedding).unsqueeze(1).expand(-1, height, -1)
    x_cos = torch.cos(x_embedding).unsqueeze(1).expand(-1, height, -1)

    y_embedding = torch.outer(freq_bands, y_positions)
    y_sin = torch.sin(y_embedding).unsqueeze(2).expand(-1, -1, width)
    y_cos = torch.cos(y_embedding).unsqueeze(2).expand(-1, -1, width)

    pos_embedding = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device=x.device)
    x = x + pos_embedding

    return x