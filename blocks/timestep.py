import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Implementation from https://arxiv.org/abs/2112.10752
class Timestep(nn.Module):
  def __init__(self, timestep_embedding_channels: int, max_period: int = 10000):
    super().__init__()
    self.timestep_embedding_channels = timestep_embedding_channels
    self.max_period = max_period
  
  def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
    half_dim = self.timestep_embedding_channels // 2
    exponent = -math.log(self.max_period) 
    exponent = exponent * torch.arange(0, half_dim).to(device=timesteps.device, dtype=torch.float32)
    exponent = exponent / half_dim

    timesteps_embedding = torch.exp(exponent)
    timesteps_embedding = timesteps[:, None] * timesteps_embedding[None, :]
    timesteps_embedding = torch.cat([torch.cos(timesteps_embedding), torch.sin(timesteps_embedding)], dim=-1)

    if self.timestep_embedding_channels % 2:
      padding = (0, 1, 0, 0) # L R U D
      timesteps_embedding = F.pad(timesteps_embedding, padding)

    return timesteps_embedding
  
class TimestepEmbedding(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.linear_1 = nn.Linear(in_channels, out_channels)
    self.act = nn.SiLU()
    self.linear_2 = nn.Linear(out_channels, out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear_1(x)
    x = self.act(x)
    x = self.linear_2(x)

    return x