import torch
import torch.nn as nn
from torch.nn import functional as F

class VisionTransformer(nn.Module):
  def __init__(self):
    super().__init__()