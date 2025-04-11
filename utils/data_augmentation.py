import torch
import random
import torchvision
import torch.nn.functional as F

from typing import Literal, Optional
from torch.utils.data import Dataset
from torchvision.transforms.v2 import RandAugment

class MixupDataset(Dataset):
  def __init__(self, dataset: Dataset, alpha: float = 0.4):
    self.dataset = dataset
    self.alpha = alpha

  def __len__(self):
    return len(self.dataset) #type: ignore

  def __getitem__(self, index: int):
    x1, y1 = self.dataset[index]
    x2, y2 = self.dataset[random.randint(0, len(self.dataset) - 1)] #type: ignore
    
    lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().item()
    mixed_x = lam * x1 + (1 - lam) * x2

    one_hot_y1 = F.one_hot(torch.tensor(y1), num_classes=10).float()
    one_hot_y2 = F.one_hot(torch.tensor(y2), num_classes=10).float()
    mixed_y = lam * one_hot_y1 + (1 - lam) * one_hot_y2

    return mixed_x, mixed_y

class DataAugmentor:
  @staticmethod
  def FashionMNIST(
    root: str,
    train: bool,
    transform: Optional[torchvision.transforms.Compose] = None,
    download: bool = False,
    subset: float = 1.0,
    augmentation: Literal["none", "mixup", "randaug", "all"] = "none",
  ):
    if augmentation in ["randaug", "all"]:
      randaug_transform = RandAugment(num_ops=2, magnitude=9)
      if transform is None:
        transform = torchvision.transforms.Compose([
          randaug_transform,
          torchvision.transforms.ToTensor(),
        ])
      else:
        transform.transforms.insert(0, randaug_transform)
    elif transform is None:
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=download, transform=transform)

    if subset < 1.0:
      total = len(dataset)
      subset_size = int(total * subset)
      indices = random.sample(range(total), subset_size)
      dataset = torch.utils.data.Subset(dataset, indices)

    if augmentation in ["mixup", "all"]:
      dataset = MixupDataset(dataset, alpha=2)

    return dataset