import os
from pickletools import optimize
import time
import json
import torch
import constants
import torch.nn as nn

from utils import EarlyStopper
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import Tuple, List, Optional
from safetensors.torch import save_file, load_file

class Trainer:
  def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, early_stopper: EarlyStopper, device: Optional[torch.device] = None, verbose: bool = True) -> None:
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.early_stopper = early_stopper
    self.device = device if device is not None else next(iter(model.parameters())).device
    self.verbose = verbose

  def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int = constants.NUM_EPOCHS) -> Tuple[List[float], List[float], List[float], List[float], float]:
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()
    if self.verbose: print("Starting training...")

    for epoch in range(1, num_epochs + 1):
      self.model.train()
      train_loss = 0.0
      train_correct = 0
      train_total = 0
      for images, labels in train_dataloader:
        images, labels = images.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        true_indices = torch.argmax(labels, dim=1) if labels.ndim == 2 and labels.shape[1] > 1 else labels

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == true_indices).sum().item()
        train_loss += loss.item()

      train_losses.append(train_loss / len(train_dataloader))
      train_accuracies.append(train_correct / train_total)
      if self.verbose: print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_correct / train_total:.4f}")

      self.model.eval()
      val_loss = 0.0
      val_correct = 0
      val_total = 0
      with torch.no_grad():
        for images, labels in val_dataloader:
          images, labels = images.to(self.device), labels.to(self.device)

          outputs = self.model(images)
          loss = self.criterion(outputs, labels)

          true_indices = torch.argmax(labels, dim=1) if labels.ndim == 2 and labels.shape[1] > 1 else labels
          _, predicted = torch.max(outputs, 1)
          val_total += labels.size(0)
          val_correct += (predicted == true_indices).sum().item()
          val_loss += loss.item()

      val_losses.append(val_loss / len(val_dataloader))
      val_accuracies.append(val_correct / val_total)
      if self.verbose: print(f"Epoch [{epoch}/{num_epochs}] - Val Loss: {val_loss:.4f}, Val Accuracy: {val_correct / val_total:.4f}")
      
      if self.early_stopper.early_stop(val_loss):
        if self.verbose: print("Early stopping...")
        break
    end_time = time.time()
    if self.verbose: print(f"Training completed in {end_time - start_time:.2f} seconds")

    return train_losses, val_losses, train_accuracies, val_accuracies, end_time - start_time
  
  def test(self, test_dataloader: DataLoader) -> float:
    correct = 0
    total = 0
    test_accuracy = 0.0
    self.model.eval()

    if self.verbose: print("Starting testing...")
    with torch.no_grad():
      for images, labels in test_dataloader:
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)

        true_indices = torch.argmax(labels, dim=1) if labels.ndim == 2 and labels.shape[1] > 1 else labels
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == true_indices).sum().item()
    test_accuracy = correct / total
    if self.verbose: print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

  def save(self, filePath: Optional[str] = "safetensors", fileName: Optional[str] = None) -> None:
    os.makedirs(filePath, exist_ok=True) #type: ignore

    # Prepare filename with timestamp and save model
    model_name = fileName if fileName is not None else type(self.model).__name__
    full_filepath = os.path.join(filePath, f"{model_name}.safetensor") #type: ignore
    save_file(self.model.state_dict(), full_filepath)
    print(f"Model saved to {full_filepath}")

    # Saving config file.
    config = {k: v for k, v in constants.__dict__.items() if k.isupper() and not k.startswith("__")}
    config_filename = os.path.join(filePath, f"{model_name}_config.json") #type: ignore
    with open(config_filename, "w") as f:
      json.dump(config, f, indent=4)

    print(f"Config file saved: {config_filename}")

  def load(self, filePath: str) -> None:
    if not os.path.exists(filePath):
      raise FileNotFoundError(f"File not found: {filePath}")

    state_dict = load_file(filePath)
    self.model.load_state_dict(state_dict)
    print(f"Model loaded from {filePath}")