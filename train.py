import os
import time
import json
import torch
import constants
import torchvision
import torch.nn as nn
import torch.optim as optim

from constants import *
from torchvision import transforms
from safetensors.torch import save_file
from utils import EarlyStopper, DataAugmentor
from models import SimplifiedVisionTransformer, VisionTransformer, VisionTransformerDepth
from torch.utils.data import DataLoader, random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_index = 1

# Prepare dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train_dataset = DataAugmentor.FashionMNIST(root="./data", train=True, download=True, transform=transform, augmentation="all")
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
train_size = int(TRAIN_VAL_SPLIT * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Prepare model
# Create a interface layer that, depending on "train_index", will call either SimplifiedVisionTransformer, VisionTransformer, or DepthVisionTransformer
if train_index == 0:
    model = SimplifiedVisionTransformer(
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=10,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        image_height=28,
        image_width=28,
    ).to(device)
elif train_index == 1:
    model = VisionTransformer(
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=10,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        patch_size=7,
        image_height=28,
        image_width=28,
    ).to(device)
elif train_index == 2:
    model = VisionTransformerDepth(
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=10,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        patch_size=7,
        image_height=28,
        image_width=28,
    ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0)

train_losses = []
val_losses = []

# Train model
print("Starting training...")
start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss / len(train_dataloader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_losses.append(val_loss / len(val_dataloader))
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if early_stopper.early_stop(val_loss):
        print("Early stopping...")
        break
end_time = time.time()

# Test model
test_accuracy = 0
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Save model
print("\nTraining complete.")
print("Training Losses:", train_losses)
print("Validation Losses:", val_losses)
print("Test Accuracy:", test_accuracy)

save_dir = "safetensors/train_script"
os.makedirs(save_dir, exist_ok=True)

# Prepare filename with timestamp
model_name = type(model).__name__
filename = os.path.join(save_dir, f"{model_name}_{test_accuracy}.safetensor")
save_file(model.state_dict(), filename)
print(f"Model saved as safetensor: {filename}")

time_taken = end_time - start_time  # Fixed calculation (was reversed)
config = {k: v for k, v in constants.__dict__.items() if k.isupper() and not k.startswith("__")}
config["time_taken"] = time_taken  # Add time_taken to config dictionary
config_filename = os.path.join(save_dir, f"{model_name}_{test_accuracy}_config.json")
with open(config_filename, "w") as f:
    json.dump(config, f, indent=4)
print(f"Config file saved: {config_filename}")
print(f"Training time: {time_taken:.2f} seconds")