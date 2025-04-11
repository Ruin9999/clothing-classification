import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from constants import *
from models import SimplifiedVisionTransformer, VisionTransformer, VisionTransformerDepth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
# model = SimplifiedVisionTransformer(
#     in_channels=1,
#     hidden_channels=HIDDEN_CHANNELS,
#     out_channels=10,
#     num_transformer_layers=NUM_TRANSFORMER_LAYERS,
#     num_heads=NUM_HEADS,
#     mlp_ratio=MLP_RATIO,
#     dropout=DROPOUT,
#     image_height=28,
#     image_width=28,
# ).to(device)

# model = VisionTransformer(
#     in_channels=1,
#     hidden_channels=HIDDEN_CHANNELS,
#     out_channels=10,
#     num_transformer_layers=NUM_TRANSFORMER_LAYERS + 3,
#     num_heads=NUM_HEADS,
#     mlp_ratio=MLP_RATIO,
#     dropout=DROPOUT,
#     patch_size=7,
#     image_height=28,
#     image_width=28,
# ).to(device)

model = VisionTransformerDepth(
    in_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=10,
    num_transformer_layers=NUM_TRANSFORMER_LAYERS + 3,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    patch_size=7,
    image_height=28,
    image_width=28,
).to(device)


pretrained_model_path = "safetensors/all/VisionTransformerDepth_0.8708.safetensor"
model.load_state_dict(load_file(pretrained_model_path))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inference
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")