from torch import nn
import numpy as np

# Block class for ResNet18
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.dwc1 = self.create_dws_layer(in_channels, out_channels, k=3, s=stride, p=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    # Depthwise Separable Convolution
    def create_dws_layer(self, in_dim, out_dim, k=3, s=1, p=1):
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=k, stride=s, padding=p, groups=in_dim), # Depthwise Conv
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1), # Pointwise Conv
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        identity = x
        x = self.dwc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
 
# ResNet18 
class ResNet18(nn.Module):
    def __init__(self, channels=[1, 64, 128, 256, 512], num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=7, stride=2, padding=3, dilation=1)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.create_block(channels[1], channels[1], stride=1)
        self.layer2 = self.create_block(channels[1], channels[2], stride=2)
        self.layer3 = self.create_block(channels[2], channels[3], stride=2)
        self.layer4 = self.create_block(channels[3], channels[-1], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def create_block(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )       
     
'''
Early Stopper
'''    
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False