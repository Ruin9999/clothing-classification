import torch.nn as nn

# VGG Base Class
class VGG(nn.Module):
    def __init__(self, fc, num_classes):
        super(VGG, self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(fc, num_classes)
    
    # Standard Convolution
    def create_conv_layer(self, in_dim, out_dim, k=3, s=1, p=1):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        
    # Depthwise Separable Convolution
    def create_dws_layer(self, in_dim, out_dim, k=3, s=1, p=1):
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=k, stride=s, padding=p, groups=in_dim), # Depthwise Conv
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1), # Pointwise Conv
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        
    def create_fc_layer(self, in_dim, out_dim, drop=0.2):
        return nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
            )

# VGG-8 layers
class VGG8(VGG):
    def __init__(self, channels=[1, 32, 64, 128], fc=4096, num_classes=10):
        super().__init__(fc, num_classes)
        self.conv1 = self.create_conv_layer(channels[0], channels[1])
        self.conv2 = self.create_conv_layer(channels[1], channels[1])
        self.conv3 = self.create_conv_layer(channels[1], channels[2])
        self.conv4 = self.create_conv_layer(channels[2], channels[2])
        self.conv5 = self.create_conv_layer(channels[2], channels[-1])
        self.fc1 = self.create_fc_layer(7*7*channels[-1], fc)
        self.fc2 = self.create_fc_layer(fc, fc)
        
    def forward(self, x):
        out = self.conv1(x) # Layer 1
        out = self.conv2(out)   # Layer 2
        out = self.maxPool(out)
        out = self.conv3(out)   # Layer 3
        out = self.conv4(out)   # Layer 4
        out = self.maxPool(out)
        out = self.conv5(out)   # Layer 5
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out) # Layer 6
        out = self.fc2(out) # Layer 7
            
        return self.fc(out) # Layer 8

# VGG-8 layers
class VGG8_DWS(VGG):
    def __init__(self, channels=[1, 32, 64, 128], fc=4096, num_classes=10):
        super().__init__(fc, num_classes)
        self.conv1 = self.create_conv_layer(channels[0], channels[1])
        self.conv2 = self.create_dws_layer(channels[1], channels[1])
        self.conv3 = self.create_dws_layer(channels[1], channels[2])
        self.conv4 = self.create_dws_layer(channels[2], channels[2])
        self.conv5 = self.create_dws_layer(channels[2], channels[-1])
        self.fc1 = self.create_fc_layer(7*7*channels[-1], fc)
        self.fc2 = self.create_fc_layer(fc, fc)
        
    def forward(self, x):
        out = self.conv1(x) # Layer 1
        out = self.conv2(out)   # Layer 2
        out = self.maxPool(out)
        out = self.conv3(out)   # Layer 3
        out = self.conv4(out)   # Layer 4
        out = self.maxPool(out)
        out = self.conv5(out)   # Layer 5
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out) # Layer 6
        out = self.fc2(out) # Layer 7
            
        return self.fc(out) # Layer 8
        
# VGG-12 layers
class VGG12(VGG):
    def __init__(self, channels=[1, 8, 16, 32, 64, 128, 256, 512, 1024], fc=4096, num_classes=10):
        super().__init__(fc, num_classes)
        self.conv1 = self.create_conv_layer(channels[0], channels[1])
        self.conv2 = self.create_dws_layer(channels[1], channels[1])
        self.conv3 = self.create_dws_layer(channels[1], channels[2])
        self.conv4 = self.create_dws_layer(channels[2], channels[3])
        self.conv5 = self.create_dws_layer(channels[3], channels[4])
        self.conv6 = self.create_dws_layer(channels[4], channels[5])
        self.conv7 = self.create_dws_layer(channels[5], channels[6])
        self.conv8 = self.create_dws_layer(channels[6], channels[7])
        self.conv9 = self.create_dws_layer(channels[7], channels[-1])
        self.fc1 = self.create_fc_layer(7*7*channels[-1], fc)
        self.fc2 = self.create_fc_layer(fc, fc)
        
    def forward(self, x):
        out = self.conv1(x) # Layer 1
        out = self.conv2(out)   # Layer 2
        out = self.conv3(out)   # Layer 3
        out = self.maxPool(out)
        out = self.conv4(out)   # Layer 4
        out = self.conv5(out)   # Layer 5
        out = self.conv6(out)   # Layer 6
        out = self.maxPool(out)
        out = self.conv7(out)   # Layer 7
        out = self.conv8(out)   # Layer 8
        out = self.conv9(out)   # Layer 9
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out) # Layer 10
        out = self.fc2(out) # Layer 11
            
        return self.fc(out) # Layer 12
       