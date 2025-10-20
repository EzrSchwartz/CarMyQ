import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5] # Return features at each level for skip connections
    
    
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use normal convolutions to reduce channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: upsampled features from previous layer
        x2: skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch if input wasn't perfectly divisible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Decoder path with skip connections"""
    def __init__(self, n_classes, bilinear=False):
        super().__init__()
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, features):
        """
        features: list of [x1, x2, x3, x4, x5] from encoder
        """
        x1, x2, x3, x4, x5 = features

        x = self.up1(x5, x4)  # Upsample from 1024 and concat with 512
        x = self.up2(x, x3)   # Upsample from 512 and concat with 256
        x = self.up3(x, x2)   # Upsample from 256 and concat with 128
        x = self.up4(x, x1)   # Upsample from 128 and concat with 64
        logits = self.outc(x)
        return logits


class UNet(nn.Module):
    """Complete U-Net architecture"""
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = UNetEncoder(n_channels)
        self.decoder = UNetDecoder(n_classes, bilinear)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits
    
    
model = UNet(n_channels=3, n_classes=10, bilinear=False)


for epoch in range(10):
    for input_tensor in dataset:
        output = model(input_tensor)  # Shape: (1, n_classes, 256, 256)


        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, targets)
        print(f"Cross-Entropy Loss: {loss.item()}")

