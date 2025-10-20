import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 7):
        super(UNetEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels * (2 ** (i - 1))
            output_ch = out_channels * (2 ** i)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(input_ch, output_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(output_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        self.encoder = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features
class UNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bottleneck(x)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 7):
        super(UNetDecoder, self).__init__()
        layers = []
        for i in range(num_layers - 1, -1, -1):
            input_ch = in_channels if i == num_layers - 1 else out_channels * (2 ** (i + 1))
            output_ch = out_channels * (2 ** i)
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(input_ch, output_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(output_ch),
                    nn.ReLU(inplace=True)
                )
            )
        self.decoder = nn.ModuleList(layers)

    def forward(self, x, encoder_features):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(encoder_features):
                x = torch.cat((x, encoder_features[-(i + 1)]), dim=1)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_layers=7):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, out_channels, num_layers)
        self.bottleneck = UNetBottleneck(out_channels * (2 ** (num_layers - 1)), out_channels * (2 ** num_layers))
        self.decoder = UNetDecoder(out_channels * (2 ** (num_layers - 1)), out_channels, num_layers)
        self.final_layer = nn.Conv2d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        bottleneck_output = self.bottleneck(encoder_features[-1])
        decoder_output = self.decoder(bottleneck_output, encoder_features)
        output = self.final_layer(decoder_output)
        return output
