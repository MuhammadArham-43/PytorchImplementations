import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from torchsummary import summary


class DoubleConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super(DoubleConvolution, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # For Same convolutional. Output Size equals input size
            bias=False
        )
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.batchNorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.activation(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        features: list = [64, 128, 256, 512],
    ) -> None:

        super(UNet, self).__init__()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Create Downsampling part
        for feature in features:
            self.down_convs.append(
                DoubleConvolution(in_channels, feature)
            )
            in_channels = feature

        # Create Upsampling part
        for feature in reversed(features):
            self.up_convs.append(
                nn.ModuleDict({
                    'convTranspose':
                        nn.ConvTranspose2d(
                            in_channels=feature*2,
                            out_channels=feature,
                            kernel_size=2,
                            stride=2
                        ),
                    'doubleConv': DoubleConvolution(feature*2, feature, )
                })
            )

        self.bottleneck = nn.Conv2d(
            features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1)

        self.output_layer = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        for down in self.down_convs:
            x = down(x)
            skip_connections.insert(0, x)
            x = self.pooling_layer(x)

        # print(x.shape)
        x = self.bottleneck(x)
        # print(x.shape)
        for idx, module in enumerate(self.up_convs):
            up_conv = module['convTranspose']
            double_conv = module['doubleConv']

            x = up_conv(x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = resize(x, size=(skip_connection.shape[2:]))
            # Dim 1 across Channels (Batch, Channels, Height, Width)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = double_conv(concat_skip)
        # print(x.shape)
        x = self.output_layer(x)
        # print(x.shape)
        return x


if __name__ == "__main__":
    # print(summary(DoubleConvolution(3, 64), (3, 512, 512)))
    # print(summary(UNet(3, 1), (3, 511, 511)))
    input_tensor = torch.randn((16, 3, 160, 160)).to('cuda')
    unet = UNet(3, 1).to('cuda')
    output = unet(input_tensor)
    print(output.shape)
