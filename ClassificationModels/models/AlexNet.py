import torch
import torch.nn as nn

from torchsummary import summary


class AlexNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_and_norm: bool = True) -> None:
        super(AlexNetBlock, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

        self.pool_and_norm = pool_and_norm
        if pool_and_norm:
            self.norm_layer = nn.LocalResponseNorm(
                size=5, alpha=0.0001, beta=0.75, k=2)
            self.pool_layer = nn.MaxPool2d(stride=2, kernel_size=3)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.relu(x)

        if self.pool_and_norm:
            x = self.norm_layer(x)
            x = self.pool_layer(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels) -> None:
        super(AlexNet, self).__init__()

        self.block1 = AlexNetBlock(
            in_channels, 96, 11, 4, 0, pool_and_norm=True)
        self.block2 = AlexNetBlock(96, 256, 5, 1, 2, pool_and_norm=True)
        self.block3 = AlexNetBlock(256, 384, 3, 1, 1, pool_and_norm=False)
        self.block4 = AlexNetBlock(384, 384, 3, 1, 1, pool_and_norm=False)
        self.block5 = AlexNetBlock(384, 256, 3, 1, 1, pool_and_norm=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.classification_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.classification_layer(x)
        return x


if __name__ == "__main__":
    net = AlexNet(num_classes=10, in_channels=3)
    summary(net, (3, 227, 227))
