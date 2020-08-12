
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_block(in_channels: int, out_channels: int, kernel_size=3) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    size -> size - (kernel_size + 1) / 2 (if padding is 0)
    size -> size / 2 (because padding is 1)
    """


    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )