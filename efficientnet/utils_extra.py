"""
Modify: 2025.11.15
Author: SG.SUH
PyTorch: 1.8
Python: 3.8.5
"""

# Author: Zylo117

import math

from torch import nn
import torch.nn.functional as F


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, list) and len(stride) == 1:
            stride = stride * 2
        elif isinstance(stride, tuple) and len(stride) == 1:
            stride = (stride[0], stride[0])

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        pad_width = self.kernel_size[0] - self.stride[0]
        left = pad_width //2
        if pad_width //2 != pad_width /2:
            left += 1
        top = left
        right = left    #pad_width - left
        bottom = right

        self.pad_size = [left, right, top, bottom]

    def forward(self, x):
        x = F.pad(x, self.pad_size)

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        pad_width = (self.kernel_size - self.stride)
        self.pad_left = pad_width // 2
        self.pad_top = self.pad_left
        self.pad_right = pad_width - self.pad_left
        self.pad_bottom = self.pad_right

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        x = F.pad(x, [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom])

        x = self.pool(x)
        return x
