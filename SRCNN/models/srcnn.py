"""
srcnn.py

This file implements the SRCNN (Super-Resolution Convolutional Neural Network) model
for image super-resolution tasks. SRCNN is a basic convolutional neural network structure
designed to enhance the clarity of low-resolution images, producing high-resolution outputs.

Dependencies:
- torch.nn: PyTorch neural network module

Model Architecture:
- Input: RGB image with shape [batch_size, 3, H, W]
- Output: Super-resolved RGB image with shape [batch_size, 3, H, W]

Usage:
- Import and initialize the model, then pass the input tensor:
    model = SRCNN()
    output = model(input_image)

Author: [Chengyu Yang and Jiahua Zhao]
Date: [2024/11/10]
"""

import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        # Input channel changed from 1 to 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)

        # Output channel changed from 1 to 3
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
