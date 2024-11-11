"""
srcnn.py

实现了用于图像超分辨率任务的SRCNN（Super-Resolution Convolutional Neural Network）模型。
SRCNN是一种基础的卷积神经网络结构，通过提升低分辨率图像的清晰度，生成高分辨率图像。

依赖项:
- torch.nn: PyTorch中的神经网络模块

模型架构:
- 输入: RGB图像，形状为 [batch_size, 3, H, W]
- 输出: 超分辨率后的RGB图像，形状为 [batch_size, 3, H, W]

使用方式:
- 将模型导入后初始化并加载训练数据:
    model = SRCNN()
    output = model(input_image)

作者: [你的名字]
日期: [日期]
"""

import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)  # 输入通道从1改为 3
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)  # 输出通道改为 3
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
