"""
train.py

用于训练SRCNN（Super-Resolution Convolutional Neural Network）模型的脚本。
该脚本使用DIV2K数据集的低分辨率和高分辨率图像对，通过监督学习方式训练模型，以提升图像的分辨率。

依赖项:
- torch
- torchvision
- os
- models.srcnn (SRCNN模型定义)
- utils.SuperResolutionDataset (自定义的数据集加载类)

输入:
- 低分辨率图像路径: "DIV2K/DIV2K_train_LR_bicubic/X4"
- 高分辨率图像路径: "DIV2K/DIV2K_train_HR"

输出:
- 训练好的模型权重文件, 默认保存路径为 "trained_models/srcnn_div2k.pth"

使用方式:
- 直接运行该文件以启动模型训练:
    python train.py

作者: [Chengyu Yang] and [Jiahua Zhao]
日期: [2024/11/10]
"""

import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
from utils import SuperResolutionDataset
import torchvision.transforms as transforms


# 训练函数
def train_model(model, train_loader, num_epochs=20, learning_rate=1e-4, save_path="trained_models/srcnn_div2k.pth"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建保存目录（若不存在）
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)

            optimizer.zero_grad()
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    print("Training complete")

    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == '__main__':
    # 配置数据加载器和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    train_dataset = SuperResolutionDataset("DIV2K/DIV2K_train_LR_bicubic/X4", "DIV2K/DIV2K_train_HR",
                                           transform=transform, crop_size=256)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    model = SRCNN().to(device)
    train_model(model, train_loader)
