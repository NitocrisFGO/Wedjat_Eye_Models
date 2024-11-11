"""
utils.py

定义超分辨率数据集加载类 SuperResolutionDataset，用于加载低分辨率和高分辨率图像对，支持RGB图像的读取和中心裁剪。
该数据集类通常用于图像超分辨率任务，将图像对转换为张量并返回给数据加载器。

依赖项:
- os: 文件和路径管理
- PIL (Python Imaging Library): 图像加载和处理
- torch.utils.data.Dataset: PyTorch数据集基类
- torchvision.transforms: 图像转换工具

输入:
- low_res_dir: 低分辨率图像的文件夹路径
- high_res_dir: 高分辨率图像的文件夹路径
- transform: 可选的图像转换（例如 ToTensor）
- crop_size: 图像裁剪大小，默认为 256

使用方式:
- 初始化 SuperResolutionDataset 类，并传入低分辨率和高分辨率图像文件夹路径：
    dataset = SuperResolutionDataset("path/to/low_res_images", "path/to/high_res_images", transform=transforms.ToTensor(), crop_size=256)

- 使用 DataLoader 加载数据集：
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

作者: [你的名字]
日期: [日期]
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SuperResolutionDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None, crop_size=256):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.crop_size = crop_size  # 设置裁剪的尺寸
        self.low_res_images = sorted(os.listdir(low_res_dir))
        self.high_res_images = sorted(os.listdir(high_res_dir))

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_images[idx])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_images[idx])

        low_res_image = Image.open(low_res_path).convert("RGB")  # 使用 RGB 模式
        high_res_image = Image.open(high_res_path).convert("RGB")  # 使用 RGB 模式

        # 统一裁剪为相同大小
        crop = transforms.CenterCrop(self.crop_size)
        low_res_image = crop(low_res_image)
        high_res_image = crop(high_res_image)

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image
