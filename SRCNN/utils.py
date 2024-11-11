"""
utils.py

Defines the SuperResolutionDataset class for loading paired low-resolution and high-resolution images,
supporting RGB image reading and center cropping. This dataset class is typically used for image
super-resolution tasks, converting image pairs to tensors and returning them for the data loader.

Dependencies:
- os: For file and path management
- PIL (Python Imaging Library): For image loading and processing
- torch.utils.data.Dataset: PyTorch dataset base class
- torchvision.transforms: For image transformations

Inputs:
- low_res_dir: Directory path for low-resolution images
- high_res_dir: Directory path for high-resolution images
- transform: Optional image transformation (e.g., ToTensor)
- crop_size: Size for center cropping, default is 256

Usage:
- Initialize the SuperResolutionDataset class with paths for low-resolution and high-resolution images:
    dataset = SuperResolutionDataset("path/to/low_res_images", "path/to/high_res_images", transform=transforms.ToTensor(), crop_size=256)

- Use DataLoader to load the dataset:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

Authors: [Chengyu Yang and Jiahua Zhao]
Date: [2024/11/10]
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
        self.crop_size = crop_size  # Set the size of the crop
        self.low_res_images = sorted(os.listdir(low_res_dir))
        self.high_res_images = sorted(os.listdir(high_res_dir))

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_images[idx])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_images[idx])

        low_res_image = Image.open(low_res_path).convert("RGB")  # Use RGB
        high_res_image = Image.open(high_res_path).convert("RGB")  # Use RGB

        # Uniform cropping to the same size
        crop = transforms.CenterCrop(self.crop_size)
        low_res_image = crop(low_res_image)
        high_res_image = crop(high_res_image)

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image
