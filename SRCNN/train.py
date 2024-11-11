"""
train.py

Script for training the SRCNN (Super-Resolution Convolutional Neural Network) model.
This script uses paired low-resolution and high-resolution images from the DIV2K dataset
to train the model through supervised learning, enhancing image resolution.
Note: Download the dataset from the DIV2K website (https://data.vision.ee.ethz.ch/cvl/DIV2K/)
      Need to download: Train Data Track 1 bicubic downscaling x4 (LR images)
                        Validation Data Track 1 bicubic downscaling x4 (LR images)
                        Train Data (HR images)

Dependencies:
- torch
- torchvision
- os
- models.srcnn (SRCNN model definition)
- utils.SuperResolutionDataset (custom dataset loader class)

Input:
- Low-resolution image path: "DIV2K/DIV2K_train_LR_bicubic/X4"
- High-resolution image path: "DIV2K/DIV2K_train_HR"

Output:
- Trained model weights file, saved by default to "trained_models/srcnn_div2k.pth"

Usage:
- Run this file directly to start model training:
    python train.py

Authors: [Chengyu Yang and Jiahua Zhao]
Date: [2024/11/10]
"""

import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
from utils import SuperResolutionDataset
import torchvision.transforms as transforms


# # Training function
def train_model(model, train_loader, num_epochs=20, learning_rate=1e-4, save_path="trained_models/srcnn_div2k.pth"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a save directory (if it does not exist)
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

    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == '__main__':

    # Configure the data loader and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    train_dataset = SuperResolutionDataset("DIV2K/DIV2K_train_LR_bicubic/X4", "DIV2K/DIV2K_train_HR",
                                           transform=transform, crop_size=256)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    model = SRCNN().to(device)
    train_model(model, train_loader)
