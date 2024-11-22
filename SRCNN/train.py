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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
from utils import SuperResolutionDataset
import torchvision.transforms as transforms


# # Training function
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, save_path="trained_models/srcnn_div2k.pth"):
    # 使用感知损失

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dynamic learning rate adjustment
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
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

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss / len(train_loader):.4f}")

        # Verifying model performance
        val_loss = validate_model(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model improved and saved to {save_path}")

    print("Training complete")

    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
            val_loss += loss.item()

    return val_loss / len(val_loader)


if __name__ == '__main__':

    # Configure the data loader and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    train_dataset = SuperResolutionDataset("DIV2K/DIV2K_train_LR_bicubic/X4",
                                           "DIV2K/DIV2K_train_HR",
                                           transform=transform,
                                           crop_size=128
                                           )
    val_dataset = SuperResolutionDataset(
        "DIV2K/DIV2K_valid_LR_bicubic/X4",
        "DIV2K/DIV2K_valid_HR",
        transform=transform,
        crop_size=128
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = SRCNN().to(device)
    train_model(model, train_loader, val_loader)
