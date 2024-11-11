"""
super_resolve.py

This script uses a trained SRCNN (Super-Resolution Convolutional Neural Network) model
to generate super-resolution images. It loads a low-resolution image and produces a
high-resolution output using the SRCNN model, supporting color image processing (RGB mode).

Dependencies:
- torch
- PIL (Python Imaging Library)
- models.srcnn (SRCNN model definition)
- utils.transforms (image transformation tools)

Input:
- Path to the low-resolution image: The file path to the input image
  (e.g., "DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png")

Output:
- Generated super-resolution image, saved by default as "output_high_res_image.png"

Usage:
- Run this file directly to generate a super-resolution image:
    python super_resolve.py

Author: [Chengyu Yang and Jiahua Zhao]
Date: [2024/11/10]
"""

import torch
from PIL import Image
from models.srcnn import SRCNN
from utils import transforms
import matplotlib.pyplot as plt


# Super resolution generation function
def super_resolve(model, image_path, device):
    model.eval()
    transform = transforms.ToTensor()
    # image = Image.open(image_path).convert("L")
    image = Image.open(image_path).convert("RGB")  # Change to RGB mode
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image).squeeze(0).cpu().clamp(0, 1)
    output_image = transforms.ToPILImage()(output)

    return output_image


# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
model.load_state_dict(torch.load("trained_models/srcnn_div2k.pth"))  # Load the trained model

# Generate super resolution images
output_image = super_resolve(model, "DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png", device)
output_image.show()  # Show images
output_image.save("output_high_res_image.png")  # Save images
