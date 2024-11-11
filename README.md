```
Team:	WedjatEye
Author:     ChengyuYang and Jiahua Zhao
Start Date: 30-Oct-2024
Commit Date: 10-Nov-2024
```
# Introduce 
    This project implements the SRCNN (Super-Resolution Convolutional Neural Network) model for image super-resolution. 
    The SRCNN model is a deep learning-based approach that enhances low-resolution images by generating high-resolution outputs. 
    This project is developed using PyTorch and utilizes the DIV2K dataset for training and evaluation.

# Need to modify
    The effect of super resolution is not ideal, and further modification of SRCNN model is needed.

# Project Structure
    SRCNN/
    ├── models/
    │   └── srcnn.py                # SRCNN model definition
    ├── DIV2K/                      # DIV2K dataset (not included in the repository)
    │   ├── DIV2K_train_HR/         # High-resolution training images
    │   ├── DIV2K_train_LR_bicubic/ # Low-resolution training images (bicubic downsampled)
            └── X4/                 # 4x downsampling
    │   └── DIV2K_valid_LR_bicubic/ # Low-resolution validation images (bicubic downsampled)
            └── X4/                 # 4x downsampling
    ├── train.py                    # Script to train the SRCNN model
    ├── super_resolve.py            # Script to generate super-resolution images with the trained model
    ├── utils.py                    # Utility functions and custom dataset class
    └── README.md                   # Project documentation

# Note 
     Download the dataset from the DIV2K website (https://data.vision.ee.ethz.ch/cvl/DIV2K/)
            Need to download:   Train Data Track 1 bicubic downscaling x4 (LR images)
                                Validation Data Track 1 bicubic downscaling x4 (LR images)
                                Train Data (HR images)

# DIV2K Dataset Structure
    The DIV2K dataset is used for training and evaluating image super-resolution models. This dataset includes high-resolution images and their corresponding low-resolution versions downsampled using bicubic interpolation. The expected folder structure for the dataset is as follows:
    DIV2K/
    ├── DIV2K_train_HR/                   # High-resolution training set
    │   ├── 0001.png
    │   ├── 0002.png
    │   ├── ...
    │   └── 0800.png                      # Total of 800 high-resolution training images
    ├── DIV2K_train_LR_bicubic/           # Low-resolution training set (generated with bicubic downsampling)
    │   └── X4/                           # 4x downsampling
    │       ├── 0001x4.png
    │       ├── 0002x4.png
    │       └── ...
    └── DIV2K_valid_LR_bicubic/           # Low-resolution validation set (generated with bicubic downsampling)
        └── X4/                           # 4x downsampling
            ├── 0801x4.png
            ├── 0802x4.png
            └── ...
# Usage
    Make sure the dataset is organized in this structure before running the training and evaluation scripts. The paths in the code should point to these directories to properly load the dataset.
    1. Training the SRCNN Model
    To train the SRCNN model on the DIV2K dataset, use train.py. Make sure the dataset is organized as shown above, and then run:
    python train.py
    This script will train the model on the low-resolution (4x downsampled) and high-resolution image pairs and save the trained model weights to trained_models/srcnn_div2k.pth by default.
    
    2. Generating Super-Resolution Images
    To generate a super-resolution image using the trained SRCNN model, use super_resolve.py. Update the path to the low-resolution input image in the script or use the provided one and run:
    python super_resolve.py
    This script will load the trained model and apply it to the specified low-resolution image. The output high-resolution image will be displayed and saved as output_high_res_image.png.