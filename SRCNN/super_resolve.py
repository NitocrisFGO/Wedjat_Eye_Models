"""
super_resolve.py

用于使用训练好的SRCNN（Super-Resolution Convolutional Neural Network）模型生成超分辨率图像。
该脚本加载低分辨率图像并通过SRCNN模型生成高分辨率输出，支持彩色图像处理（RGB模式）。

依赖项:
- torch
- PIL (Python Imaging Library)
- models.srcnn (SRCNN模型定义)
- utils.transforms (图像转换工具)

输入:
- 低分辨率图像路径: 需要传入的图像文件路径（例如 "DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png"）

输出:
- 生成的超分辨率图像，默认保存路径为 "output_high_res_image.png"

使用方式:
- 直接运行该文件以生成超分辨率图像:
    python super_resolve.py

作者: [你的名字]
日期: [日期]
"""

import torch
from PIL import Image
from models.srcnn import SRCNN
from utils import transforms
import matplotlib.pyplot as plt


# 超分辨率生成函数
def super_resolve(model, image_path, device):
    model.eval()
    transform = transforms.ToTensor()
    # image = Image.open(image_path).convert("L")
    image = Image.open(image_path).convert("RGB")  # 修改为 RGB 模式
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image).squeeze(0).cpu().clamp(0, 1)
    output_image = transforms.ToPILImage()(output)

    return output_image


# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
model.load_state_dict(torch.load("trained_models/srcnn_div2k.pth"))  # 加载训练完成的模型

# 生成超分辨率图像
output_image = super_resolve(model, "DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png", device)
output_image.show()  # 展示图像
output_image.save("output_high_res_image.png")  # 保存图像
