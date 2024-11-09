import torch
from PIL import Image
from models.srcnn import SRCNN
from utils import transforms
import matplotlib.pyplot as plt


# 超分辨率生成函数
def super_resolve(model, image_path, device):
    model.eval()
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("L")
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image).squeeze(0).cpu().clamp(0, 1)
    output_image = transforms.ToPILImage()(output)

    return output_image


# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
model.load_state_dict(torch.load("path_to_saved_model.pth"))  # 加载训练完成的模型

# 生成超分辨率图像
output_image = super_resolve(model, "path/to/low_res_image.png", device)
output_image.show()  # 展示图像
output_image.save("output_high_res_image.png")  # 保存图像
