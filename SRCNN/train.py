import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
from utils import SuperResolutionDataset
import torchvision.transforms as transforms


# 训练函数
def train_model(model, train_loader, num_epochs=20, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


# 配置数据加载器和模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()
train_dataset = SuperResolutionDataset("data/low_res_images", "data/high_res_images", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

model = SRCNN().to(device)
train_model(model, train_loader)
