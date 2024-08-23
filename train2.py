import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random

# 定义类别数
num_classes = 10  # CIFAR-10 数据集有 10 个类别

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(root='D:\pytorch\pytorch\datasets\cifar10', train=True, download=True, transform=transform_train)
val_dataset = torchvision.datasets.CIFAR10(root='D:\pytorch\pytorch\datasets\cifar10', train=False, download=True, transform=transform_val)

# 划分数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 使用预训练模型
def create_model(weights=None):
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # 使用定义的 num_classes
    return model

# 主模块
if __name__ == '__main__':
    # 优化器和学习率调度
    optimizer = optim.Adam(create_model().parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)

    num_epochs = 50
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    print("Training complete.")
