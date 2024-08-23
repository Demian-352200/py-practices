import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 训练数据
transform = transforms.Compose(
    [transforms.ToTensor(),  # 转为tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

# 测试数据
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 卷积神经网络定义
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 2层卷积池化
        x = self.pool(F.relu(self.conv2(x)))  # 2层卷积池化
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet()
criterion = nn.CrossEntropyLoss()  # 损失函数定义
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 优化器定义

# 训练网络
for epoch in range(50):  # 50个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):  # 遍历训练集
        inputs, labels = data

        optimizer.zero_grad()  # 梯度清零

        outputs = model(inputs)  # 神经网络前向传播
        loss = criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失

    loss = running_loss / len(trainset)  # 打印Loss
    print(f'Epoch {epoch + 1}, Loss: {loss}')

print('Finished Training')
