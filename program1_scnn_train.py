# program1_scnn_train.py
# ---------------------------------------------------------------
# 该脚本使用 SpikingJelly 框架实现一个基于脉冲神经网络（SCNN）的
# Fashion-MNIST 分类器。模型结构类似 LeNet，采用 IF 神经元。
# ---------------------------------------------------------------

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

# 从 SpikingJelly 导入关键模块：
# neuron：神经元类型（如 IFNode）
# functional：用于重置网络状态
# layer：封装的层结构（如 Conv2d、Linear）
from spikingjelly.activation_based import neuron, functional, layer


# ---------------------- 超参数设置 ----------------------
LEARNING_RATE = 1e-3      # 学习率
BATCH_SIZE = 8192         # 批大小（过大可能导致显存不足）
EPOCHS = 100              # 训练轮数
T_TIMESTEPS = 8         # 时间步数（脉冲神经网络时间维长度）


# ---------------------- 定义模型结构 ----------------------
class SCNN(nn.Module):
    def __init__(self, T: int):
        super(SCNN, self).__init__()
        self.T = T  # 时间步数

        # 第一层卷积：输入通道1 → 输出通道6，卷积核大小5x5
        self.conv1 = layer.Conv2d(1, 6, 5)
        self.if1 = neuron.IFNode()          # 脉冲神经元层（Integrate-and-Fire）
        self.pool1 = layer.MaxPool2d(2, 2)  # 池化层，窗口2x2

        # 第二层卷积：输入通道6 → 输出通道16，卷积核大小5x5
        self.conv2 = layer.Conv2d(6, 16, 5)
        self.if2 = neuron.IFNode()
        self.pool2 = layer.MaxPool2d(2, 2)

        # 扁平化层，将卷积输出转为向量
        self.flatten = layer.Flatten()
        
        # 全连接层1：16*4*4 → 120
        self.fc1 = layer.Linear(16 * 4 * 4, 120)
        self.if3 = neuron.IFNode()

        # 全连接层2：120 → 84
        self.fc2 = layer.Linear(120, 84)
        self.if4 = neuron.IFNode()

        # 输出层：84 → 10（Fashion-MNIST 共10类）
        self.fc3 = layer.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        """
        前向传播：重复运行 T 次时间步，每次输入同一帧，
        最后对所有时间步的输出求平均。
        """
        outputs = []  # 用于存储每个时间步的输出
        for t in range(self.T):
            y = self.conv1(x)   # 卷积层1
            y = self.if1(y)     # IF神经元1
            y = self.pool1(y)   # 池化层1

            y = self.conv2(y)   # 卷积层2
            y = self.if2(y)     # IF神经元2
            y = self.pool2(y)   # 池化层2

            y = self.flatten(y) # 扁平化
            y = self.fc1(y)     # 全连接1
            y = self.if3(y)     # IF神经元3
            y = self.fc2(y)     # 全连接2
            y = self.if4(y)     # IF神经元4
            y = self.fc3(y)     # 输出层

            outputs.append(y)   # 存储当前时间步输出
        
        # 将所有时间步的输出堆叠并在时间维度上求平均
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(0)


# ---------------------- 数据集准备 ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录

# 数据增强与归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),              # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 均值、标准差归一化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载 Fashion-MNIST 数据集
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

trainset = torchvision.datasets.FashionMNIST(
    data_dir, download=True, train=True, transform=train_transform)
testset = torchvision.datasets.FashionMNIST(
    data_dir, download=True, train=False, transform=test_transform)

# 构建 DataLoader
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ---------------------- 模型与优化器设置 ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU或CPU
model = SCNN(T=T_TIMESTEPS).to(device)                   # 初始化模型

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)        # 余弦退火学习率调度器
criterion = nn.CrossEntropyLoss()                             # 交叉熵损失函数


# ---------------------- 训练循环 ----------------------
print("--- Starting SCNN Training (Tuned for Convergence) ---")
max_accuracy = 0.0  # 记录最佳准确率

for epoch in range(EPOCHS):
    # 设置为训练模式
    model.train()
    running_loss = 0.0

    # ---------- 训练阶段 ----------
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()           # 梯度清零
        functional.reset_net(model)     # 重置膜电位等状态，防止跨batch积累
        
        outputs = model(inputs)         # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                 # 反向传播
        optimizer.step()                # 参数更新

        running_loss += loss.item()     # 累积损失

    # 输出当前epoch的平均损失和学习率
    print(f'Epoch [{epoch + 1}/{EPOCHS}], '
          f'Loss: {running_loss / len(trainloader):.4f}, '
          f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    scheduler.step()  # 更新学习率


    # ---------- 测试阶段 ----------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            functional.reset_net(model)     # 重置膜电位
            outputs = model(images)         # 前向推理
            _, predicted = torch.max(outputs.data, 1)  # 取最大概率类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f} %')
    

    # ---------- 保存最优模型 ----------
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        print(f'New best accuracy: {max_accuracy:.2f} %. Saving model parameters...')
        output_dir = os.path.join(script_dir, "weights")
        os.makedirs(output_dir, exist_ok=True)

        # 将每一层参数单独保存为 .txt 文件
        for name, param in model.named_parameters():
            np.savetxt(os.path.join(output_dir, f'{name}.txt'),
                       param.detach().cpu().numpy().flatten())


# ---------------------- 训练结束 ----------------------
print('--- Finished Training ---')
print(f'Best accuracy achieved: {max_accuracy:.2f} %')
print("--- Final model parameters have been exported. ---")
