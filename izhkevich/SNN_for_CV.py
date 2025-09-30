import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
# LIF神经元模型
class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, tau=10.0, dt=1.0):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.tau = tau  # 膜时间常数
        self.dt = dt    # 时间步长
        self.decay = torch.exp(torch.tensor(-dt / tau))
        
    def forward(self, x, membrane_potential=None, spike=None):
        """
        x: 输入电流
        membrane_potential: 上一时刻的膜电位
        spike: 上一时刻的脉冲输出
        """
        batch_size, n_neurons = x.shape
        
        if membrane_potential is None:
            membrane_potential = torch.zeros(batch_size, n_neurons, device=x.device)
        if spike is None:
            spike = torch.zeros(batch_size, n_neurons, device=x.device)
        
        # LIF动力学方程
        membrane_potential = membrane_potential * self.decay + x * (1 - self.decay)
        
        # 发放脉冲
        spike = (membrane_potential >= self.threshold).float()
        
        # 重置机制
        membrane_potential = membrane_potential * (1 - spike)  # 发放后重置
        
        return spike, membrane_potential

# 脉冲编码层 - 将静态图像转换为脉冲序列
class PoissonEncoder(nn.Module):
    def __init__(self, timesteps=10):
        super(PoissonEncoder, self).__init__()
        self.timesteps = timesteps
        
    def forward(self, x):
        """
        x: 输入图像 [batch_size, channels, height, width]
        返回: 脉冲序列 [timesteps, batch_size, channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        
        # 生成随机数并与输入比较，模拟泊松过程
        spike_train = torch.rand(self.timesteps, batch_size, channels, height, width, device=x.device)
        spike_train = (spike_train < x.unsqueeze(0)).float()
        
        return spike_train

# LIF层 - 全连接版本
class LIFLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold=1.0, tau=10.0, dt=1.0):
        super(LIFLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.lif = LIFNeuron(threshold, tau, dt)
        self.output_size = output_size
    # 这种架构和用突触神经元表述的方式不一样。
    def forward(self, x, membrane_potential=None, spike=None):
        """
        x: 输入脉冲 [batch_size, input_size]
        """
        # 线性变换
        current = self.linear(x) # 前突触权重
        
        # LIF神经元
        spike, membrane_potential = self.lif(current, membrane_potential, spike)
        
        return spike, membrane_potential

# 卷积LIF层
class ConvLIFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, threshold=1.0, tau=10.0, dt=1.0):
        super(ConvLIFLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.lif = LIFNeuron(threshold, tau, dt)
        
    def forward(self, x, membrane_potential=None, spike=None):
        """
        x: 输入脉冲 [batch_size, channels, height, width]
        """
        # 卷积操作
        current = self.conv(x)
        
        # 重塑为LIF需要的形状
        batch_size, channels, height, width = current.shape
        current_flat = current.view(batch_size, -1)
        
        if membrane_potential is not None:
            membrane_potential = membrane_potential.view(batch_size, -1)
        if spike is not None:
            spike = spike.view(batch_size, -1)
        
        # LIF神经元
        spike_flat, membrane_potential_flat = self.lif(current_flat, membrane_potential, spike)
        
        # 重塑回原始形状
        spike = spike_flat.view(batch_size, channels, height, width)
        membrane_potential = membrane_potential_flat.view(batch_size, channels, height, width)
        
        return spike, membrane_potential

# 完整的LIF-SNN网络
class LIFSNN(nn.Module):
    def __init__(self, timesteps=10, threshold=1.0, tau=10.0):
        super(LIFSNN, self).__init__()
        self.timesteps = timesteps
        self.encoder = PoissonEncoder(timesteps)
        
        # 网络结构
        self.conv1 = ConvLIFLayer(3, 32, 3, threshold, tau)
        self.pool1 = nn.AvgPool2d(2)
        
        self.conv2 = ConvLIFLayer(32, 64, 3, threshold, tau)
        self.pool2 = nn.AvgPool2d(2)
        
        self.fc1 = LIFLayer(64 * 8 * 8, 128, threshold, tau)
        self.fc2 = nn.Linear(128, 10)  # 输出层不使用脉冲
        
        # 用于存储膜电位和脉冲
        self.membrane_potentials = []
        self.spikes = []
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 编码为脉冲序列
        spike_train = self.encoder(x)
        
        # 初始化膜电位和脉冲
        self.membrane_potentials = [None] * 4  # 4个层
        self.spikes = [None] * 4
        
        # 时间步循环
        total_spikes = torch.zeros(batch_size, 10, device=x.device)
        
        for t in range(self.timesteps):
            # 获取当前时间步的脉冲
            current_spike = spike_train[t]
            
            # 第一卷积层
            spike1, mem1 = self.conv1(current_spike, 
                                    self.membrane_potentials[0], 
                                    self.spikes[0])
            spike1_pooled = self.pool1(spike1)
            
            # 第二卷积层
            spike2, mem2 = self.conv2(spike1_pooled, 
                                    self.membrane_potentials[1], 
                                    self.spikes[1])
            spike2_pooled = self.pool2(spike2)
            
            # 全连接层
            spike2_flat = spike2_pooled.view(batch_size, -1)
            spike3, mem3 = self.fc1(spike2_flat, 
                                  self.membrane_potentials[2], 
                                  self.spikes[2])
            
            # 输出层（非脉冲）
            output = self.fc2(spike3)
            total_spikes += output
            
            # 存储状态用于下一时间步
            self.membrane_potentials = [mem1, mem2, mem3]
            self.spikes = [spike1, spike2, spike3]
        
        # 使用时间平均作为最终输出
        return total_spikes / self.timesteps

# 训练函数
def train_snn(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        """if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')"""
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

# 测试函数
def test_snn(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# 主函数
def main():
    # 超参数设置
    timesteps = 20
    batch_size = 320
    learning_rate = 1
    epochs = 10
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 路径设置
    output_path = 'SNN_for_CV'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = LIFSNN(timesteps=timesteps, threshold=0.5, tau=5.0).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    pbar = tqdm(range(epochs), desc="Processing")
    for epoch in pbar:
        #print(f'\nEpoch: {epoch+1}/{epochs}')
        
        # 训练
        train_loss, train_acc = train_snn(model, trainloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试
        test_acc = test_snn(model, testloader, device)
        test_accuracies.append(test_acc)
        pbar.set_description(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # 绘制结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,'results_.png'))

if __name__ == "__main__":
    main()