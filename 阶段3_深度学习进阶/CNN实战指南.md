# CNN实战指南（3-4周）

## 学习目标
掌握卷积神经网络原理，理解经典CNN架构，能够构建图像分类、目标检测模型。

---

## 什么是CNN？

**一句话**：CNN是专门为处理图像设计的神经网络，能自动提取图像特征。

**类比**：
```
人眼看图片：
1. 先看边缘（线条）
2. 再看纹理（形状）
3. 最后识别物体

CNN看图片：
1. 卷积层提取边缘
2. 卷积层提取形状
3. 全连接层识别物体
```

---

## 第一周：卷积运算

### 1.1 什么是卷积？

**直观理解**：用一个小窗口在图片上滑动，计算加权和。

```
输入图片（5×5）：          卷积核（3×3）：         输出特征图（3×3）：
1  1  1  0  0              1  0  1                  4  3  4
0  1  1  1  0       ×      0  1  0        =        2  4  3
0  0  1  1  1              1  0  1                  0  2  4
0  0  1  1  0
0  1  1  0  0

卷积核在图片上滑动，每次计算对应位置的乘积和
```

**类比**：
- 卷积核 = 放大镜，用来检测特定特征
- 不同的卷积核检测不同的特征（边缘、角点、纹理等）

### 1.2 卷积的参数

| 参数 | 含义 | 影响 |
|------|------|------|
| 卷积核大小 | 窗口大小 | 感受野大小 |
| 步长 | 滑动步长 | 输出大小 |
| 填充 | 边缘补0 | 保持大小 |
| 通道数 | 卷积核数量 | 特征数量 |

**输出大小公式**：
```
输出大小 = (输入大小 - 卷积核大小 + 2×填充) / 步长 + 1
```

### 1.3 Python实现

```python
import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    """简单的2D卷积"""
    # 添加填充
    if padding > 0:
        input = np.pad(input, padding, mode='constant')
    
    h, w = input.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = input[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

# 示例
input = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

output = conv2d(input, kernel)
print("卷积输出：")
print(output)
```

---

### 1.4 池化层

#### 什么是池化？

**直观理解**：压缩图片，保留重要信息。

```
最大池化（2×2）：
1  3  2  4         3  4
     ↓       →     
5  2  1  0         5  2

取每个区域的最大值
```

**作用**：
- 减少计算量
- 提取主要特征
- 增加平移不变性

**类型**：
- 最大池化：取最大值（最常用）
- 平均池化：取平均值

#### Python实现
```python
def max_pool2d(input, pool_size=2, stride=2):
    """最大池化"""
    h, w = input.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output
```

---

## 第二周：经典CNN架构

### 2.1 LeNet-5（1998）

**特点**：最早的CNN，用于手写数字识别

```
输入(32×32) → Conv(6@5×5) → Pool → Conv(16@5×5) → Pool → FC(120) → FC(84) → Output(10)
```

### 2.2 AlexNet（2012）

**特点**：深度学习复兴的标志，ImageNet冠军

```
输入(224×224) → Conv(96@11×11) → Pool → Conv(256@5×5) → Pool → 
Conv(384@3×3) → Conv(384@3×3) → Conv(256@3×3) → Pool → 
FC(4096) → FC(4096) → Output(1000)
```

**创新**：
- 使用ReLU激活函数
- 使用Dropout
- 数据增强

### 2.3 VGG（2014）

**特点**：使用小卷积核（3×3），更深的网络

```
VGG-16：
Conv(64@3×3) × 2 → Pool → 
Conv(128@3×3) × 2 → Pool → 
Conv(256@3×3) × 3 → Pool → 
Conv(512@3×3) × 3 → Pool → 
Conv(512@3×3) × 3 → Pool → 
FC(4096) → FC(4096) → Output(1000)
```

**核心思想**：
- 两个3×3卷积 = 一个5×5卷积的感受野
- 但参数更少，更深

### 2.4 ResNet（2015）

**特点**：残差连接，解决了深层网络训练困难的问题

```
残差块：
输入 → [Conv → BN → ReLU → Conv → BN] → + → ReLU → 输出
  │                                          ↑
  └──────────────────────────────────────────┘
           跳跃连接（Shortcut Connection）
```

**为什么有效？**
```
没有残差：H(x) = F(x)
有残差：H(x) = F(x) + x

学习F(x) = H(x) - x（残差）比直接学习H(x)更容易
```

**PyTorch实现**：
```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果维度不同，需要调整shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out
```

---

### 2.5 CNN架构对比

| 架构 | 年份 | 深度 | Top-5错误率 | 特点 |
|------|------|------|-------------|------|
| LeNet | 1998 | 5层 | - | 首个CNN |
| AlexNet | 2012 | 8层 | 15.3% | 深度学习复兴 |
| VGG | 2014 | 16-19层 | 7.3% | 小卷积核 |
| ResNet | 2015 | 152层 | 3.57% | 残差连接 |

---

## 第三周：目标检测

### 3.1 什么是目标检测？

**任务**：找出图片中所有物体的位置和类别

```
输入图片 → 输出：
- 物体1：位置(x1,y1,x2,y2)，类别=猫
- 物体2：位置(x1,y1,x2,y2)，类别=狗
```

### 3.2 目标检测方法

**两阶段方法（先找区域，再分类）**：
- R-CNN → Fast R-CNN → Faster R-CNN
- 精度高，速度慢

**单阶段方法（直接预测）**：
- YOLO、SSD
- 速度快，精度稍低

### 3.3 YOLO

**核心思想**：把图片分成网格，每个网格预测边界框

```
图片分成S×S网格：
┌─┬─┬─┬─┐
│ │ │ │ │  每个格子预测：
│ │●│ │ │  - 是否有物体
├─┼─┼─┼─┤  - 边界框位置
│ │ │ │ │  - 类别概率
│ │ │●│ │
└─┴─┴─┴─┘
```

**PyTorch实现（简化版）**：
```python
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_boxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        # 每个格子预测：num_boxes * (5 + num_classes)
        # 5 = x, y, w, h, confidence
        self.conv = nn.Conv2d(in_channels, num_boxes * (5 + num_classes), 1)
    
    def forward(self, x):
        return self.conv(x)
```

---

## 第四周：图像分割

### 4.1 什么是图像分割？

**任务**：给图片中每个像素分类

```
语义分割：          实例分割：
所有猫都标为"猫"    每只猫单独标记
```

### 4.2 U-Net

**特点**：编码器-解码器结构，跳跃连接

```
编码器（下采样）          解码器（上采样）
○○○○○                   ○○○○○
 ↓池化                    ↑上采样
○○○○                     ○○○○
 ↓池化                    ↑上采样
○○○                      ○○○
 ↓池化                    ↑上采样
○○         ←跳跃连接→     ○○
```

**PyTorch实现**：
```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 编码器
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # 瓶颈
        self.bottleneck = self.conv_block(512, 1024)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self.conv_block(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # 瓶颈
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        # 解码 + 跳跃连接
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        return self.out(d1)
```

---

## 完整项目：图像分类

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 数据准备
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# 2. 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# 4. 评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试准确率: {100 * correct / total:.2f}%")
```

---

## 学习资源
- CS231n：斯坦福计算机视觉课程
- PyTorch官方教程：https://pytorch.org/tutorials/

## 进入下一阶段
完成CNN后，进入[[RNN序列建模]]。
