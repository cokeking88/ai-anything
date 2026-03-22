# PyTorch实战（2-3周）

## 学习目标
掌握PyTorch框架核心概念，能够使用PyTorch构建、训练和部署深度学习模型。

---

## 什么是PyTorch？

**一句话**：PyTorch是Facebook开发的深度学习框架，以"动态图"和"Pythonic"著称。

**类比**：
```
TensorFlow 1.x：像C语言，先定义再运行（静态图）
PyTorch：像Python，边写边运行（动态图）

PyTorch更灵活，调试更方便，研究界更流行
```

---

## 第一周：张量与自动微分

### 1.1 张量（Tensor）

**什么是张量？**

**直观理解**：张量就是多维数组

```
标量：0维 → 5
向量：1维 → [1, 2, 3]
矩阵：2维 → [[1, 2], [3, 4]]
张量：n维 → 更高维数组
```

**创建张量**：
```python
import torch

# 从Python列表创建
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[1, 2], [3, 4]])

# 创建特殊张量
zeros = torch.zeros(3, 4)  # 全0
ones = torch.ones(3, 4)    # 全1
rand = torch.randn(3, 4)   # 随机正态分布

# 从NumPy创建
import numpy as np
arr = np.array([1, 2, 3])
t3 = torch.from_numpy(arr)

# 转回NumPy
arr2 = t3.numpy()
```

**张量操作**：
```python
# 基本运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)  # 加法
print(a * b)  # 乘法
print(a @ b)  # 点积

# 形状操作
t = torch.randn(2, 3, 4)
print(t.shape)           # torch.Size([2, 3, 4])
print(t.view(6, 2))      # 重塑
print(t.permute(2, 0, 1)) # 转置维度
```

---

### 1.2 自动微分（Autograd）

**什么是自动微分？**

**直观理解**：PyTorch自动帮你计算梯度

```
传统：手动推导梯度公式
PyTorch：设置 requires_grad=True，自动计算

类比：
传统：手算微积分
PyTorch：用计算器自动算
```

**基本用法**：
```python
import torch

# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 定义计算
y = x ** 2 + 3 * x + 1

# 反向传播，计算梯度
y.backward()

# 查看梯度
print(x.grad)  # dy/dx = 2x + 3 = 7
```

**计算图**：
```
x → x² → + → y
    ↘ 3x ↗

PyTorch自动构建计算图，反向传播时沿图计算梯度
```

**实际应用**：
```python
# 神经网络训练
optimizer.zero_grad()    # 清零梯度
loss = criterion(output, target)  # 计算损失
loss.backward()          # 反向传播（自动计算梯度）
optimizer.step()         # 更新参数
```

---

## 第二周：模型定义与训练

### 2.1 定义模型

**方式1：继承nn.Module**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel(10, 64, 2)
```

**方式2：使用nn.Sequential**
```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
```

### 2.2 损失函数与优化器

**常用损失函数**：
```python
# 回归
criterion = nn.MSELoss()

# 二分类
criterion = nn.BCEWithLogitsLoss()

# 多分类
criterion = nn.CrossEntropyLoss()
```

**常用优化器**：
```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam（最常用）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 2.3 完整训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 准备数据
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 2)
)

# 3. 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
for epoch in range(10):
    model.train()  # 训练模式
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 5. 评估
model.eval()  # 评估模式
with torch.no_grad():
    outputs = model(X)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
```

---

## 第三周：数据加载与GPU加速

### 3.1 自定义数据集

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = int(os.path.basename(image_path).split('_')[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### 3.2 数据增强

```python
import torchvision.transforms as transforms

# 图像数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.ColorJitter(brightness=0.2), # 颜色抖动
    transforms.ToTensor(),                  # 转为张量
    transforms.Normalize(                   # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 3.3 GPU加速

```python
import torch

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 将模型移到GPU
model = model.to(device)

# 训练时将数据移到GPU
for batch_X, batch_y in dataloader:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    
    outputs = model(batch_X)
    # ...
```

---

## 第四周：模型保存与部署

### 4.1 保存和加载模型

**方式1：保存整个模型**
```python
# 保存
torch.save(model, 'model.pth')

# 加载
model = torch.load('model.pth')
```

**方式2：只保存参数（推荐）**
```python
# 保存
torch.save(model.state_dict(), 'model_params.pth')

# 加载
model = MyModel(input_size=10, hidden_size=64, output_size=2)
model.load_state_dict(torch.load('model_params.pth'))
```

**保存训练状态**：
```python
# 保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 4.2 模型推理

```python
def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    return prediction, probabilities

# 使用
input_tensor = torch.randn(1, 10).to(device)
pred, probs = predict(model, input_tensor)
print(f"Prediction: {pred.item()}, Probabilities: {probs}")
```

---

## 完整项目：手写数字识别

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# 3. 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# 4. 测试
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = correct / len(test_dataset)
print(f'Test Accuracy: {accuracy:.4f}')

# 5. 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

---

## 学习资源
- PyTorch官方教程：https://pytorch.org/tutorials/
- PyTorch文档：https://pytorch.org/docs/stable/

## 进入下一阶段
完成深度学习后，进入[[阶段4_概述|大模型应用]]。
