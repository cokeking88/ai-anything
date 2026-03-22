# RNN序列建模（2-3周）

## 学习目标
掌握循环神经网络原理，理解LSTM和GRU，能够构建序列到序列模型。

---

## 什么是RNN？

**一句话**：RNN是有"记忆"的神经网络，能处理序列数据。

**类比**：
```
普通神经网络：每次看一张图，独立处理
RNN：看书时，记住前面的内容，理解当前句子

普通网络：健忘症患者
RNN：有记忆力的人
```

**应用场景**：
- 文本生成
- 机器翻译
- 语音识别
- 时间序列预测

---

## 第一周：RNN基础

### 1.1 RNN结构

**直观理解**：RNN把上一步的输出作为下一步的输入。

```
时间步1      时间步2      时间步3
  x₁    →    x₂    →    x₃
  ↓          ↓          ↓
[h₁]   →   [h₂]   →   [h₃]
  ↓          ↓          ↓
  y₁         y₂         y₃

每个h都依赖于当前输入x和上一个隐藏状态h
```

**数学公式**：
```
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
yₜ = Wₕᵧhₜ + bᵧ

hₜ：当前隐藏状态
xₜ：当前输入
hₜ₋₁：上一步隐藏状态
```

**类比**：
- hₜ = 你的当前记忆（结合新信息和旧记忆）
- xₜ = 你刚看到的新信息
- hₜ₋₁ = 你之前的记忆

### 1.2 Python实现

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # 初始化权重
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)
    
    def forward(self, inputs):
        """前向传播"""
        h = np.zeros(self.hidden_size)  # 初始隐藏状态
        outputs = []
        
        for x in inputs:
            # 更新隐藏状态
            h = np.tanh(np.dot(x, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            # 计算输出
            y = np.dot(h, self.Why) + self.by
            outputs.append(y)
        
        return outputs, h

# 示例
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
inputs = [np.random.randn(10) for _ in range(5)]  # 5个时间步
outputs, final_h = rnn.forward(inputs)
print(f"输出数量: {len(outputs)}, 最终隐藏状态形状: {final_h.shape}")
```

---

### 1.3 RNN的问题：梯度消失

**问题**：RNN记不住长期依赖

```
"我在中国出生，...（100个词）...，我会说中文"

RNN的问题：
- "中国"离"中文"太远
- 梯度在传播过程中消失
- 记不住前面的信息
```

**类比**：
```
传话游戏：
第1人：我在中国出生
第10人：我...什么来着？
第50人：？？？
```

**解决方案**：LSTM和GRU

---

## 第二周：LSTM

### 2.1 什么是LSTM？

**一句话**：LSTM = 有"门控机制"的RNN，能选择性地记住或忘记信息。

**类比**：
```
普通RNN：大脑直连，所有信息都记住（记不住长期）

LSTM：有选择的记忆系统
- 忘记门：决定忘掉什么旧信息
- 输入门：决定记住什么新信息
- 输出门：决定输出什么信息
```

### 2.2 LSTM结构

```
        ┌─────────────────────────────────────┐
        │                                     │
        ↓                                     │
    ┌───────┐    ┌───────┐    ┌───────┐      │
xₜ →│忘记门f │  → │输入门i │  → │输出门o │ → hₜ
    └───────┘    └───────┘    └───────┘      │
        │             │           │          │
        ↓             ↓           ↓          │
    忘记旧记忆    选择新信息    输出信息    ┌──┴──┐
        │             │           │      │细胞状态│
        └──────┬──────┘           │      │  Cₜ  │
               ↓                  │      └──────┘
            更新记忆 ─────────────┴─────────┘
```

**三个门**：
```
忘记门：fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)  → 决定忘掉什么
输入门：iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)  → 决定记住什么
输出门：oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)  → 决定输出什么
```

**细胞状态更新**：
```
候选记忆：C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
新记忆：Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ
输出：hₜ = oₜ × tanh(Cₜ)
```

### 2.3 LSTM为什么有效？

**关键**：细胞状态Cₜ像一条"传送带"

```
Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ

- 如果fₜ=1，iₜ=0：完全保留旧记忆
- 如果fₜ=0，iₜ=1：完全更新为新记忆
- 梯度可以无损地沿着Cₜ传播（加法路径）
```

**类比**：
```
普通RNN：每次都重新学习
LSTM：
- 遇到"句号"：忘记门打开，清空记忆
- 遇到重要词：输入门打开，记住
- 需要输出时：输出门打开，输出相关信息
```

### 2.4 PyTorch实现

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 门的权重（合并计算）
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        # 拼接输入和上一隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        
        # 计算四个门
        gates = self.W(combined)
        f, i, o, g = gates.chunk(4, dim=1)
        
        f = torch.sigmoid(f)  # 忘记门
        i = torch.sigmoid(i)  # 输入门
        o = torch.sigmoid(o)  # 输出门
        g = torch.tanh(g)     # 候选记忆
        
        # 更新细胞状态
        c = f * c_prev + i * g
        
        # 计算隐藏状态
        h = o * torch.tanh(c)
        
        return h, c

# 使用PyTorch内置LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# 输入：(batch, seq_len, input_size)
x = torch.randn(32, 10, 10)  # 32个样本，10个时间步，10维输入
output, (h_n, c_n) = lstm(x)

print(f"输出形状: {output.shape}")  # (32, 10, 20)
print(f"最终隐藏状态形状: {h_n.shape}")  # (2, 32, 20)
print(f"最终细胞状态形状: {c_n.shape}")  # (2, 32, 20)
```

---

## 第三周：GRU与序列到序列

### 3.1 GRU

**一句话**：GRU是LSTM的简化版，只有两个门。

```
LSTM：忘记门 + 输入门 + 输出门 + 细胞状态
GRU：重置门 + 更新门（更简单）
```

**GRU结构**：
```
重置门：rₜ = σ(Wr·[hₜ₋₁, xₜ])
更新门：zₜ = σ(Wz·[hₜ₋₁, xₜ])
候选隐藏：h̃ₜ = tanh(W·[rₜ × hₜ₋₁, xₜ])
新隐藏：hₜ = (1-zₜ) × hₜ₋₁ + zₜ × h̃ₜ
```

**LSTM vs GRU**：

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 参数量 | 多 | 少 |
| 训练速度 | 慢 | 快 |
| 效果 | 通常更好 | 相当 |

**PyTorch实现**：
```python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(32, 10, 10)
output, h_n = gru(x)  # GRU没有细胞状态
```

---

### 3.2 序列到序列（Seq2Seq）

**任务**：将一个序列转换为另一个序列

```
输入序列：I love you → 编码器 → 向量 → 解码器 → 输出序列：我爱你
```

**结构**：
```
编码器（Encoder）：          解码器（Decoder）：
  x₁ → x₂ → x₃              → y₁ → y₂ → y₃
  ↓    ↓    ↓                ↓    ↓    ↓
 [h₁] [h₂] [h₃] → 向量 → [s₁] [s₂] [s₃]

编码器：压缩输入为向量
解码器：从向量生成输出
```

**PyTorch实现**：
```python
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, target_len):
        # 编码
        _, (h, c) = self.encoder(x)
        
        # 解码
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.fc.out_features).to(x.device)
        outputs = []
        
        for _ in range(target_len):
            output, (h, c) = self.decoder(decoder_input, (h, c))
            output = self.fc(output)
            outputs.append(output)
            decoder_input = output  # 使用预测作为下一步输入
        
        return torch.cat(outputs, dim=1)
```

---

### 3.3 注意力机制

**问题**：Seq2Seq把整个输入压缩成一个向量，信息损失

**解决**：注意力机制让解码器"关注"输入的不同部分

```
无注意力：输入 → [向量] → 输出
有注意力：输入 → [多个向量] → 加权组合 → 输出

类比：
无注意力：读完一本书，记住大概
有注意力：读完一本书，记住重点段落
```

**注意力计算**：
```
注意力分数：score(sₜ, hᵢ) = sₜᵀ × hᵢ
注意力权重：αᵢ = softmax(score)
上下文向量：cₜ = Σ αᵢ × hᵢ
```

**PyTorch实现**：
```python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        
        # 计算注意力分数
        scores = self.v(torch.tanh(self.W(encoder_outputs)))
        scores = scores.squeeze(-1)  # (batch, seq_len)
        
        # 计算注意力权重
        weights = torch.softmax(scores, dim=1)
        
        # 加权求和
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, weights
```

---

## 完整项目：文本情感分析

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embeds = self.dropout(self.embedding(x))  # (batch, seq_len, embed_size)
        lstm_out, (h_n, _) = self.lstm(embeds)    # h_n: (1, batch, hidden)
        output = self.fc(h_n.squeeze(0))           # (batch, output_size)
        return output

# 模拟数据
vocab_size = 10000
embed_size = 128
hidden_size = 256
output_size = 2  # 正面/负面

model = SentimentLSTM(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环（简化）
for epoch in range(10):
    # 模拟一批数据
    x = torch.randint(0, vocab_size, (32, 50))  # 32个样本，长度50
    y = torch.randint(0, 2, (32,))
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 学习资源
- Understanding LSTM Networks：经典LSTM解释文章
- CS224n：斯坦福NLP课程

## 进入下一阶段
完成RNN后，进入[[Transformer详解]]。
