# Transformer详解（3-4周）

## 学习目标
掌握Transformer架构原理，理解Self-Attention机制，能够使用BERT和GPT进行NLP任务。

---

## 什么是Transformer？

**一句话**：Transformer是基于"注意力机制"的架构，能并行处理序列，是ChatGPT、BERT等大模型的基础。

**类比**：
```
RNN：像读书，一个字一个字读（串行）
Transformer：像拍照，一眼看到所有字（并行）

RNN：记忆力有限，前面的容易忘
Transformer：每个字都能直接看到所有字，没有遗忘问题
```

**为什么Transformer取代了RNN？**
| 特性 | RNN | Transformer |
|------|-----|-------------|
| 并行化 | 难（串行） | 容易（并行） |
| 长距离依赖 | 差（梯度消失） | 好（直接连接） |
| 训练速度 | 慢 | 快 |

---

## 第一周：Self-Attention机制

### 1.1 什么是注意力？

**直观理解**：注意力就是"关注重点"。

```
句子："小明在北京大学学习人工智能"

当翻译"学习"时，注意力分配：
- 小明：0.1
- 在：0.05
- 北京大学：0.15
- 学习：0.5  ← 最关注
- 人工智能：0.2
```

**类比**：
```
你读这句话时：
- 看到"学习"，会特别关注主语"小明"和宾语"人工智能"
- 不会平等地看每个字
```

### 1.2 Self-Attention计算

**核心思想**：每个词都和其他所有词计算相关性

```
输入：I love you
        ↓
计算相关性：
I-love：0.7
I-you：0.3
love-I：0.2
love-you：0.8
...
```

**三个向量**：
```
Query（查询）：我在找什么？
Key（键）：我能提供什么？
Value（值）：我的内容是什么？

类比：
- Query：你的搜索词
- Key：文章的标题/关键词
- Value：文章的内容
```

**计算公式**：
```
Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V

Q × Kᵀ：计算相似度
/ √dₖ：缩放（防止梯度消失）
softmax：归一化为概率
× V：加权求和
```

**图解**：
```
输入序列：[x₁, x₂, x₃]

        ┌─────────┐
        │    Q    │  Query
x₁ ────→│    K    │────→ 注意力分数 ────→ softmax ────→ 加权求和 ────→ 输出
        │    V    │  Value
        └─────────┘
            ↑
          权重矩阵
```

### 1.3 Python实现

```python
import numpy as np

def self_attention(Q, K, V):
    """Self-Attention计算"""
    d_k = K.shape[-1]
    
    # 计算注意力分数
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # softmax归一化
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # 加权求和
    output = np.dot(weights, V)
    
    return output, weights

# 示例
seq_len = 3
d_k = 4

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, weights = self_attention(Q, K, V)
print(f"注意力权重:\n{weights}")
print(f"输出形状: {output.shape}")
```

---

### 1.4 多头注意力

**为什么需要多头？**

**直观理解**：一个注意力只能关注一种关系，多个头可以关注多种关系。

```
句子："我在北京学习人工智能"

头1关注：语法关系（主语-谓语-宾语）
头2关注：位置关系（地点-动作）
头3关注：语义关系（学习-人工智能）
```

**计算方式**：
```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) × Wᴼ

headᵢ = Attention(Q × WᵢQ, K × WᵢK, V × WᵢV)
```

**PyTorch实现**：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 分头
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, weights
```

---

## 第二周：Transformer架构

### 2.1 整体结构

```
┌─────────────────────────────────────────┐
│              Transformer                │
│  ┌─────────────────────────────────┐   │
│  │           编码器                 │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │ Multi-Head Attention    │   │   │
│  │  └───────────┬─────────────┘   │   │
│  │              ↓                  │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │ Add & Layer Norm        │   │   │
│  │  └───────────┬─────────────┘   │   │
│  │              ↓                  │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │ Feed Forward Network    │   │   │
│  │  └───────────┬─────────────┘   │   │
│  │              ↓                  │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │ Add & Layer Norm        │   │   │
│  │  └─────────────────────────┘   │   │
│  └─────────────────────────────────┘   │
│                    ↓                    │
│  ┌─────────────────────────────────┐   │
│  │           解码器                 │   │
│  │  (类似结构，增加Masked Attention)│   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2.2 编码器

**组成部分**：
1. **Multi-Head Attention**：捕捉序列内部关系
2. **Add & Layer Norm**：残差连接 + 层归一化
3. **Feed Forward**：前馈神经网络

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差 + 归一化
        
        # Feed Forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 2.3 解码器

**与编码器的区别**：
1. **Masked Attention**：只能看到前面的词（防止看到未来）
2. **Cross Attention**：关注编码器输出

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked Self-Attention（只能看前面）
        attn1, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        
        # Cross Attention（关注编码器）
        attn2, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        
        # Feed Forward
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

### 2.4 位置编码

**问题**：Transformer没有位置信息（不像RNN有顺序）

**解决**：添加位置编码，告诉模型每个词的位置

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos：位置
i：维度
```

**类比**：
```
给每个座位贴标签：
座位1：[sin(1), cos(1), sin(1/100), cos(1/100), ...]
座位2：[sin(2), cos(2), sin(2/100), cos(2/100), ...]
```

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## 第三周：BERT与GPT

### 3.1 预训练模型概述

**什么是预训练？**
```
传统方法：每个任务从头训练
预训练：先在大量数据上学通用知识，再在具体任务上微调

类比：
传统：每门课都从第一页学
预训练：先学基础知识，再学专业课
```

**两大流派**：
| 模型 | 方向 | 任务 |
|------|------|------|
| BERT | 编码器 | 理解类任务 |
| GPT | 解码器 | 生成类任务 |

### 3.2 BERT

**全称**：Bidirectional Encoder Representations from Transformers

**核心思想**：双向理解上下文

```
句子：小明[MASK]在北京大学学习人工智能

BERT任务：预测[MMASK]是什么
答案：是
```

**预训练任务**：
1. **MLM**（Masked Language Model）：随机遮盖词，预测
2. **NSP**（Next Sentence Prediction）：判断两句话是否连续

**使用方式**：
```python
from transformers import BertTokenizer, BertModel

# 加载BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 编码
text = "我喜欢人工智能"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# 获取词向量
last_hidden_states = outputs.last_hidden_state
print(f"输出形状: {last_hidden_states.shape}")  # (1, seq_len, 768)
```

### 3.3 GPT

**全称**：Generative Pre-trained Transformer

**核心思想**：自回归生成

```
输入：今天天气
输出：今天天气真好，适合出去玩
```

**预训练任务**：
- **CLM**（Causal Language Model）：预测下一个词

```
输入：I love
输出：I love you

训练时：
I → love
I love → you
```

**使用方式**：
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 生成文本
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)

generated_text = tokenizer.decode(outputs[0])
print(generated_text)
```

### 3.4 BERT vs GPT

| 特性 | BERT | GPT |
|------|------|-----|
| 结构 | 编码器 | 解码器 |
| 方向 | 双向 | 单向（左→右） |
| 任务 | 理解 | 生成 |
| 应用 | 分类、NER、问答 | 文本生成、对话 |

---

## 第四周：微调与应用

### 4.1 微调（Fine-tuning）

**什么是微调？**
```
预训练模型：通用知识
微调：在特定任务上调整

类比：
预训练 = 大学生（通用知识）
微调 = 职业培训（专业技能）
```

**微调步骤**：
1. 加载预训练模型
2. 添加任务特定层
3. 在任务数据上训练

### 4.2 文本分类微调

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

# 加载模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 准备数据
texts = ["这个产品很好", "这个产品很差"]
labels = [1, 0]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(labels)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 4.3 命名实体识别（NER）

```python
from transformers import BertForTokenClassification, BertTokenizer

# 加载模型（假设4类实体）
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=4)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 输入
text = "小明在北京大学学习人工智能"
inputs = tokenizer(text, return_tensors='pt')

# 预测
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

# 解码
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = ['O', 'B-PER', 'B-LOC', 'B-ORG']  # 示例标签

print("实体识别结果：")
for token, pred in zip(tokens, predictions[0]):
    print(f"{token}: {labels[pred]}")
```

---

## 完整项目：情感分析

```python
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

# 自定义数据集
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

# 准备数据
train_texts = ["这部电影很好看", "太无聊了", "演员演技很棒", "剧情很烂"]
train_labels = [1, 0, 1, 0]

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 加载模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 预测
model.eval()
test_text = "这部作品非常出色"
inputs = tokenizer(test_text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

print(f"输入: {test_text}")
print(f"预测: {'正面' if prediction == 1 else '负面'}")
```

---

## 学习资源
- Attention Is All You Need：Transformer原始论文
- The Illustrated Transformer：可视化讲解
- Hugging Face教程：https://huggingface.co/learn

## 进入下一阶段
完成Transformer后，进入[[PyTorch实战]]。
