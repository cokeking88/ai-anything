# LLM基础原理（4-5周）

## 学习目标
理解大语言模型的工作原理，掌握LLM的核心概念和发展历史。

---

## 什么是大语言模型？

**一句话**：大语言模型（LLM）是通过阅读海量文本学会"说话"的AI。

**类比**：
```
传统NLP：教小孩做数学题（规则明确）
LLM：教小孩读万卷书后写文章（从数据中学习）

ChatGPT = 读了互联网上几乎所有文字后学会的"超级学生"
```

---

## 第一周：LLM发展历史

### 1.1 发展历程

```
2017：Transformer提出（注意力机制）
  ↓
2018：BERT（双向理解）+ GPT（单向生成）
  ↓
2020：GPT-3（1750亿参数，少样本学习）
  ↓
2022：ChatGPT（对话式AI，RLHF）
  ↓
2023：GPT-4、LLaMA、Claude、通义千问
  ↓
2024：多模态、Agent、更高效模型
```

### 1.2 主要模型对比

| 模型 | 公司 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-4 | OpenAI | 未公开 | 多模态、强大推理 |
| Claude | Anthropic | 未公开 | 安全、长上下文 |
| LLaMA 3 | Meta | 8B-70B | 开源、可本地部署 |
| 通义千问 | 阿里 | 72B | 中文优秀 |
| 文心一言 | 百度 | 未公开 | 中文理解强 |
| DeepSeek | 深度求索 | 67B | 开源、代码能力强 |

---

## 第二周：LLM工作原理

### 2.1 核心思想：预测下一个词

**一句话**：LLM的本质就是"预测下一个词"

```
输入：今天天气
模型预测：真
输入：今天天气真
模型预测：好
输入：今天天气真好
模型预测：，
...

逐步生成完整句子
```

**类比**：
```
手机输入法：你打"我"，它预测"们/是/的"
LLM：一样的原理，但更强大
```

### 2.2 自回归生成

**数学表达**：
```
P(下一个词 | 前面所有词)

P("好" | "今天天气真") = 0.8
P("差" | "今天天气真") = 0.1
P("不错" | "今天天气真") = 0.05
...
```

**生成过程**：
```python
# 伪代码
def generate(model, prompt, max_length):
    tokens = tokenize(prompt)
    
    for _ in range(max_length):
        # 预测下一个token的概率分布
        logits = model(tokens)
        probs = softmax(logits)
        
        # 采样下一个token
        next_token = sample(probs)
        tokens.append(next_token)
        
        # 如果遇到结束符，停止
        if next_token == EOS:
            break
    
    return detokenize(tokens)
```

### 2.3 Transformer架构回顾

**编码器-解码器 vs 仅解码器**：

```
BERT（编码器）：         GPT（解码器）：
双向理解                 单向生成
适合理解任务             适合生成任务
分类、NER               写作、对话
```

**GPT架构**：
```
输入：Token嵌入 + 位置编码
  ↓
Transformer Block × N
  ├─ Masked Self-Attention（只能看前面）
  ├─ Layer Norm
  ├─ Feed Forward
  └─ Layer Norm
  ↓
输出：下一个token的概率
```

---

## 第三周：Tokenization

### 3.1 什么是Tokenization？

**一句话**：把文字拆成小块（token），让模型能处理。

**类比**：
```
中文：一个字一个字
英文：一个词一个词
Token：可能是词、词的一部分、甚至单个字符

"我爱人工智能" → ["我", "爱", "人工", "智能"]
"I love AI" → ["I", "love", " ", "AI"]
```

### 3.2 常见分词方法

**BPE（Byte Pair Encoding）**：
```
初始：每个字符是一个token
迭代：合并出现频率最高的相邻pair

示例：
"low low low lower" 
→ ["l", "o", "w", " ", "l", "o", "w", ...]
→ 合并高频对：["lo", "w", " ", "lo", "w", ...]
→ 继续合并：["low", " ", "low", " ", "low", "er"]
```

**为什么用BPE？**
- 平衡词汇表大小和序列长度
- 能处理未见过的词（拆成已知的子词）

### 3.3 Python实现

```python
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 编码
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")

# 解码
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# 查看token详情
tokens_with_text = tokenizer.tokenize(text)
print(f"Tokens: {tokens_with_text}")
```

---

## 第四周：预训练

### 4.1 什么是预训练？

**一句话**：在海量文本上训练模型，让它学会语言规律。

**类比**：
```
预训练 = 大学通识教育（学语言、学知识）
微调 = 职业培训（学专业技能）
```

### 4.2 预训练任务

**语言建模（LM）**：
```
输入：The cat sat on the
目标：mat

P("mat" | "The cat sat on the")
```

**掩码语言建模（MLM，BERT用）**：
```
输入：The [MASK] sat on the mat
目标：cat
```

### 4.3 预训练数据

**常见数据集**：
- Common Crawl：互联网网页
- Wikipedia：维基百科
- Books：书籍
- GitHub：代码

**数据处理**：
```
原始数据 → 去重 → 过滤低质量 → 分词 → 训练
```

### 4.4 训练成本

| 模型 | GPU数量 | 训练时间 | 估计成本 |
|------|---------|----------|----------|
| GPT-3 | 10000+ V100 | 数周 | $460万+ |
| LLaMA-65B | 2048 A100 | 21天 | $200万+ |
| LLaMA-7B | 128 A100 | 8天 | $10万 |

---

## 第五周：推理优化

### 5.1 推理挑战

**问题**：
- 模型太大，显存放不下
- 生成太慢，用户体验差
- 成本太高，难以规模化

### 5.2 量化（Quantization）

**核心思想**：用更少的位数表示参数

```
FP32（32位）→ FP16（16位）→ INT8（8位）→ INT4（4位）

精度降低，但模型变小、速度变快
```

**效果**：
```
7B模型，FP16需要14GB显存
INT4量化后只需4GB显存
```

**代码示例**：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 5.3 推理加速

**KV Cache**：
```
传统：每次生成都要重新计算所有token
KV Cache：缓存之前的计算结果，只计算新token

速度提升：10倍以上
```

**Flash Attention**：
```
优化注意力计算，减少显存占用
速度更快，显存更省
```

**批处理（Batching）**：
```
同时处理多个请求
提高GPU利用率
```

---

## 常见问题

### Q1：LLM为什么会产生幻觉？

**原因**：模型只学会了"统计规律"，不理解"真假"

```
训练数据中有："地球是平的"（错误信息）
模型学到："地球"后面可能接"是平的"

模型不知道什么是真的，只知道什么"看起来像真的"
```

### Q2：LLM有意识吗？

**答案**：没有

```
LLM只是在做"下一个词预测"
它不理解文字的含义
只是学会了模式匹配
```

### Q3：如何选择LLM？

**考虑因素**：
| 需求 | 推荐 |
|------|------|
| 最强能力 | GPT-4、Claude |
| 开源可部署 | LLaMA、Qwen |
| 中文任务 | 通义千问、文心一言 |
| 代码任务 | DeepSeek-Coder |
| 低成本 | 小模型 + 量化 |

---

## 学习资源
- Hugging Face课程：https://huggingface.co/learn
- LLM可视化：https://bbycroft.net/llm
- Andrej Karpathy视频：Let's build GPT

## 进入下一阶段
完成LLM基础后，进入[[Prompt工程指南]]。
