# RAG应用开发（3-4周）

## 学习目标
掌握检索增强生成（RAG）原理，能够构建基于知识库的问答系统。

---

## 什么是RAG？

**一句话**：RAG = 先检索相关文档，再让LLM根据文档生成答案

**类比**：
```
普通LLM：开卷考试，但只能凭记忆答题
RAG：开卷考试，可以翻书找答案

RAG让LLM能"查资料"再回答
```

**为什么需要RAG？**
- LLM知识有截止日期
- LLM可能产生幻觉
- 企业有私有数据需要利用
- 比微调更简单、更可控

---

## 第一周：RAG原理

### 1.1 RAG架构

```
用户问题
    ↓
┌─────────────────┐
│    检索器        │  ← 从知识库中找到相关文档
└────────┬────────┘
         ↓
    相关文档片段
         ↓
┌─────────────────┐
│    生成器        │  ← LLM根据文档生成答案
└────────┬────────┘
         ↓
      最终答案
```

### 1.2 RAG vs 微调

| 特性 | RAG | 微调 |
|------|-----|------|
| 知识更新 | 实时更新 | 需要重新训练 |
| 幻觉风险 | 低（有依据） | 中等 |
| 实现难度 | 中等 | 较高 |
| 成本 | 较低 | 较高 |
| 适用场景 | 知识问答 | 风格/能力调整 |

### 1.3 RAG工作流程

```
1. 文档加载：读取PDF、Word、网页等
2. 文档分块：切分成小段落
3. 向量化：用Embedding模型转换为向量
4. 存储：存入向量数据库
5. 检索：根据问题找到相关文档
6. 生成：LLM根据检索结果生成答案
```

---

## 第二周：文档处理

### 2.1 文档加载

```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)

# 加载PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 加载目录下所有txt文件
loader = DirectoryLoader("./docs", glob="*.txt")
documents = loader.load()

# 加载Markdown
loader = UnstructuredMarkdownLoader("readme.md")
documents = loader.load()
```

### 2.2 文档分块

**为什么分块？**
- Embedding模型有长度限制
- 小块更精确匹配
- 减少噪声

**分块策略**：

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

# 按字符分块
text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # 每块1000字符
    chunk_overlap=200,    # 重叠200字符
    separator="\n"
)

# 递归分块（推荐）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", ".", " "]
)

chunks = text_splitter.split_documents(documents)
```

**分块大小选择**：
| 文档类型 | 推荐块大小 | 重叠大小 |
|----------|------------|----------|
| 一般文档 | 500-1000字符 | 100-200 |
| 代码 | 300-500字符 | 50-100 |
| 对话 | 200-500字符 | 50-100 |

### 2.3 文本清洗

```python
def clean_text(text):
    # 移除多余空白
    text = " ".join(text.split())
    # 移除特殊字符
    text = text.replace("\x00", "")
    # 移除页眉页脚
    lines = text.split("\n")
    lines = [l for l in lines if not l.strip().isdigit()]  # 移除纯数字行
    return "\n".join(lines)
```

---

## 第三周：向量数据库

### 3.1 什么是向量数据库？

**一句话**：存储和检索向量的数据库

**类比**：
```
传统数据库：按关键词搜索（精确匹配）
向量数据库：按含义搜索（语义匹配）

搜索"苹果"：
传统数据库：只返回包含"苹果"的结果
向量数据库：还返回"水果"、"iPhone"等相关结果
```

### 3.2 Embedding模型

**什么是Embedding？**
```
文字 → 向量（数字数组）

"你好" → [0.2, -0.5, 0.8, ...]  （768维或更高）
"Hello" → [0.3, -0.4, 0.7, ...]  （相似的向量）
```

**常用Embedding模型**：
| 模型 | 维度 | 特点 |
|------|------|------|
| text-embedding-ada-002 | 1536 | OpenAI，效果好 |
| BGE | 768/1024 | 中文优秀 |
| M3E | 768 | 中文开源 |
| all-MiniLM-L6-v2 | 384 | 轻量快速 |

```python
from langchain.embeddings import HuggingFaceEmbeddings

# 使用开源Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 或使用OpenAI
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

### 3.3 向量数据库选择

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| FAISS | 本地、快速 | 小规模、实验 |
| Chroma | 简单易用 | 原型开发 |
| Pinecone | 云服务 | 生产环境 |
| Milvus | 开源、可扩展 | 大规模部署 |
| Weaviate | 功能丰富 | 复杂需求 |

### 3.4 FAISS使用

```python
from langchain.vectorstores import FAISS

# 创建向量数据库
vectorstore = FAISS.from_documents(chunks, embeddings)

# 保存
vectorstore.save_local("faiss_index")

# 加载
vectorstore = FAISS.load_local("faiss_index", embeddings)

# 检索
docs = vectorstore.similarity_search("什么是机器学习？", k=3)
```

### 3.5 Chroma使用

```python
from langchain.vectorstores import Chroma

# 创建
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 检索
docs = vectorstore.similarity_search("什么是深度学习？", k=3)
```

---

## 第四周：RAG系统构建

### 4.1 完整RAG流程

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 2. 创建LLM
llm = OpenAI(temperature=0)

# 3. 创建Prompt模板
prompt_template = """使用以下上下文回答问题。如果不知道答案，就说不知道，不要编造。

上下文：
{context}

问题：{question}

回答："""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 4. 创建RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 5. 查询
result = qa_chain.run("什么是机器学习？")
print(result)
```

### 4.2 高级检索技巧

**混合检索**：
```python
# 结合关键词和语义检索
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# BM25（关键词）
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

**重排序**：
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 使用重排序模型
compressor = CrossEncoderReranker()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

docs = compression_retriever.get_relevant_documents("问题")
```

### 4.3 对话式RAG

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 添加记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建对话式RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 多轮对话
result1 = qa_chain({"question": "什么是机器学习？"})
result2 = qa_chain({"question": "它和深度学习有什么区别？"})
```

---

## 完整项目：知识库问答系统

```python
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class KnowledgeQA:
    def __init__(self, docs_path, index_path="faiss_index"):
        self.docs_path = docs_path
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        self.llm = OpenAI(temperature=0)
        
        # 加载或创建向量库
        if os.path.exists(index_path):
            self.vectorstore = FAISS.load_local(index_path, self.embeddings)
        else:
            self._build_index()
        
        self._setup_chain()
    
    def _build_index(self):
        """构建向量索引"""
        print("加载文档...")
        loader = DirectoryLoader(self.docs_path, glob="**/*.txt")
        documents = loader.load()
        
        print("分块...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        print("创建向量索引...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.index_path)
        print("索引创建完成！")
    
    def _setup_chain(self):
        """设置问答链"""
        prompt_template = """基于以下上下文回答问题。如果答案不在上下文中，说"我不确定"。

上下文：
{context}

问题：{question}

回答："""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question):
        """查询"""
        return self.qa_chain.run(question)

# 使用
qa = KnowledgeQA("./documents")
answer = qa.query("什么是机器学习？")
print(answer)
```

---

## 学习资源
- LangChain文档：https://docs.langchain.com/
- FAISS文档：https://faiss.ai/
- 向量数据库对比：https://benchmark.vectorview.ai/

## 进入下一阶段
完成RAG后，进入[[Agent开发入门]]。
