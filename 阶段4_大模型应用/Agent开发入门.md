# Agent开发入门（4-5周）

## 学习目标
理解AI Agent概念和架构，能够开发具有工具调用能力的Agent应用。

---

## 什么是AI Agent？

**一句话**：AI Agent = 能思考、能行动的AI助手

**类比**：
```
普通LLM：只能聊天，像一个"只会说话的顾问"
AI Agent：能使用工具，像一个"能干活的助手"

普通LLM："我帮你查一下天气"（但其实查不了）
AI Agent：*调用天气API* "今天北京晴天，25度"
```

**Agent的核心能力**：
1. **思考**：理解任务，制定计划
2. **行动**：使用工具完成任务
3. **记忆**：记住上下文和历史
4. **反思**：检查结果，调整策略

---

## 第一周：Agent架构

### 1.1 Agent核心组件

```
┌─────────────────────────────────────┐
│              Agent                   │
│  ┌─────────────────────────────┐   │
│  │           LLM大脑           │   │
│  │    （思考、推理、决策）       │   │
│  └─────────────────────────────┘   │
│                  ↓                  │
│  ┌─────────┬─────────┬─────────┐   │
│  │  工具1   │  工具2   │  工具3   │   │
│  │ (搜索)   │ (计算)   │ (代码)   │   │
│  └─────────┴─────────┴─────────┘   │
│                  ↓                  │
│  ┌─────────────────────────────┐   │
│  │          记忆模块            │   │
│  │   （短期记忆 + 长期记忆）     │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 1.2 Agent循环

```
1. 观察：接收用户请求
2. 思考：LLM分析任务
3. 行动：调用工具
4. 观察：获取工具结果
5. 思考：决定下一步
6. 行动：继续或结束
7. 输出：返回最终结果
```

### 1.3 ReAct框架

**核心思想**：推理（Reasoning）+ 行动（Acting）

```
问题：北京今天天气怎么样？

思考：我需要查询北京的天气信息
行动：调用天气API，参数：城市=北京
观察：北京今天晴天，25度
思考：我已经获得了天气信息
行动：结束，返回结果
答案：北京今天晴天，温度25度
```

---

## 第二周：工具调用

### 2.1 什么是工具调用？

**一句话**：让LLM能够使用外部工具（API、函数、数据库等）

**类比**：
```
人：遇到数学题，用计算器算
Agent：遇到数学题，调用计算工具

人：需要查资料，用搜索引擎
Agent：需要信息，调用搜索工具
```

### 2.2 工具定义

```python
from langchain.tools import Tool, tool

# 方法1：使用装饰器
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入应该是一个有效的Python表达式。"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息。"""
    # 实际实现会调用搜索API
    return f"搜索结果：关于{query}的信息..."

# 方法2：使用Tool类
weather_tool = Tool(
    name="weather",
    description="获取指定城市的天气信息",
    func=lambda city: f"{city}今天晴天，25度"
)
```

### 2.3 工具调用实现

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 定义工具列表
tools = [calculator, search_web, weather_tool]

# 创建LLM
llm = OpenAI(temperature=0)

# 创建Agent
prompt = PromptTemplate.from_template("""
你是一个有用的AI助手。你可以使用以下工具：

{tools}

使用以下格式：

问题：你必须回答的输入问题
思考：你应该总是思考要做什么
行动：要使用的工具，应该是[{tool_names}]之一
行动输入：工具的输入
观察：工具的结果
... (这个思考/行动/行动输入/观察可以重复N次)
思考：我现在知道最终答案了
最终答案：对原始问题的最终回答

开始！

问题：{input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
result = agent_executor.invoke({"input": "北京今天天气怎么样？"})
print(result["output"])
```

---

## 第三周：LangChain Agent

### 3.1 LangChain Agent类型

| Agent类型 | 特点 | 适用场景 |
|-----------|------|----------|
| ReAct | 推理+行动 | 通用 |
| OpenAI Functions | 使用函数调用 | OpenAI模型 |
| Plan-and-Execute | 先计划再执行 | 复杂任务 |
| Conversational | 对话式 | 聊天机器人 |

### 3.2 使用OpenAI Functions Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 创建LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手。"),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

# 创建Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
result = agent_executor.invoke({"input": "计算 (15 + 27) * 3 等于多少"})
print(result["output"])
```

### 3.3 添加记忆

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建带记忆的Agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 多轮对话
result1 = agent_executor.invoke({"input": "我叫小明"})
result2 = agent_executor.invoke({"input": "我叫什么名字？"})
print(result2["output"])  # 会记住你叫小明
```

---

## 第四周：规划与记忆

### 4.1 规划能力

**什么是规划？**
```
复杂任务：写一份市场分析报告

Agent规划：
1. 搜索市场数据
2. 分析竞争对手
3. 整理用户反馈
4. 生成报告大纲
5. 撰写各部分内容
6. 检查和润色
```

**Plan-and-Execute Agent**：
```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)

# 创建规划器
planner = load_chat_planner(llm)

# 创建执行器
executor = load_agent_executor(llm, tools, verbose=True)

# 创建Agent
agent = PlanAndExecute(planner=planner, executor=executor)

# 运行
result = agent.run("分析一下当前AI行业的发展趋势")
```

### 4.2 记忆机制

**短期记忆**：
```python
# 对话历史
memory = ConversationBufferMemory()
memory.save_context({"input": "你好"}, {"output": "你好！"})
memory.load_memory_variables({})
```

**长期记忆**：
```python
from langchain.memory import ConversationKGMemory

# 知识图谱记忆
memory = ConversationKGMemory(llm=llm)
memory.save_context(
    {"input": "小明在北京工作"},
    {"output": "好的，我记住了"}
)
# 会提取实体关系存储
```

**向量记忆**：
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS

# 创建向量存储
vectorstore = FAISS.from_texts([""], embeddings)
retriever = vectorstore.as_retriever()

# 创建向量记忆
memory = VectorStoreRetrieverMemory(retriever=retriever)
memory.save_context({"input": "我喜欢吃火锅"}, {"output": "记住了"})
```

---

## 第五周：多Agent系统

### 5.1 什么是多Agent？

**一句话**：多个Agent协作完成复杂任务

**类比**：
```
单Agent：一个人完成所有工作
多Agent：一个团队分工合作

比如：
- 研究Agent：负责搜索信息
- 写作Agent：负责撰写内容
- 审核Agent：负责检查质量
```

### 5.2 多Agent架构

```
用户请求
    ↓
┌─────────────┐
│   调度Agent  │  ← 分配任务
└──────┬──────┘
       ↓
┌──────┴──────────────────┐
↓              ↓           ↓
研究Agent    写作Agent    审核Agent
   ↓              ↓           ↓
   └──────────────┴───────────┘
                  ↓
              最终结果
```

### 5.3 实现多Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage

# 研究Agent
research_agent = create_openai_functions_agent(
    llm, [search_web], 
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是研究专家，负责搜索和整理信息。"),
        ("user", "{input}")
    ])
)

# 写作Agent
writing_agent = create_openai_functions_agent(
    llm, [calculator],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是写作专家，负责撰写内容。"),
        ("user", "{input}")
    ])
)

# 调度Agent
def coordinate_agents(task):
    # 1. 研究阶段
    research_result = AgentExecutor(agent=research_agent, tools=[search_web]).invoke(
        {"input": f"研究：{task}"}
    )
    
    # 2. 写作阶段
    writing_result = AgentExecutor(agent=writing_agent, tools=[calculator]).invoke(
        {"input": f"根据以下研究结果写作：{research_result['output']}"}
    )
    
    return writing_result["output"]
```

---

## 完整项目：智能助手

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索结果：关于{query}的最新信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}今天晴天，温度25度，适合出行"

# 创建Agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [search, calculate, get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，可以帮助用户：
1. 搜索信息
2. 计算数学题
3. 查询天气
4. 回答各种问题

请友好、准确地回答用户的问题。"""),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 使用示例
print(agent_executor.invoke({"input": "你好，我叫小明"})["output"])
print(agent_executor.invoke({"input": "北京今天天气怎么样？"})["output"])
print(agent_executor.invoke({"input": "帮我算一下 15 * 23 + 47"})["output"])
print(agent_executor.invoke({"input": "我叫什么名字？"})["output"])
```

---

## 学习资源
- LangChain Agent文档：https://docs.langchain.com/docs/modules/agents
- AutoGPT：https://github.com/Significant-Gravitas/AutoGPT
- CrewAI：https://github.com/joaomdmoura/crewAI

## 进入下一阶段
完成Agent开发后，可以开始实际项目开发。
