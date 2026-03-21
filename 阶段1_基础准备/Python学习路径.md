# Python学习路径（2-3周）

## 学习目标
掌握Python编程基础，能够进行数据处理和可视化。

## 第一周：Python基础语法

### 第1-2天：环境与基础
**学习内容：**
1. Python安装与环境配置
2. 基本数据类型：int, float, str, bool
3. 变量与赋值
4. 基本运算符

**实践任务：**
```python
# 任务1：变量与类型
name = "张三"
age = 25
height = 1.75
is_student = True
print(f"姓名：{name}，年龄：{age}，身高：{height}m")

# 任务2：基本运算
a = 10
b = 3
print(f"加法：{a + b}")
print(f"除法：{a / b}")
print(f"整除：{a // b}")
print(f"取余：{a % b}")
print(f"幂运算：{a ** b}")
```

### 第3-4天：控制结构
**学习内容：**
1. 条件语句：if-elif-else
2. 循环语句：for, while
3. 循环控制：break, continue

**实践任务：**
```python
# 任务1：成绩判断
score = 85
if score >= 90:
    grade = "优秀"
elif score >= 80:
    grade = "良好"
elif score >= 60:
    grade = "及格"
else:
    grade = "不及格"
print(f"成绩：{grade}")

# 任务2：九九乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="  ")
    print()
```

### 第5-7天：数据结构
**学习内容：**
1. 列表：创建、索引、切片、方法
2. 元组：不可变序列
3. 字典：键值对存储
4. 集合：无重复元素

**实践任务：**
```python
# 任务1：列表操作
fruits = ["苹果", "香蕉", "橙子", "葡萄"]
print(f"第一个水果：{fruits[0]}")
print(f"切片：{fruits[1:3]}")
fruits.append("西瓜")
print(f"添加后：{fruits}")

# 任务2：字典操作
student = {
    "name": "李四",
    "age": 20,
    "courses": ["数学", "英语", "计算机"]
}
print(f"学生信息：{student}")
print(f"选修课程：{student['courses']}")
```

## 第二周：函数与面向对象

### 第8-10天：函数
**学习内容：**
1. 函数定义与调用
2. 参数传递：位置参数、默认参数、可变参数
3. 返回值
4. 作用域

**实践任务：**
```python
# 任务1：定义计算函数
def calculate_area(length, width):
    """计算矩形面积"""
    return length * width

def calculate_volume(length, width, height):
    """计算长方体体积"""
    return length * width * height

area = calculate_area(5, 3)
volume = calculate_volume(5, 3, 2)
print(f"面积：{area}，体积：{volume}")

# 任务2：可变参数函数
def calculate_stats(*numbers):
    """计算多个数字的统计信息"""
    if not numbers:
        return None
    return {
        "sum": sum(numbers),
        "avg": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }

stats = calculate_stats(10, 20, 30, 40, 50)
print(f"统计信息：{stats}")
```

### 第11-14天：面向对象编程
**学习内容：**
1. 类与对象
2. 属性与方法
3. 继承与多态
4. 魔术方法

**实践任务：**
```python
# 任务1：定义学生类
class Student:
    def __init__(self, name, age, scores):
        self.name = name
        self.age = age
        self.scores = scores
    
    def get_average(self):
        """计算平均分"""
        return sum(self.scores) / len(self.scores)
    
    def get_grade(self):
        """获取等级"""
        avg = self.get_average()
        if avg >= 90:
            return "优秀"
        elif avg >= 80:
            return "良好"
        elif avg >= 60:
            return "及格"
        else:
            return "不及格"
    
    def __str__(self):
        return f"学生：{self.name}，年龄：{self.age}，平均分：{self.get_average():.1f}"

# 创建学生对象
student = Student("王五", 21, [85, 92, 78, 88, 95])
print(student)
print(f"等级：{student.get_grade()}")
```

## 第三周：数据处理库

### 第15-17天：NumPy基础
**学习内容：**
1. 数组创建与操作
2. 数组运算
3. 索引与切片
4. 广播机制

**实践任务：**
```python
import numpy as np

# 任务1：数组操作
arr = np.array([1, 2, 3, 4, 5])
print(f"数组：{arr}")
print(f"形状：{arr.shape}")
print(f"均值：{np.mean(arr)}")
print(f"标准差：{np.std(arr)}")

# 任务2：矩阵运算
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print(f"矩阵乘法：\n{np.dot(matrix1, matrix2)}")
```

### 第18-20天：Pandas基础
**学习内容：**
1. Series与DataFrame
2. 数据读取与写入
3. 数据筛选与过滤
4. 数据清洗

**实践任务：**
```python
import pandas as pd

# 任务1：创建DataFrame
data = {
    "姓名": ["张三", "李四", "王五", "赵六"],
    "年龄": [25, 30, 22, 35],
    "成绩": [85, 92, 78, 88],
    "城市": ["北京", "上海", "广州", "深圳"]
}
df = pd.DataFrame(data)
print("原始数据：")
print(df)

# 任务2：数据筛选
# 筛选成绩大于80的学生
high_score_students = df[df["成绩"] > 80]
print("\n成绩大于80的学生：")
print(high_score_students)

# 任务3：数据统计
print("\n统计信息：")
print(df.describe())
```

### 第21-24天：数据可视化
**学习内容：**
1. Matplotlib基础
2. 常用图表：折线图、柱状图、散点图
3. Seaborn进阶
4. 图表美化

**实践任务：**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 任务1：折线图
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue')
plt.plot(x, y2, label='cos(x)', color='red')
plt.title('三角函数图')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 任务2：柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='skyblue')
plt.title('分类数据柱状图')
plt.xlabel('类别')
plt.ylabel('数值')
plt.show()
```

## 学习资源

### 书籍推荐
1. 《Python编程：从入门到实践》- 入门经典
2. 《流畅的Python》- 进阶必读
3. 《利用Python进行数据分析》- 数据处理

### 在线课程
1. Python官方教程
2. 廖雪峰Python教程
3. Coursera Python for Everybody

### 练习平台
1. LeetCode Python题库
2. HackerRank Python挑战
3. Kaggle Python入门

## 学习建议

### 每日学习安排
- **理论学习**：30分钟
- **代码实践**：1-2小时
- **项目练习**：30分钟

### 注意事项
1. **多写代码**：编程是技能，需要大量练习
2. **理解原理**：不要死记硬背，理解为什么
3. **项目驱动**：通过实际项目巩固知识
4. **及时复习**：定期回顾学过的内容

### 常见问题
1. **环境问题**：使用虚拟环境避免包冲突
2. **语法错误**：仔细阅读错误信息，学会调试
3. **性能问题**：了解Python的GIL和性能优化

## 进入下一阶段
完成Python学习后，进入[[数学学习路径]]。