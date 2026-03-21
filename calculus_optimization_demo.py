import numpy as np

print("=== 微积分与优化详解 ===\n")

# ========== 1. 导数与梯度 ==========
print("=" * 50)
print("【1. 导数与梯度】")
print("=" * 50)

# 数值导数
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试函数 f(x) = x² + 2x + 1
def f(x):
    return x**2 + 2*x + 1

def f_prime(x):
    return 2*x + 2

x = 2
numerical = numerical_derivative(f, x)
analytical = f_prime(x)

print(f"函数 f(x) = x² + 2x + 1")
print(f"在 x = {x} 处：")
print(f"  数值导数：{numerical:.6f}")
print(f"  解析导数：{analytical}")
print(f"  误差：{abs(numerical - analytical):.2e}")

# 偏导数
print(f"\n偏导数示例：f(x,y) = x² + y²")
def f_2d(x, y):
    return x**2 + y**2

def partial_x(x, y, h=1e-5):
    return (f_2d(x+h, y) - f_2d(x-h, y)) / (2*h)

def partial_y(x, y, h=1e-5):
    return (f_2d(x, y+h) - f_2d(x, y-h)) / (2*h)

x, y = 1, 2
print(f"  ∂f/∂x 在 ({x},{y}) = {partial_x(x, y):.4f} (理论值: {2*x})")
print(f"  ∂f/∂y 在 ({x},{y}) = {partial_y(x, y):.4f} (理论值: {2*y})")

# 梯度
def gradient_2d(f, x, y, h=1e-5):
    df_dx = (f(x+h, y) - f(x-h, y)) / (2*h)
    df_dy = (f(x, y+h) - f(x, y-h)) / (2*h)
    return np.array([df_dx, df_dy])

grad = gradient_2d(f_2d, x, y)
print(f"\n梯度 ∇f = {grad}")
print(f"梯度方向：函数增长最快的方向")
print(f"梯度模长：{np.linalg.norm(grad):.4f} (最大变化率)")

# 链式法则
print(f"\n链式法则示例：")
print(f"  f(g(x)) 其中 g(x) = x², f(u) = sin(u)")
print(f"  f(g(x)) = sin(x²)")
print(f"  (f(g(x)))' = f'(g(x)) × g'(x) = cos(x²) × 2x")

def g(x):
    return x**2

def f_of_g(x):
    return np.sin(g(x))

def chain_rule(x):
    return np.cos(g(x)) * 2*x

x = 1.0
numerical = numerical_derivative(f_of_g, x)
analytical = chain_rule(x)
print(f"  在 x={x}：数值导数={numerical:.6f}, 解析导数={analytical:.6f}")

# ========== 2. 梯度下降 ==========
print("\n" + "=" * 50)
print("【2. 梯度下降法】")
print("=" * 50)

def gradient_descent(f, gradient, start, lr=0.1, n_iter=20):
    x = start
    history = [x]
    for _ in range(n_iter):
        grad = gradient(x)
        x = x - lr * grad
        history.append(x)
    return x, history

# f(x) = x²
def f(x):
    return x**2

def grad_f(x):
    return 2*x

start = 5.0
minimum, history = gradient_descent(f, grad_f, start, lr=0.1, n_iter=10)

print(f"函数 f(x) = x²")
print(f"起始点 x₀ = {start}")
print(f"学习率 α = 0.1")
print(f"\n收敛过程：")
for i, x in enumerate(history[:6]):
    print(f"  第{i}步: x = {x:.6f}, f(x) = {f(x):.6f}")
print(f"  ...")
print(f"  最终: x = {history[-1]:.6f}, f(x) = {f(history[-1]):.6f}")

# 学习率影响
print(f"\n学习率对收敛的影响：")
for lr in [0.01, 0.1, 0.5, 0.9]:
    _, hist = gradient_descent(f, grad_f, start, lr=lr, n_iter=10)
    print(f"  lr={lr:4.2f}: 最终x={hist[-1]:.6f}, 最终f(x)={f(hist[-1]):.6f}")

# ========== 3. 局部最优与全局最优 ==========
print("\n" + "=" * 50)
print("【3. 局部最优与全局最优】")
print("=" * 50)

# 多峰函数
def multi_modal(x):
    return np.sin(x) + 0.1*x**2

def multi_modal_grad(x):
    return np.cos(x) + 0.2*x

# 从不同起点开始
print(f"函数 f(x) = sin(x) + 0.1x² (多峰函数)")
print(f"从不同起点收敛到不同局部最优：")
for start in [-3, 0, 3]:
    minimum, _ = gradient_descent(multi_modal, multi_modal_grad, start, lr=0.1, n_iter=50)
    print(f"  起点 x₀={start:5.1f} → 收敛到 x={minimum:.4f}, f(x)={multi_modal(minimum):.4f}")

# ========== 4. 优化器对比 ==========
print("\n" + "=" * 50)
print("【4. 优化器对比】")
print("=" * 50)

# 模拟不同优化器的行为
print("优化器更新规则对比：")
print("\n1. SGD: θ = θ - α×∇J")
print("   特点：简单直接，但可能震荡")

print("\n2. SGD+Momentum:")
print("   v = β×v + ∇J")
print("   θ = θ - α×v")
print("   特点：加速收敛，减少震荡")

print("\n3. Adam:")
print("   m = β₁×m + (1-β₁)×∇J  (一阶矩，动量)")
print("   v = β₂×v + (1-β₂)×∇J² (二阶矩，自适应学习率)")
print("   θ = θ - α×m̂/√(v̂+ε)")
print("   特点：结合动量和自适应学习率，最常用")

# 简单优化器实现
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        return params - self.lr * grads

class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum * self.v + grads
        return params - self.lr * self.v

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    def update(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# 测试优化器
def rosenbrock(x, y):
    """Rosenbrock函数，经典的优化测试函数"""
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

print(f"\n优化器在Rosenbrock函数上的表现：")
print(f"  f(x,y) = (1-x)² + 100(y-x²)²")
print(f"  全局最优: (1, 1), f=0")

# 运行优化
params = np.array([-1.0, 1.0])
optimizers = {
    'SGD': SGD(lr=0.0001),
    'SGD+Momentum': SGDMomentum(lr=0.0001, momentum=0.9),
    'Adam': Adam(lr=0.01)
}

for name, opt in optimizers.items():
    p = params.copy()
    for _ in range(100):
        grad = rosenbrock_grad(p[0], p[1])
        p = opt.update(p, grad)
    print(f"  {name:15s}: x={p[0]:.4f}, y={p[1]:.4f}, f={rosenbrock(p[0], p[1]):.6f}")

# ========== 5. 学习率调度 ==========
print("\n" + "=" * 50)
print("【5. 学习率调度】")
print("=" * 50)

print("常见学习率调度策略：")

# 步长衰减
def step_decay(initial_lr, decay_rate, decay_steps, epoch):
    return initial_lr * (decay_rate ** (epoch // decay_steps))

print(f"\n1. 步长衰减：lr = lr₀ × decay_rate^(epoch/decay_steps)")
initial_lr = 0.1
for epoch in [0, 10, 20, 30, 40, 50]:
    lr = step_decay(initial_lr, 0.5, 20, epoch)
    print(f"   epoch={epoch:2d}: lr={lr:.6f}")

# 余弦退火
import math
def cosine_annealing(initial_lr, min_lr, epoch, total_epochs):
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

print(f"\n2. 余弦退火：lr从{initial_lr}降到0")
for epoch in [0, 25, 50, 75, 100]:
    lr = cosine_annealing(initial_lr, 0, epoch, 100)
    print(f"   epoch={epoch:3d}: lr={lr:.6f}")

# ========== 6. 凸函数与非凸函数 ==========
print("\n" + "=" * 50)
print("【6. 凸函数与非凸函数】")
print("=" * 50)

print("凸函数定义：f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)")
print("性质：局部最优 = 全局最优")

# 验证凸性
def is_convex_1d(f, x1, x2, n_points=10):
    """通过检查中点值判断凸性"""
    for lam in np.linspace(0, 1, n_points):
        x_mid = lam * x1 + (1-lam) * x2
        f_mid = f(x_mid)
        f_interp = lam * f(x1) + (1-lam) * f(x2)
        if f_mid > f_interp + 1e-6:
            return False
    return True

# 凸函数示例
f_convex = lambda x: x**2
f_non_convex = lambda x: np.sin(x)

print(f"\n凸函数示例 f(x) = x²:")
print(f"  f(0)=0, f(2)=4, f(1)={f_convex(1)}")
print(f"  (f(0)+f(2))/2 = {(f_convex(0)+f_convex(2))/2}")
print(f"  f(1) ≤ (f(0)+f(2))/2? {f_convex(1) <= (f_convex(0)+f_convex(2))/2}")

print(f"\n非凸函数示例 f(x) = sin(x):")
print(f"  f(0)=0, f(π)=0, f(π/2)={f_non_convex(np.pi/2):.4f}")
print(f"  (f(0)+f(π))/2 = {(f_non_convex(0)+f_non_convex(np.pi))/2}")
print(f"  f(π/2) > (f(0)+f(π))/2, 所以不是凸函数")

# ========== 总结 ==========
print("\n" + "=" * 50)
print("【总结】")
print("=" * 50)

print("\n导数与梯度：")
print("  - 导数：函数在某点的瞬时变化率")
print("  - 梯度：指向函数增长最快的方向")
print("  - 链式法则：反向传播的理论基础")

print("\n梯度下降：")
print("  - 基本公式：θ = θ - α×∇J(θ)")
print("  - 学习率：控制更新步长")
print("  - 变种：BGD、SGD、Mini-batch GD")

print("\n优化器：")
print("  - SGD：简单直接")
print("  - SGD+Momentum：加速收敛")
print("  - Adam：结合动量和自适应学习率，最常用")

print("\nAI应用：")
print("  - 神经网络训练：梯度下降 + 反向传播")
print("  - 优化器选择：Adam通常是最安全的选择")
print("  - 学习率调度：训练后期降低学习率")
