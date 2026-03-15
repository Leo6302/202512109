import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

print("=" * 60)
print("Neural Network Learning Process Demonstration")
print("(Pure NumPy - no deep learning framework)")
print("=" * 60)


# ===========================================================
# Activation Functions
# ===========================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


# ===========================================================
# MLP Class (Pure NumPy backpropagation)
# ===========================================================

class MLP:
    """
    Multi-Layer Perceptron with configurable activation and optimizer.
    Supports: sigmoid / tanh / relu activations
    Supports: sgd / momentum / adam optimizers
    """

    def __init__(self, layer_sizes, activation='sigmoid', lr=0.1, optimizer='sgd'):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.lr = lr
        self.optimizer_name = optimizer
        self.losses = []

        if activation == 'sigmoid':
            self.act, self.act_grad = sigmoid, sigmoid_grad
        elif activation == 'tanh':
            self.act, self.act_grad = tanh_act, tanh_grad
        elif activation == 'relu':
            self.act, self.act_grad = relu, relu_grad

        # Weight initialization (He for ReLU, Xavier for others)
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i]) if activation == 'relu' \
                    else np.sqrt(1.0 / layer_sizes[i])
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.b.append(np.zeros((1, layer_sizes[i + 1])))

        # Optimizer state
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb_ = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self.zs, self.acts = [], [X]
        a = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            self.zs.append(z)
            # Hidden layers: chosen activation / Output layer: sigmoid
            a = self.act(z) if i < len(self.W) - 1 else sigmoid(z)
            self.acts.append(a)
        return a

    def backward(self, y):
        m = y.shape[0]
        dWs, dbs = [None] * len(self.W), [None] * len(self.b)

        # Output delta: dMSE/dz = (pred - y) * sigmoid'(z_out)
        delta = (self.acts[-1] - y) * sigmoid_grad(self.zs[-1])

        for i in reversed(range(len(self.W))):
            dWs[i] = self.acts[i].T @ delta / m
            dbs[i] = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.act_grad(self.zs[i - 1])
        return dWs, dbs

    def _update(self, dWs, dbs):
        self.t += 1
        for i in range(len(self.W)):
            if self.optimizer_name == 'sgd':
                self.W[i] -= self.lr * dWs[i]
                self.b[i] -= self.lr * dbs[i]

            elif self.optimizer_name == 'momentum':
                beta = 0.9
                self.vW[i] = beta * self.vW[i] + (1 - beta) * dWs[i]
                self.vb[i] = beta * self.vb[i] + (1 - beta) * dbs[i]
                self.W[i] -= self.lr * self.vW[i]
                self.b[i] -= self.lr * self.vb[i]

            elif self.optimizer_name == 'adam':
                b1, b2, eps = 0.9, 0.999, 1e-8
                self.mW[i]  = b1 * self.mW[i]  + (1 - b1) * dWs[i]
                self.mb_[i] = b1 * self.mb_[i] + (1 - b1) * dbs[i]
                self.vW[i]  = b2 * self.vW[i]  + (1 - b2) * dWs[i] ** 2
                self.vb[i]  = b2 * self.vb[i]  + (1 - b2) * dbs[i] ** 2
                mW_hat = self.mW[i]  / (1 - b1 ** self.t)
                mb_hat = self.mb_[i] / (1 - b1 ** self.t)
                vW_hat = self.vW[i]  / (1 - b2 ** self.t)
                vb_hat = self.vb[i]  / (1 - b2 ** self.t)
                self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            out = self.forward(X)
            loss = float(np.mean((out - y) ** 2))
            self.losses.append(loss)
            dWs, dbs = self.backward(y)
            self._update(dWs, dbs)
        return self.losses

    def predict(self, X):
        return self.forward(X)


# ===========================================================
# Part 1: Activation Functions Visualization
# ===========================================================

print("\n[Part 1] Activation Functions")
print("-" * 40)

x = np.linspace(-5, 5, 300)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Activation Functions & Their Derivatives', fontsize=14)

act_configs = [
    ('Sigmoid\nf(x) = 1/(1+e⁻ˣ)',   sigmoid,     sigmoid_grad,     'blue'),
    ('Tanh\nf(x) = tanh(x)',          tanh_act,    tanh_grad,        'green'),
    ('ReLU\nf(x) = max(0, x)',        relu,        relu_grad,        'red'),
    ('Leaky ReLU\nf(x) = max(αx, x)', leaky_relu, leaky_relu_grad,  'purple'),
]

for ax, (name, fn, gfn, color) in zip(axes.flat, act_configs):
    ax.plot(x, fn(x),  color=color, linewidth=2,   label='f(x)')
    ax.plot(x, gfn(x), color=color, linewidth=1.5, label="f'(x)", linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 2.5)

plt.tight_layout()
path = os.path.join(output_dir, '03_activation_functions.png')
plt.savefig(path)
plt.close()
print(f"  Sigmoid : output (0,1)  - 이진 확률 출력에 적합, vanishing gradient 문제")
print(f"  Tanh    : output (-1,1) - zero-centered, sigmoid보다 빠른 수렴")
print(f"  ReLU    : max(0,x)      - 현재 가장 널리 쓰임, vanishing gradient 해결")
print(f"  Leaky   : max(ax,x)    - 'dying ReLU' 문제 완화")
print(f"  Saved: {path}")


# ===========================================================
# Part 2: XOR Problem - Single Layer vs Multi-Layer
# ===========================================================

print("\n[Part 2] XOR Problem - Single Layer vs Multi-Layer")
print("-" * 40)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

np.random.seed(42)
single = MLP([2, 1], activation='sigmoid', lr=0.5)
single_losses = single.train(X_xor, y_xor, epochs=5000)
single_pred = single.predict(X_xor)
single_acc  = np.mean((single_pred > 0.5) == y_xor) * 100

np.random.seed(42)
multi = MLP([2, 4, 1], activation='sigmoid', lr=0.5)
multi_losses = multi.train(X_xor, y_xor, epochs=5000)
multi_pred = multi.predict(X_xor)
multi_acc  = np.mean((multi_pred > 0.5) == y_xor) * 100

print(f"  Single Layer (2→1)   accuracy: {single_acc:.1f}%  predictions: {single_pred.flatten().round(3)}")
print(f"  Multi-Layer  (2→4→1) accuracy: {multi_acc:.1f}%  predictions: {multi_pred.flatten().round(3)}")
print(f"  True labels: {y_xor.flatten()}")
print(f"  → 선형 분류기는 XOR를 풀 수 없음. 은닉층이 비선형 경계를 만들어야 해결 가능.")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('XOR Problem: Single Layer vs Multi-Layer Network', fontsize=14)

axes[0].plot(single_losses, color='red',  label='Single Layer (2→1)')
axes[0].plot(multi_losses,  color='blue', label='Multi-Layer (2→4→1)')
axes[0].set_title('Training Loss Curve')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
grid   = np.c_[xx.ravel(), yy.ravel()]

for ax, model, title in [
    (axes[1], single, f'Single Layer\nAccuracy: {single_acc:.0f}%'),
    (axes[2], multi,  f'Multi-Layer (2→4→1)\nAccuracy: {multi_acc:.0f}%'),
]:
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.7)
    ax.contour(xx,  yy, zz, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor.flatten(),
               s=200, cmap='RdBu', edgecolors='black', zorder=5)
    for (xi, yi), label in zip(X_xor, y_xor.flatten()):
        ax.text(xi + 0.05, yi + 0.05, str(int(label)), fontsize=12, fontweight='bold')
    ax.set_title(title); ax.set_xlabel('x1'); ax.set_ylabel('x2')

plt.tight_layout()
path = os.path.join(output_dir, '03_xor_problem.png')
plt.savefig(path)
plt.close()
print(f"  Saved: {path}")


# ===========================================================
# Part 3: Learning Rate Effect
# ===========================================================

print("\n[Part 3] Learning Rate Comparison")
print("-" * 40)

lr_configs = [
    (0.001, 'purple', 'Too small  (0.001)'),
    (0.01,  'blue',   'Small      (0.01) '),
    (0.1,   'green',  'Good       (0.1)  '),
    (1.0,   'orange', 'Large      (1.0)  '),
    (5.0,   'red',    'Too large  (5.0)  '),
]

plt.figure(figsize=(10, 6))
for lr, color, label in lr_configs:
    np.random.seed(42)
    model  = MLP([2, 4, 1], activation='sigmoid', lr=lr)
    losses = model.train(X_xor, y_xor, epochs=2000)
    losses_clipped = np.clip(losses, 0, 0.5)
    plt.plot(losses_clipped, color=color, label=label, alpha=0.85)
    print(f"  LR={lr:.3f}: Final Loss = {losses[-1]:.4f}")

plt.title('Effect of Learning Rate on Training (XOR task)')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss (clipped at 0.5)')
plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 0.5)
path = os.path.join(output_dir, '03_learning_rate.png')
plt.savefig(path)
plt.close()
print(f"  → 너무 작으면 학습이 느리고, 너무 크면 발산. 적절한 값이 핵심.")
print(f"  Saved: {path}")


# ===========================================================
# Part 4: Optimizer Comparison (SGD / Momentum / Adam)
# ===========================================================

print("\n[Part 4] Optimizer Comparison - Spiral Dataset")
print("-" * 40)

def make_spiral(n=120, noise=0.08):
    """두 개의 나선(spiral)으로 이루어진 이진 분류 데이터"""
    np.random.seed(42)
    theta = np.linspace(0, 4 * np.pi, n)
    r     = np.linspace(0.15, 1.0, n)
    X0 = np.c_[r * np.cos(theta),          r * np.sin(theta)]          + np.random.randn(n, 2) * noise
    X1 = np.c_[r * np.cos(theta + np.pi),  r * np.sin(theta + np.pi)] + np.random.randn(n, 2) * noise
    X  = np.vstack([X0, X1])
    y  = np.vstack([np.zeros((n, 1)), np.ones((n, 1))])
    return X, y

X_sp, y_sp = make_spiral()

opt_configs = [
    ('SGD',      'sgd',      0.10, 'red'),
    ('Momentum', 'momentum', 0.05, 'blue'),
    ('Adam',     'adam',     0.01, 'green'),
]

EPOCHS = 4000
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Optimizer Comparison on Spiral Classification', fontsize=14)

trained_models = {}
for name, opt, lr, color in opt_configs:
    np.random.seed(42)
    model  = MLP([2, 16, 8, 1], activation='tanh', lr=lr, optimizer=opt)
    losses = model.train(X_sp, y_sp, epochs=EPOCHS)
    acc    = np.mean((model.predict(X_sp) > 0.5) == y_sp) * 100
    axes[0].plot(losses, color=color, label=f'{name} (lr={lr})', alpha=0.85)
    trained_models[name] = (model, acc)
    print(f"  {name:<10}: Final Loss={losses[-1]:.4f}  Accuracy={acc:.1f}%")

axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 250), np.linspace(-1.6, 1.6, 250))
grid   = np.c_[xx.ravel(), yy.ravel()]

for ax, (name, opt, lr, color) in zip(axes[1:], opt_configs):
    model, acc = trained_models[name]
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.7)
    ax.contour(xx,  yy, zz, levels=[0.5], colors='black', linewidths=1.5)
    ax.scatter(X_sp[:, 0], X_sp[:, 1], c=y_sp.flatten(),
               cmap='RdBu', edgecolors='gray', s=20, alpha=0.8)
    ax.set_title(f'{name}\n(Accuracy: {acc:.1f}%)')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')

plt.tight_layout()
path = os.path.join(output_dir, '03_optimizer_comparison.png')
plt.savefig(path)
plt.close()
print(f"  → SGD: 단순하지만 느림 / Momentum: 관성으로 빠른 수렴 / Adam: 적응적 lr, 실전 최강")
print(f"  Saved: {path}")


# ===========================================================
# Summary
# ===========================================================

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
[Part 1] 활성화 함수 (Activation Functions)
  - Sigmoid  : 출력 (0,1). 이진 분류 출력층에 사용. Vanishing gradient 문제.
  - Tanh     : 출력 (-1,1). Zero-centered라 학습이 더 안정적.
  - ReLU     : 현재 가장 많이 쓰이는 활성화 함수. 역전파 시 gradient가 사라지지 않음.
  - Leaky ReLU: ReLU의 'dying neuron' 문제를 작은 기울기 α로 완화.

[Part 2] XOR 문제 - 단층 vs 다층
  - 단층 퍼셉트론: 선형 결정 경계만 만들 수 있어 XOR 해결 불가.
  - 다층 퍼셉트론: 은닉층이 비선형 변환을 학습하여 XOR 해결 가능.
  - 이것이 딥러닝에서 여러 레이어를 쌓는 핵심 이유.

[Part 3] 학습률 (Learning Rate)
  - 너무 작음: 학습이 매우 느려짐.
  - 너무 큼 : 손실이 발산하거나 진동함.
  - 적절한 값: 빠르고 안정적인 수렴. (보통 1e-4 ~ 1e-1 범위 탐색)

[Part 4] 옵티마이저 (Optimizer)
  - SGD      : 가장 기본적인 경사 하강법. 학습률에 민감.
  - Momentum : 이전 업데이트 방향을 기억해 수렴 가속 및 local minima 탈출.
  - Adam     : 파라미터별 adaptive 학습률. 실전에서 가장 많이 사용.
""")
print("=" * 60)
print("Output files saved to:", output_dir)
