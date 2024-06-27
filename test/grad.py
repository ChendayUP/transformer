import torch
print('\n')
# 创建两个需要计算梯度的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, 0.5], requires_grad=True)
print('Gradient of x1:', x.grad)
print('Gradient of w1:', w.grad)
# 计算一个简单的函数 y = x * w
y = x * w

# 计算 z = y 的和
z = y.sum()

# 反向传播，计算梯度
z.backward()

# 打印梯度
print('Gradient of x:', x.grad)
print('Gradient of w:', w.grad)