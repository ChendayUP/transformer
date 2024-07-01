import torch

# 假设我们有一些梯度
gradients = torch.tensor([30.0, 40.0])
print(f'Original gradients: {gradients}')

# 计算 L2 范数
norm = torch.norm(gradients, p=2)
print(f'Original L2 norm: {norm}')

# 假设 clip 值为 5.0
clip = 5.0

# 如果 L2 范数超过 clip，则进行裁剪
if norm > clip:
    gradients = gradients * (clip / norm)

# 计算裁剪后的 L2 范数
print(f'Gradients after clip: {gradients}')
new_norm = torch.norm(gradients, p=2)
print(f'Clipped L2 norm: {new_norm}')