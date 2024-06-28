import torch

# 使用 kaiming_uniform 创建一个新的初始化张量
tensor = torch.nn.init.kaiming_uniform_(torch.empty(3, 5), a=0, mode='fan_in', nonlinearity='relu')

print(tensor)  # tensor 是一个新的张量，已经被初始化

print(tensor.size())

print(tensor.numel())