import torch
import torch.nn as nn

# 假设我们有一个填充标记的索引
src_pad_idx = 0

# 创建交叉熵损失函数，忽略填充标记
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# 假设我们有一些预测值和目标值
predictions = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.1], [0.1, 0.2, 0.7]])
targets = torch.tensor([0, 1, 0])  # 真实标签

# 计算损失
loss = criterion(predictions, targets)
print(loss)