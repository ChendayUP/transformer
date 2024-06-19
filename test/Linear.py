from torch import nn
import torch
print('\n')

model = nn.Linear(2, 2) # 输入特征数为2，输出特征数为1
# 查看模型参数
for param in model.parameters():
    print(param)
input = torch.Tensor([1, 2]) # 给一个样本，该样本有2个特征（这两个特征的值分别为1和2）
output = model(input)

print(output)
