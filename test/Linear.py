from torch import nn
import torch
print('\n')

model = nn.Linear(2, 2) # 输入特征数为2，输出特征数为2
# 查看模型参数
for param in model.parameters():
    print(param)
# tensor([[ 0.1200, -0.5973],
#     [ 0.2720, -0.7068]], requires_grad=True)
# Parameter containing:
# tensor([ 0.6559, -0.0558], requires_grad=True)
input = torch.Tensor([1500, 3]) # 给一个样本，该样本有2个特征（这两个特征的值分别为1和2）
print('input', input)
output = model(input)

print(output)
# tensor([178.8698, 405.8258], grad_fn=<ViewBackward0>)