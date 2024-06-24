import torch

# 创建一个张量
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# 计算均值和方差
mean = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)

# 进行标准化
eps = 1e-12
out = (x - mean) / torch.sqrt(var + eps)



print("输入张量:\n", x)
print("均值:\n", mean)
print("方差:\n", var)
print("标准化后的张量:\n", out)

# 验算标准化后的张量, 是否均值为 0, 方差为 1
print(out.mean(-1, keepdim=True))
print(out.var(-1, unbiased=False, keepdim=True))

"""
输入张量:
 tensor([[1., 2., 3.],
        [4., 5., 6.]])
均值:
 tensor([[2.],
        [5.]])
方差:
 tensor([[0.6667],
        [0.6667]])
标准化后的张量:
 tensor([[-1.2247,  0.0000,  1.2247],
        [-1.2247,  0.0000,  1.2247]])
"""