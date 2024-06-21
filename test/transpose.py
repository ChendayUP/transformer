import torch
print('\n')
# 创建一个形状为 (2, 3, 4, 5) 的张量
# 解释: (batch_size=2, length=3, n_head=4, d_tensor=5)
x = torch.tensor([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]
])

print("原始张量形状:", x.shape)

# 应用 transpose(1, 2) 操作
y = x.transpose(1, 2)

print("转置后张量形状:", y.shape)

# 让我们看看具体的数值变化
print("\n原始张量的一个切片 (第一个batch):")
print(x[0])

print("\n转置后张量的对应切片 (第一个batch):")
print(y[0])