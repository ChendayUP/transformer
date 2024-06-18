import torch
max_len = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#创建一个 3x4 的全 0 矩阵
# x = torch.zeros(3, 4)
# print('/n')
# print(x)

pos = torch.arange(0, max_len, device=device)
print(pos)
pos = pos.float().unsqueeze(dim=1)
print(pos)
