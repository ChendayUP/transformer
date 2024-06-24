import torch

x = torch.tensor([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],
    
    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
], dtype=torch.float32)
print('/n')
print(x.shape)  # 输出: torch.Size([2, 3, 4])

result = x.mean(-1, keepdim=True)
print(result)
print(result.shape)

result_no_keepdim = x.mean(-1)
print(result_no_keepdim)
print(result_no_keepdim.shape)  # 输出: torch.Size([2, 3])