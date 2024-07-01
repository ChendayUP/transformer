import torch
loss = torch.randn(2,2)
print(loss)
print(loss[1,1])
print(loss[1,1].item())
# 输出结果
# tensor([[-2.0274,-1.5974]
# [-1.4775,1.9320]])
# tensor(1.9320)
# 1.9319512844085693