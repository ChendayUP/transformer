import torch
import math

# 空的张量, 意思是对应的内存直接显示, 不做任何初始化
x = torch.empty(3, 4)
print(type(x))
print(x)

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)