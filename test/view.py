import torch

x = torch.ones(2,2,2)

print('x.size', x.size())

y = x.view(-1, 4)
print('y.size', y.size())