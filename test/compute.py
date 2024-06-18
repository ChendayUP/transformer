# tensor([0., 1., 2., 3., 4., 5.], device='cuda:0') / (10000 ** (0.0 / 8)))
import math
print(math.sin(0/ (10000 ** (0.0 / 8))))
print(math.sin(1/ (10000 ** (0.0 / 8))))
print(math.sin(2/ (10000 ** (0.0 / 8))))
print(math.sin(3/ (10000 ** (0.0 / 8))))
print(math.sin(4/ (10000 ** (0.0 / 8))))
print(math.sin(5/ (10000 ** (0.0 / 8))))


# import torch
# max_len = 6
# d_model = 8

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
# sin_input = pos / (10000 ** (0 / d_model))
# print('\n')
# print(sin_input)