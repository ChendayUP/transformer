import torch

# 假设 trg_len = 4
trg_len = 4

# 步骤 1: 创建一个填充了1的方阵
ones_matrix = torch.rand(trg_len, trg_len)
print("Step 1 - Ones matrix:")
print(ones_matrix)

# 步骤 2: 应用 torch.tril 函数
tril_matrix = torch.tril(ones_matrix)
print("\nStep 2 - Lower triangular matrix:")
print(tril_matrix)

# 步骤 3: 转换为 ByteTensor
byte_matrix = tril_matrix.type(torch.ByteTensor)
print("\nStep 3 - ByteTensor:")
print(byte_matrix)

# 注意：.to(self.device) 部分在这里省略，因为它依赖于具体的运行环境

# 打印最终结果的数据类型
print("\nFinal matrix dtype:", byte_matrix.dtype)