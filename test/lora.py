import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, original_layer, rank=4):
        super(LoRA, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        
        # 初始化低秩矩阵 A 和 B
        self.A = nn.Parameter(torch.randn(original_layer.weight.size(0), rank))
        self.B = nn.Parameter(torch.randn(rank, original_layer.weight.size(1)))
        
        # 保持原始权重不变
        self.register_buffer('W0', original_layer.weight.data.clone())
    
    def forward(self, x):
        # 计算 LoRA 的权重矩阵
        W = self.W0 + torch.matmul(self.A, self.B)
        return nn.functional.linear(x, W, self.original_layer.bias)

# 示例使用
original_linear = nn.Linear(128, 64)
lora_linear = LoRA(original_linear, rank=8)

# 测试前向传播
input_data = torch.randn(32, 128)
output_data = lora_linear(input_data)
print(output_data.shape)  # torch.Size([32, 64])
