"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 最后一个维度的均值
        mean = x.mean(-1, keepdim=True)
        # 最后一个维度的方差
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        # 实现了输入张量 x 的归一化操作，使得数据具有零均值和单位方差 
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
