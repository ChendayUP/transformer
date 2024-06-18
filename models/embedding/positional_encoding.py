"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        # torch.arange是用于生成一个一维张量，包含从起始值到结束值（不包括结束值）的等间隔值。
        pos = torch.arange(0, max_len, device=device)
        # tensor([0, 1, 2], device='cuda:0')

        # unsqueeze用于在指定维度上增加一个新的维度。
        pos = pos.float().unsqueeze(dim=1) 
        # tensor([[0.],
        #         [1.],
        #         [2.]], device='cuda:0')
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        # 每个位置上的的单词, 一个词的偶数位和奇数位对应的数据计算sin和cos, 所以计算出来的数据, 在特定的位置上面是固定的, max_len和d_model都是固定的
        #  torch.sin(pos / (10000 ** (_2i / d_model))) 
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        #  torch.cos(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
