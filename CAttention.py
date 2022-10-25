import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.dropout = nn.Dropout(0.5)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, img, txt): ## 最重要的都是forword函数了
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        qk = self.to_qkv(txt).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)
        # 对tensor张量分块 x :1 197 1024   qkv 最后 是一个元组，tuple，长度是3，每个元素形状：1 197 1024
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # 分成多少个Head,与TRM生成qkv 的方式不同， 要更简单，不需要区分来自Encoder还是Decoder
        attention = torch.matmul(q, k.transpose(-1,-2)) * self.scale
        attn = self.attend(attention)
        out = torch.matmul(attn, img)
        # 矩阵转置,k.transpose
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


if __name__=="__main__":
    input1 = np.random.standard_normal([64,200,768])
    input2 = np.random.standard_normal([64,200,768])
    att = Attention(2048)
    print(att(input1, input2))